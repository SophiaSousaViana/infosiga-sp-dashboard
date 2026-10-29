"""
ETL dos microdados do Infosiga SP (2021–2025).

Lê os CSVs disponíveis em data/raw, valida e padroniza os campos
essenciais, gera as bases Parquet do dashboard e do aprendizado de máquina
e registra um relatório de qualidade dos dados.

Dependências principais:
    pandas
    numpy
    pyarrow (ou outro mecanismo Parquet compatível com pandas)
"""
import glob
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ANO_INICIAL = 2021
ANO_FINAL = 2025
PASTA_RAW = Path("data") / "raw"
PASTA_PROCESSADOS = Path("data") / "processed"
ARQUIVO_DASHBOARD = PASTA_PROCESSADOS / "infosiga_dashboard_2021_2025.parquet"
ARQUIVO_ML = PASTA_PROCESSADOS / "infosiga_ml_2021_2025.parquet"
ARQUIVO_RELATORIO = PASTA_PROCESSADOS / "relatorio_qualidade_etl.txt"
COLUNAS_OBRIGATORIAS = ["data_sinistro", "tipo_registro"]
COLUNAS_TEXTO = [
    "tipo_registro",
    "tp_sinistro_primario",
    "tipo_via",
    "tipo_local",
    "administracao",
    "circunscricao",
    "municipio",
    "turno",
]
TIPOS_CONFIRMADOS = {"SINISTRO FATAL", "SINISTRO NAO FATAL"}
COLUNAS_EXCLUIR_ML = [
    "id_sinistro",
    "tipo_registro",
    "qtd_gravidade_fatal",
    "qtd_gravidade_grave",
    "qtd_gravidade_leve",
    "qtd_gravidade_ileso",
    "qtd_gravidade_nao_disponivel",
]
def imprimir_secao(titulo):
    print("\n" + "=" * 72)
    print(titulo)
    print("=" * 72)
def normalizar_texto(serie):
    """Remove espaços externos e padroniza texto sem converter NA em string."""
    resultado = serie.astype("string").str.strip().str.upper()
    resultado = resultado.replace(
        {
            "": pd.NA,
            "NAN": pd.NA,
            "NONE": pd.NA,
            "NULL": pd.NA,
            "N/A": pd.NA,
            "NA": pd.NA,
        }
    )
    return resultado
def ler_csv_com_validacao(caminho):
    """Lê um CSV usando o padrão esperado e informa falhas claramente."""
    try:
        df = pd.read_csv(
            caminho,
            sep=";",
            encoding="cp1252",
            low_memory=False,
        )
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(
                caminho,
                sep=";",
                encoding="utf-8-sig",
                low_memory=False,
            )
        except Exception as erro:
            raise RuntimeError(
                f"Não foi possível ler '{caminho}' em cp1252 ou utf-8-sig: {erro}"
            ) from erro
    except pd.errors.EmptyDataError as erro:
        raise RuntimeError(f"O arquivo '{caminho}' está vazio.") from erro
    except Exception as erro:
        raise RuntimeError(f"Falha ao ler '{caminho}': {erro}") from erro
    if df.empty:
        print(f"AVISO: '{caminho}' foi lido, mas não contém registros.")
    df.columns = [str(coluna).strip() for coluna in df.columns]
    if df.columns.duplicated().any():
        duplicadas = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(
            f"O arquivo '{caminho}' contém nomes de colunas duplicados: {duplicadas}"
        )
    return df
def converter_data(serie):
    """Converte datas; valores não interpretáveis tornam-se NaT."""
    return pd.to_datetime(serie, errors="coerce", dayfirst=True)
def validar_coordenadas(df):
    """
    Converte latitude/longitude para valores numéricos quando presentes.
    Não exclui linhas da base principal por falta de coordenadas; registra
    os problemas para que análises não espaciais continuem disponíveis.
    """
    if not {"latitude", "longitude"}.issubset(df.columns):
        ausentes = [c for c in ("latitude", "longitude") if c not in df.columns]
        return df, {
            "colunas_coordenadas_ausentes": ausentes,
            "coordenadas_ausentes": None,
            "coordenadas_fora_faixa_global": None,
            "coordenadas_fora_limites_aproximados_sp": None,
        }
    for coluna in ("latitude", "longitude"):
        valores = (
            df[coluna]
            .astype("string")
            .str.strip()
            .str.replace(",", ".", regex=False)
        )
        df[coluna] = pd.to_numeric(valores, errors="coerce")
    ausentes = df["latitude"].isna() | df["longitude"].isna()
    fora_global = (
        ~df["latitude"].between(-90, 90)
        | ~df["longitude"].between(-180, 180)
    ) & ~ausentes
    fora_sp = (
        ~df["latitude"].between(-26, -19)
        | ~df["longitude"].between(-54, -44)
    ) & ~ausentes & ~fora_global
    return df, {
        "colunas_coordenadas_ausentes": [],
        "coordenadas_ausentes": int(ausentes.sum()),
        "coordenadas_fora_faixa_global": int(fora_global.sum()),
        "coordenadas_fora_limites_aproximados_sp": int(fora_sp.sum()),
    }
def salvar_relatorio(linhas):
    PASTA_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    with open(ARQUIVO_RELATORIO, "w", encoding="utf-8") as arquivo:
        arquivo.write("RELATÓRIO DE QUALIDADE — ETL INFOSIGA SP\n")
        arquivo.write(f"Período analisado: {ANO_INICIAL}–{ANO_FINAL}\n\n")
        for linha in linhas:
            arquivo.write(f"{linha}\n")
    print(f"\nRelatório de qualidade salvo em: {ARQUIVO_RELATORIO}")
def executar_etl():
    imprimir_secao(f"INICIANDO ETL DO INFOSIGA: {ANO_INICIAL}–{ANO_FINAL}")
    arquivos = sorted(glob.glob(str(PASTA_RAW / "*.csv")))
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum CSV foi encontrado em '{PASTA_RAW}'. "
            "Verifique a pasta e o diretório em que o comando foi executado."
        )
    lista_df = []
    linhas_relatorio = []
    total_por_arquivo = {}
    for arquivo in arquivos:
        print(f"Lendo: {arquivo}")
        df_temp = ler_csv_com_validacao(arquivo)
        total_por_arquivo[arquivo] = len(df_temp)
        lista_df.append(df_temp)
        print(f"  Registros: {len(df_temp):,} | Colunas: {len(df_temp.columns)}")
    df_bruto = pd.concat(lista_df, ignore_index=True, sort=False)
    total_bruto = len(df_bruto)
    imprimir_secao("VALIDAÇÃO E TRATAMENTO")
    print(f"Registros brutos carregados: {total_bruto:,}")
    faltantes = [c for c in COLUNAS_OBRIGATORIAS if c not in df_bruto.columns]
    if faltantes:
        raise ValueError(
            "A base não contém as colunas obrigatórias: "
            + ", ".join(faltantes)
            + ". Confira os nomes das colunas dos CSVs."
        )
    df_bruto["data_sinistro"] = converter_data(df_bruto["data_sinistro"])
    datas_invalidas = int(df_bruto["data_sinistro"].isna().sum())
    df_bruto["ano_sinistro"] = df_bruto["data_sinistro"].dt.year
    fora_periodo = int(
        (~df_bruto["ano_sinistro"].between(ANO_INICIAL, ANO_FINAL)).sum()
    )
    df_tratado = df_bruto[
        df_bruto["ano_sinistro"].between(ANO_INICIAL, ANO_FINAL)
    ].copy()
    for coluna in COLUNAS_TEXTO:
        if coluna in df_tratado.columns:
            df_tratado[coluna] = normalizar_texto(df_tratado[coluna])
    df_tratado, relatorio_coordenadas = validar_coordenadas(df_tratado)
    coluna_anomala = "tp_sinistro_colisao_traseira"
    coluna_anomala_removida = coluna_anomala in df_tratado.columns
    if coluna_anomala_removida:
        df_tratado = df_tratado.drop(columns=[coluna_anomala])
    duplicatas_exatas = int(df_tratado.duplicated().sum())
    ausentes_tipo = int(df_tratado["tipo_registro"].isna().sum())
    print(f"Datas inválidas/ausentes na base bruta: {datas_invalidas:,}")
    print(f"Registros fora do período ou sem data válida: {fora_periodo:,}")
    print(
        f"Registros após filtro temporal ({ANO_INICIAL}–{ANO_FINAL}): "
        f"{len(df_tratado):,}"
    )
    print(f"Linhas exatamente duplicadas na base tratada (diagnóstico): {duplicatas_exatas:,}")
    print(f"Registros sem tipo_registro na base tratada: {ausentes_tipo:,}")
    imprimir_secao("DISTRIBUIÇÃO DOS TIPOS DE REGISTRO")
    print(df_tratado["tipo_registro"].value_counts(dropna=False).to_string())
    PASTA_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    try:
        df_tratado.to_parquet(ARQUIVO_DASHBOARD, index=False)
    except ImportError as erro:
        raise RuntimeError(
            "Não foi possível salvar Parquet. Instale um mecanismo compatível, "
            "por exemplo: pip install pyarrow"
        ) from erro
    print(f"\nBase do dashboard salva: {ARQUIVO_DASHBOARD}")
    df_ml = df_tratado[
        df_tratado["tipo_registro"].isin(TIPOS_CONFIRMADOS)
    ].copy()
    if df_ml.empty:
        raise ValueError(
            "Nenhum registro fatal/não fatal foi encontrado para a base de ML."
        )
    df_ml["target_fatal"] = (
        df_ml["tipo_registro"] == "SINISTRO FATAL"
    ).astype("int8")
    colunas_remover = [
        coluna for coluna in COLUNAS_EXCLUIR_ML if coluna in df_ml.columns
    ]
    df_ml = df_ml.drop(columns=colunas_remover)
    classes = set(df_ml["target_fatal"].unique().tolist())
    if classes != {0, 1}:
        raise ValueError(
            "A base de ML precisa conter as duas classes (0 e 1). "
            f"Classes encontradas: {sorted(classes)}"
        )
    try:
        df_ml.to_parquet(ARQUIVO_ML, index=False)
    except ImportError as erro:
        raise RuntimeError(
            "Não foi possível salvar a base de ML em Parquet. "
            "Instale um mecanismo compatível, por exemplo: pip install pyarrow"
        ) from erro
    print(f"Base de ML salva: {ARQUIVO_ML}")
    print(f"Registros confirmados para ML: {len(df_ml):,}")
    imprimir_secao("DISTRIBUIÇÃO DO ALVO DE ML")
    nomes_classes = {0: "Não fatal (0)", 1: "Fatal (1)"}
    quantidades = df_ml["target_fatal"].value_counts().sort_index()
    percentuais = (
        df_ml["target_fatal"].value_counts(normalize=True).sort_index() * 100
    )
    for classe in (0, 1):
        quantidade = int(quantidades.get(classe, 0))
        percentual = float(percentuais.get(classe, 0.0))
        print(f"{nomes_classes[classe]}: {quantidade:,} ({percentual:.2f}%)")
    linhas_relatorio.extend([
        f"Arquivos CSV lidos: {len(arquivos)}",
        f"Registros brutos carregados: {total_bruto}",
        f"Registros por arquivo: {total_por_arquivo}",
        f"Datas inválidas/ausentes na base bruta: {datas_invalidas}",
        f"Registros fora do período ou sem data válida: {fora_periodo}",
        f"Registros no período selecionado: {len(df_tratado)}",
        f"Registros sem tipo_registro: {ausentes_tipo}",
        f"Linhas exatamente duplicadas (diagnóstico, não removidas): {duplicatas_exatas}",
        f"Coluna anômala removida: {coluna_anomala_removida}",
        f"Diagnóstico de coordenadas: {relatorio_coordenadas}",
        f"Colunas excluídas da base de ML: {colunas_remover}",
        f"Registros na base de ML: {len(df_ml)}",
        f"Distribuição do alvo: {quantidades.to_dict()}",
        f"Percentuais do alvo: {(percentuais.round(2)).to_dict()}",
        f"Arquivo dashboard: {ARQUIVO_DASHBOARD}",
        f"Arquivo ML: {ARQUIVO_ML}",
    ])
    salvar_relatorio(linhas_relatorio)
    imprimir_secao("ETL CONCLUÍDO COM SUCESSO")
if __name__ == "__main__":
    try:
        executar_etl()
    except Exception as erro:
        print(f"\nERRO NO ETL: {erro}", file=sys.stderr)
        raise
