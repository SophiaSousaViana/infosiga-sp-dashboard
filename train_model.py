"""
Treinamento e avaliação dos modelos de classificação de fatalidade do Infosiga.

Gera:
- models/random_forest_model.pkl: pipeline final da Random Forest;
- data/processed/metricas_modelos.json: métricas e matrizes de confusão dos três modelos;
- data/processed/relatorio_modelos.txt: relatório legível da avaliação.

Observação metodológica:
A divisão treino/teste é estratificada e aleatória. Antes de interpretar os resultados,
verifique se o conjunto contém múltiplas linhas do mesmo sinistro e se as variáveis
preditoras estão disponíveis no momento em que a ocorrência seria analisada.
"""
import json
import os
from copy import deepcopy
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
CAMINHO_ML = "data/processed/infosiga_ml_2021_2025.parquet"
CAMINHO_METRICAS = "data/processed/metricas_modelos.json"
CAMINHO_RELATORIO = "data/processed/relatorio_modelos.txt"
CAMINHO_MODELO_RF = "models/random_forest_model.pkl"
COLUNAS_PREDITORAS = [
    "tp_sinistro_primario",
    "tipo_via",
    "tipo_local",
    "administracao",
    "circunscricao",
    "turno",
]
COLUNAS_EXCLUIR_POR_SEGURANCA = {
    "target_fatal",
    "id_sinistro",
    "tipo_registro",
    "fatal",
    "obito",
    "morte",
    "qtd_obitos",
    "qtd_vitimas_fatais",
}
def carregar_e_preparar_dados(caminho_parquet):
    """Carrega a base e valida o alvo e as variáveis disponíveis."""
    df = pd.read_parquet(caminho_parquet)
    if "target_fatal" not in df.columns:
        raise ValueError("A coluna obrigatória 'target_fatal' não existe no Parquet.")
    df = df.loc[df["target_fatal"].notna()].copy()
    valores_alvo = set(pd.to_numeric(df["target_fatal"], errors="coerce").dropna().unique())
    if not valores_alvo.issubset({0, 1}) or not valores_alvo:
        raise ValueError(
            "A coluna 'target_fatal' deve conter somente 0 (não fatal) e 1 (fatal)."
        )
    df["target_fatal"] = pd.to_numeric(df["target_fatal"], errors="raise").astype(int)
    y = df["target_fatal"]
    colunas_faltantes = [c for c in COLUNAS_PREDITORAS if c not in df.columns]
    if colunas_faltantes:
        raise ValueError(
            "Colunas preditoras esperadas ausentes: "
            + ", ".join(colunas_faltantes)
            + ". Confira o ETL e o dicionário de dados do Infosiga."
        )
    X = df[COLUNAS_PREDITORAS].copy()
    X = X.replace(r"^\s*$", np.nan, regex=True)
    X = X.astype("object").where(X.notna(), np.nan)
    duplicatas = int(df.duplicated().sum())
    diagnostico = {
        "total_registros": int(len(df)),
        "total_nao_fatais": int((y == 0).sum()),
        "total_fatais": int((y == 1).sum()),
        "duplicatas_exatas_na_base_ml": duplicatas,
        "colunas_preditoras": list(X.columns),
        "distribuicao_alvo_percentual": {
            "nao_fatal": round(float((y == 0).mean() * 100), 4),
            "fatal": round(float((y == 1).mean() * 100), 4),
        },
    }
    return X, y, diagnostico
def criar_preprocessador(colunas):
    """Cria o preprocessador categórico usado dentro de cada pipeline."""
    pipeline_categorico = Pipeline(steps=[
        ("imputador", SimpleImputer(strategy="constant", fill_value="NAO_INFORMADO")),
        ("codificador", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        transformers=[("categoricas", pipeline_categorico, list(colunas))],
        remainder="drop",
    )
def criar_modelos():
    return {
        "Regressão Logística": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Árvore de Decisão": DecisionTreeClassifier(
            max_depth=10, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }
def avaliar_modelo(nome, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    matriz = confusion_matrix(y_test, y_pred, labels=[0, 1])
    relatorio_dict = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Não fatal", "Fatal"],
        output_dict=True,
        zero_division=0,
    )
    precisao, recall, f1, suporte = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1], zero_division=0
    )
    return {
        "nome": nome,
        "acuracia": float(accuracy_score(y_test, y_pred)),
        "classes": ["Não fatal (0)", "Fatal (1)"],
        "matriz_confusao": matriz.astype(int).tolist(),
        "metricas_por_classe": {
            "nao_fatal": {
                "precisao": float(precisao[0]),
                "recall": float(recall[0]),
                "f1_score": float(f1[0]),
                "suporte": int(suporte[0]),
            },
            "fatal": {
                "precisao": float(precisao[1]),
                "recall": float(recall[1]),
                "f1_score": float(f1[1]),
                "suporte": int(suporte[1]),
            },
        },
        "media_macro": {
            chave: float(relatorio_dict["macro avg"][chave])
            for chave in ("precision", "recall", "f1-score")
        },
        "media_ponderada": {
            chave: float(relatorio_dict["weighted avg"][chave])
            for chave in ("precision", "recall", "f1-score")
        },
        "relatorio_classificacao": relatorio_dict,
    }
def formatar_relatorio(diagnostico, resultados, observacoes):
    linhas = [
        "RELATÓRIO DE TREINAMENTO E AVALIAÇÃO — INFOSIGA",
        "=" * 64,
        f"Registros utilizados: {diagnostico['total_registros']:,}".replace(",", "."),
        f"Não fatais: {diagnostico['total_nao_fatais']:,}".replace(",", "."),
        f"Fatais: {diagnostico['total_fatais']:,}".replace(",", "."),
        f"Duplicatas exatas na base ML: {diagnostico['duplicatas_exatas_na_base_ml']}",
        "Variáveis: " + ", ".join(diagnostico["colunas_preditoras"]),
        "",
        "AVALIAÇÃO NO CONJUNTO DE TESTE",
        "Nota: matriz na ordem [não fatal, fatal], linhas = classe real, colunas = prevista.",
        "",
    ]
    for resultado in resultados:
        fatal = resultado["metricas_por_classe"]["fatal"]
        linhas.extend([
            resultado["nome"],
            f"  Acurácia: {resultado['acuracia'] * 100:.2f}%",
            f"  Precisão fatal: {fatal['precisao'] * 100:.2f}%",
            f"  Recall fatal: {fatal['recall'] * 100:.2f}%",
            f"  F1 fatal: {fatal['f1_score']:.3f}",
            "  Matriz de confusão:",
            "  " + str(resultado["matriz_confusao"]),
            "",
        ])
    linhas.extend(["OBSERVAÇÕES METODOLÓGICAS", *[f"- {o}" for o in observacoes], ""])
    return "\n".join(linhas)
def treinar_e_avaliar_modelos():
    if not os.path.exists(CAMINHO_ML):
        raise FileNotFoundError(
            f"Arquivo '{CAMINHO_ML}' não encontrado. Execute etl.py primeiro."
        )
    X, y, diagnostico = carregar_e_preparar_dados(CAMINHO_ML)
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("A base precisa conter as duas classes e ao menos dois exemplos por classe.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    resultados = []
    pipeline_rf_final = None
    observacoes = [
        "A divisão treino/teste é aleatória e estratificada; se houver várias linhas do mesmo sinistro, a avaliação pode ser otimista.",
        "A variável 'tp_sinistro_primario' foi mantida conforme a lista original, mas seu possível vínculo com a fatalidade deve ser conferido no dicionário do Infosiga.",
        "O conjunto é desbalanceado; interpretar precisão, recall e F1 da classe fatal, não apenas acurácia.",
        "Os resultados avaliam classificação de registros na base, não previsão determinística de acidentes futuros.",
    ]
    for nome_modelo, modelo in criar_modelos().items():
        print(f"Treinando: {nome_modelo}...")
        pipeline = Pipeline(steps=[
            ("preprocessamento", criar_preprocessador(X.columns)),
            ("modelo", modelo),
        ])
        pipeline.fit(X_train, y_train)
        resultado = avaliar_modelo(nome_modelo, pipeline, X_test, y_test)
        resultados.append(resultado)
        fatal = resultado["metricas_por_classe"]["fatal"]
        print(f"\nResultados — {nome_modelo}:")
        print(f"Acurácia geral: {resultado['acuracia'] * 100:.2f}%")
        print(f"Precisão da classe fatal: {fatal['precisao'] * 100:.2f}%")
        print(f"Recall da classe fatal: {fatal['recall'] * 100:.2f}%")
        print(f"F1-score da classe fatal: {fatal['f1_score']:.3f}")
        print("\nMatriz de confusão [linhas = real; colunas = prevista]:")
        print(np.array(resultado["matriz_confusao"]))
        print("\nRelatório de classificação:")
        print(classification_report(
            y_test, pipeline.predict(X_test), labels=[0, 1],
            target_names=["Não Fatal (0)", "Fatal (1)"], zero_division=0
        ))
        print("-" * 50)
        if nome_modelo == "Random Forest":
            pipeline_rf_final = pipeline
    os.makedirs(os.path.dirname(CAMINHO_METRICAS), exist_ok=True)
    os.makedirs(os.path.dirname(CAMINHO_MODELO_RF), exist_ok=True)
    dados_metricas = {
        "versao": 1,
        "alvo": "target_fatal",
        "ordem_matriz_confusao": {
            "labels": [0, 1],
            "linhas": "classe real",
            "colunas": "classe prevista",
        },
        "divisao": {
            "metodo": "train_test_split estratificado",
            "test_size": 0.20,
            "random_state": 42,
            "registros_treino": int(len(X_train)),
            "registros_teste": int(len(X_test)),
        },
        "diagnostico_dados": diagnostico,
        "observacoes_metodologicas": observacoes,
        "modelos": resultados,
    }
    with open(CAMINHO_METRICAS, "w", encoding="utf-8") as arquivo:
        json.dump(dados_metricas, arquivo, ensure_ascii=False, indent=2)
    with open(CAMINHO_RELATORIO, "w", encoding="utf-8") as arquivo:
        arquivo.write(formatar_relatorio(diagnostico, resultados, observacoes))
    if pipeline_rf_final is not None:
        joblib.dump(pipeline_rf_final, CAMINHO_MODELO_RF)
    print(f"\nMétricas salvas em: {CAMINHO_METRICAS}")
    print(f"Relatório salvo em: {CAMINHO_RELATORIO}")
    print(f"Modelo salvo em: {CAMINHO_MODELO_RF}")
if __name__ == "__main__":
    treinar_e_avaliar_modelos()
