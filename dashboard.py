import os
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
st.set_page_config(
    page_title="Infosiga SP | Análise de Sinistros",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
PALETA = {
    "fatal": "#C0392B",
    "nao_fatal": "#2874A6",
    "principal": "#24476B"
}
TIPOS_CONFIRMADOS = ["SINISTRO FATAL", "SINISTRO NAO FATAL"]
st.markdown(
    """

    <style>

        .block-container {

            padding-top: 2rem;

            padding-bottom: 2rem;

        }



        h1, h2, h3 {

            color: #263747;

        }



        div[data-testid="stMetric"] {

            background-color: #F5F7F9;

            border: 1px solid #E5E9ED;

            padding: 16px;

            border-radius: 10px;

        }



        div[data-testid="stMetricLabel"] {

            color: #536273;

        }



        div[data-testid="stMetricValue"] {

            color: #263747;

        }



        section[data-testid="stSidebar"] {

            background-color: #F3F5F8;

        }

    </style>

    """,
    unsafe_allow_html=True
)
def configurar_grafico(fig, altura=450):
    fig.update_layout(
        height=altura,
        margin=dict(l=35, r=25, t=105, b=75),
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#303846"
        ),
        title=dict(
            font=dict(size=17, color="#263747"),
            x=0.02,
            xanchor="left",
            y=0.98,
            yanchor="top"
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="left",
            x=0,
            font=dict(size=10)
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12
        )
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(
        gridcolor="#E8ECF0",
        zeroline=False
    )
    return fig
class DataLoader:
    @staticmethod
    @st.cache_data(show_spinner="Carregando os dados...")
    def carregar_dataset(caminho_parquet):
        if not os.path.exists(caminho_parquet):
            return None
        df = pd.read_parquet(caminho_parquet)
        if "data_sinistro" in df.columns:
            df["data_sinistro"] = pd.to_datetime(
                df["data_sinistro"],
                errors="coerce"
            )
        if "ano_sinistro" in df.columns:
            df["ano_sinistro"] = pd.to_numeric(
                df["ano_sinistro"],
                errors="coerce"
            )
        return df
    @staticmethod
    def aplicar_filtros(df, municipio, ano_range, mes, turno, tipo_via):
        if df is None or df.empty:
            return df
        resultado = df.copy()
        if municipio != "Todos" and "municipio" in resultado.columns:
            resultado = resultado[
                resultado["municipio"].astype(str) == str(municipio)
            ]
        if ano_range and "ano_sinistro" in resultado.columns:
            resultado = resultado[
                resultado["ano_sinistro"].between(
                    ano_range[0],
                    ano_range[1]
                )
            ]
        if mes != "Todos" and "data_sinistro" in resultado.columns:
            resultado = resultado[
                resultado["data_sinistro"].dt.month == int(mes)
            ]
        if turno != "Todos" and "turno" in resultado.columns:
            resultado = resultado[
                resultado["turno"].astype(str) == str(turno)
            ]
        if tipo_via != "Todos" and "tipo_via" in resultado.columns:
            resultado = resultado[
                resultado["tipo_via"].astype(str) == str(tipo_via)
            ]
        return resultado
class MetricsVisualizer:
    @staticmethod
    def obter_confirmados(df):
        if "tipo_registro" not in df.columns:
            return df.copy()
        return df[
            df["tipo_registro"].isin(TIPOS_CONFIRMADOS)
        ].copy()
    @staticmethod
    def renderizar_kpis(df):
        confirmados = MetricsVisualizer.obter_confirmados(df)
        total = len(confirmados)
        if "tipo_registro" in confirmados.columns:
            fatais = int(
                (confirmados["tipo_registro"] == "SINISTRO FATAL").sum()
            )
        else:
            fatais = 0
        nao_fatais = total - fatais
        percentual_fatais = fatais / total * 100 if total else 0
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Sinistros confirmados",
            f"{total:,}".replace(",", ".")
        )
        col2.metric(
            "Sinistros fatais",
            f"{fatais:,}".replace(",", ".")
        )
        col3.metric(
            "Sinistros não fatais",
            f"{nao_fatais:,}".replace(",", ".")
        )
        col4.metric(
            "Participação dos fatais",
            f"{percentual_fatais:.2f}%"
        )
        st.caption(
            "Os indicadores consideram somente registros classificados "
            "como fatais ou não fatais. Notificações não confirmadas "
            "não são incluídas nesses totais."
        )
    @staticmethod
    def gerar_grafico_temporal(df):
        if not {"ano_sinistro", "tipo_registro"}.issubset(df.columns):
            return None
        confirmados = MetricsVisualizer.obter_confirmados(df)
        agrupado = (
            confirmados
            .groupby(["ano_sinistro", "tipo_registro"])
            .size()
            .reset_index(name="quantidade")
        )
        if agrupado.empty:
            return None
        agrupado["categoria"] = agrupado["tipo_registro"].map({
            "SINISTRO FATAL": "Fatal",
            "SINISTRO NAO FATAL": "Não fatal"
        })
        fig = px.bar(
            agrupado,
            x="ano_sinistro",
            y="quantidade",
            color="categoria",
            barmode="group",
            title="Evolução temporal dos sinistros confirmados",
            labels={
                "ano_sinistro": "Ano",
                "quantidade": "Número de registros",
                "categoria": "Categoria"
            },
            color_discrete_map={
                "Fatal": PALETA["fatal"],
                "Não fatal": PALETA["nao_fatal"]
            },
            category_orders={
                "categoria": ["Fatal", "Não fatal"]
            }
        )
        fig.update_xaxes(dtick=1)
        fig.update_yaxes(rangemode="tozero")
        fig.update_traces(
            hovertemplate=(
                "Ano: %{x}<br>"
                "Registros: %{y:,}<extra>%{fullData.name}</extra>"
            )
        )
        return configurar_grafico(fig)
    @staticmethod
    def gerar_grafico_turno(df):
        if "turno" not in df.columns:
            return None
        confirmados = MetricsVisualizer.obter_confirmados(df)
        dados = (
            confirmados
            .dropna(subset=["turno"])
            .groupby("turno")
            .size()
            .reset_index(name="quantidade")
            .sort_values("quantidade", ascending=True)
        )
        if dados.empty:
            return None
        fig = px.bar(
            dados,
            x="quantidade",
            y="turno",
            orientation="h",
            title="Sinistros confirmados por turno",
            labels={
                "turno": "Turno do dia",
                "quantidade": "Número de registros"
            },
            text="quantidade",
            color_discrete_sequence=[PALETA["principal"]]
        )
        fig.update_traces(
            texttemplate="%{text:,}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "Turno: %{y}<br>"
                "Registros: %{x:,}<extra></extra>"
            )
        )
        fig.update_layout(
            showlegend=False,
            margin=dict(l=25, r=55, t=105, b=45)
        )
        return configurar_grafico(fig, altura=450)
    @staticmethod
    def renderizar_resumo(df):
        confirmados = MetricsVisualizer.obter_confirmados(df)
        st.subheader("Resumo da seleção")
        col1, col2 = st.columns(2)
        with col1:
            if "municipio" in confirmados.columns:
                quantidade = confirmados["municipio"].nunique()
                st.write(
                    f"Municípios com registros: **{quantidade:,}**"
                    .replace(",", ".")
                )
            if "ano_sinistro" in confirmados.columns:
                anos = confirmados["ano_sinistro"].dropna()
                if not anos.empty:
                    st.write(
                        f"Período observado: **{int(anos.min())}–"
                        f"{int(anos.max())}**"
                    )
        with col2:
            if "tipo_via" in confirmados.columns:
                moda = confirmados["tipo_via"].dropna().mode()
                if not moda.empty:
                    st.write(
                        f"Tipo de via mais frequente: **{moda.iloc[0]}**"
                    )
            if "turno" in confirmados.columns:
                moda = confirmados["turno"].dropna().mode()
                if not moda.empty:
                    st.write(
                        f"Turno mais frequente: **{moda.iloc[0]}**"
                    )
class MapVisualizer:
    @staticmethod
    def gerar_mapa_calor(df):
        if not {"latitude", "longitude"}.issubset(df.columns):
            return None, 0, 0
        if "tipo_registro" in df.columns:
            dados = df[
                df["tipo_registro"].isin(TIPOS_CONFIRMADOS)
            ].copy()
        else:
            dados = df.copy()
        for coluna in ["latitude", "longitude"]:
            dados[coluna] = pd.to_numeric(
                dados[coluna]
                .astype(str)
                .str.strip()
                .str.replace(",", ".", regex=False),
                errors="coerce"
            )
        dados = dados.dropna(
            subset=["latitude", "longitude"]
        )
        dados = dados[
            dados["latitude"].between(-26, -19)
            & dados["longitude"].between(-54, -44)
        ].copy()
        total_validos = len(dados)
        if dados.empty:
            return None, 0, 0
        quantidade_amostra = min(total_validos, 5000)
        if total_validos > quantidade_amostra:
            amostra = dados.sample(
                n=quantidade_amostra,
                random_state=42
            )
        else:
            amostra = dados
        centro_lat = float(amostra["latitude"].mean())
        centro_lon = float(amostra["longitude"].mean())
        mapa = folium.Map(
            location=[centro_lat, centro_lon],
            zoom_start=7,
            tiles="OpenStreetMap",
            control_scale=True
        )
        pontos = amostra[
            ["latitude", "longitude"]
        ].values.tolist()
        HeatMap(
            pontos,
            name="Concentração de sinistros",
            radius=14,
            blur=18,
            min_opacity=0.35,
            max_zoom=18,
            gradient={
                0.2: "blue",
                0.4: "cyan",
                0.6: "lime",
                0.8: "yellow",
                1.0: "red"
            }
        ).add_to(mapa)
        lat_min = float(amostra["latitude"].min())
        lat_max = float(amostra["latitude"].max())
        lon_min = float(amostra["longitude"].min())
        lon_max = float(amostra["longitude"].max())
        if lat_min == lat_max:
            lat_min -= 0.01
            lat_max += 0.01
        if lon_min == lon_max:
            lon_min -= 0.01
            lon_max += 0.01
        mapa.fit_bounds([
            [lat_min, lon_min],
            [lat_max, lon_max]
        ])
        return mapa, total_validos, quantidade_amostra
class MLClassifier:
    @staticmethod
    @st.cache_resource
    def carregar_modelo(caminho_pkl):
        if not os.path.exists(caminho_pkl):
            return None
        return joblib.load(caminho_pkl)
    @staticmethod
    def simular_classificacao(pipeline, entradas_usuario):
        entrada = pd.DataFrame([entradas_usuario])
        predicao = pipeline.predict(entrada)[0]
        probabilidades = pipeline.predict_proba(entrada)[0]
        classes = list(pipeline.classes_)
        prob_fatal = 0.0
        prob_nao_fatal = 0.0
        for classe, probabilidade in zip(classes, probabilidades):
            if int(classe) == 1:
                prob_fatal = float(probabilidade) * 100
            elif int(classe) == 0:
                prob_nao_fatal = float(probabilidade) * 100
        return int(predicao), prob_fatal, prob_nao_fatal
class ModelMetricsVisualizer:
    """Carrega e apresenta a avaliação estática feita no conjunto de teste."""
    @staticmethod
    @st.cache_data(show_spinner=False)
    def carregar_metricas(caminho_json):
        if not os.path.exists(caminho_json):
            return None
        try:
            with open(caminho_json, "r", encoding="utf-8") as arquivo:
                dados = json.load(arquivo)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(dados.get("modelos"), list) or not dados["modelos"]:
            return None
        return dados
    @staticmethod
    def renderizar(caminho_json):
        st.markdown("### Avaliação dos modelos no conjunto de teste")
        st.caption(
            "Estas métricas são da avaliação registrada no treinamento, "
            "não são recalculadas pelos filtros do dashboard."
        )
        dados = ModelMetricsVisualizer.carregar_metricas(caminho_json)
        if dados is None:
            st.info(
                f"Arquivo de métricas não encontrado ou inválido: `{caminho_json}`. "
                "Execute `train_model.py` para gerá-lo."
            )
            return
        modelos = dados["modelos"]
        nomes = [item.get("nome", f"Modelo {i + 1}") for i, item in enumerate(modelos)]
        nome_selecionado = st.selectbox(
            "Modelo para inspecionar",
            nomes,
            key="ml_modelo_metricas"
        )
        resultado = next(
            item for item in modelos
            if item.get("nome", "") == nome_selecionado
        )
        fatal = resultado.get("metricas_por_classe", {}).get("fatal", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Acurácia", f"{resultado.get('acuracia', 0) * 100:.2f}%")
        col2.metric("Precisão — fatal", f"{fatal.get('precisao', 0) * 100:.2f}%")
        col3.metric("Recall — fatal", f"{fatal.get('recall', 0) * 100:.2f}%")
        col4.metric("F1-score — fatal", f"{fatal.get('f1_score', 0):.3f}")
        matriz = resultado.get("matriz_confusao")
        if (
            isinstance(matriz, list) and len(matriz) == 2
            and all(isinstance(linha, list) and len(linha) == 2 for linha in matriz)
        ):
            fig = px.imshow(
                matriz,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                labels={
                    "x": "Classe prevista",
                    "y": "Classe real",
                    "color": "Registros",
                },
                x=["Não fatal (0)", "Fatal (1)"],
                y=["Não fatal (0)", "Fatal (1)"],
                title="Matriz de confusão (linhas = real; colunas = prevista)",
            )
            st.plotly_chart(configurar_grafico(fig, altura=390), use_container_width=True)
        st.markdown("#### Comparação dos três modelos")
        linhas = []
        for item in modelos:
            fatal_item = item.get("metricas_por_classe", {}).get("fatal", {})
            linhas.append({
                "Modelo": item.get("nome", "Não informado"),
                "Acurácia (%)": round(item.get("acuracia", 0) * 100, 2),
                "Precisão fatal (%)": round(fatal_item.get("precisao", 0) * 100, 2),
                "Recall fatal (%)": round(fatal_item.get("recall", 0) * 100, 2),
                "F1 fatal": round(fatal_item.get("f1_score", 0), 3),
            })
        st.dataframe(pd.DataFrame(linhas), hide_index=True, use_container_width=True)
        with st.expander("Como interpretar e limitações da avaliação"):
            st.markdown(
                """
                - **Precisão fatal:** proporção das classificações fatais que realmente eram fatais.
                - **Recall fatal:** proporção dos registros fatais reais identificados pelo modelo.
                - **F1-score fatal:** equilíbrio entre precisão e recall.
                - Como a classe fatal é minoritária, a acurácia isolada não é suficiente.
                - Muitos falsos positivos podem ocorrer; os resultados são experimentais e não devem
                  ser usados isoladamente para decisões reais de segurança viária.
                - A avaliação reflete a divisão treino/teste registrada no treinamento e depende
                  da qualidade das variáveis preditoras e da separação dos registros.
                """
            )
        observacoes = dados.get("observacoes_metodologicas", [])
        if observacoes:
            with st.expander("Observações metodológicas registradas no treinamento"):
                for observacao in observacoes:
                    st.write(f"- {observacao}")
def obter_opcoes(df, coluna):
    if coluna not in df.columns:
        return []
    return sorted(
        df[coluna].dropna().astype(str).unique().tolist()
    )
def main():
    st.title("Análise de Sinistros de Trânsito — Infosiga SP")
    st.markdown(
        "Análise espaço-temporal e classificação experimental "
        "de sinistros quanto à fatalidade, com dados de 2021 a 2025."
    )
    caminho_parquet = (
        "data/processed/infosiga_dashboard_2021_2025.parquet"
    )
    caminho_modelo = "models/random_forest_model.pkl"
    df_bruto = DataLoader.carregar_dataset(caminho_parquet)
    if df_bruto is None:
        st.error(
            f"Base não encontrada em `{caminho_parquet}`. "
            "Execute o `etl.py` primeiro."
        )
        st.stop()
    if df_bruto.empty:
        st.error("A base de dados está vazia.")
        st.stop()
    st.sidebar.title("Filtros de consulta")
    st.sidebar.caption(
        "Os filtros atualizam os indicadores, os gráficos e o mapa."
    )
    if "municipio" in df_bruto.columns:
        municipios = ["Todos"] + obter_opcoes(
            df_bruto,
            "municipio"
        )
        municipio_sel = st.sidebar.selectbox(
            "Município",
            municipios,
            key="sb_municipio"
        )
    else:
        municipio_sel = "Todos"
    if "ano_sinistro" in df_bruto.columns:
        anos = pd.to_numeric(
            df_bruto["ano_sinistro"],
            errors="coerce"
        ).dropna()
        if anos.empty:
            st.error("Não foram encontrados anos válidos na base.")
            st.stop()
        ano_min = int(anos.min())
        ano_max = int(anos.max())
        if ano_min == ano_max:
            anos_sel = (ano_min, ano_max)
            st.sidebar.write(f"Ano disponível: {ano_min}")
        else:
            anos_sel = st.sidebar.slider(
                "Intervalo de anos",
                min_value=ano_min,
                max_value=ano_max,
                value=(ano_min, ano_max),
                key="sl_anos"
            )
    else:
        anos_sel = None
    if "data_sinistro" in df_bruto.columns:
        nomes_meses = {
            1: "01 - Janeiro", 2: "02 - Fevereiro", 3: "03 - Março",
            4: "04 - Abril", 5: "05 - Maio", 6: "06 - Junho",
            7: "07 - Julho", 8: "08 - Agosto", 9: "09 - Setembro",
            10: "10 - Outubro", 11: "11 - Novembro", 12: "12 - Dezembro",
        }
        meses_disponiveis = sorted(
            int(valor) for valor in df_bruto["data_sinistro"].dt.month.dropna().unique()
        )
        opcoes_meses = ["Todos"] + [nomes_meses[m] for m in meses_disponiveis]
        mes_label_sel = st.sidebar.selectbox(
            "Mês",
            opcoes_meses,
            key="sb_mes"
        )
        mes_sel = int(mes_label_sel[:2]) if mes_label_sel != "Todos" else "Todos"
    else:
        mes_sel = "Todos"
    if "turno" in df_bruto.columns:
        turnos = ["Todos"] + obter_opcoes(
            df_bruto,
            "turno"
        )
        turno_sel = st.sidebar.selectbox(
            "Turno do dia",
            turnos,
            key="sb_turno"
        )
    else:
        turno_sel = "Todos"
    if "tipo_via" in df_bruto.columns:
        vias = ["Todos"] + obter_opcoes(
            df_bruto,
            "tipo_via"
        )
        via_sel = st.sidebar.selectbox(
            "Tipo de via",
            vias,
            key="sb_via"
        )
    else:
        via_sel = "Todos"
    if st.sidebar.button(
        "Restaurar filtros",
        use_container_width=True
    ):
        for chave in [
            "sb_municipio",
            "sl_anos",
            "sb_turno",
            "sb_via"
        ]:
            st.session_state.pop(chave, None)
        st.rerun()
    df_filtrado = DataLoader.aplicar_filtros(
        df_bruto,
        municipio_sel,
        anos_sel,
        mes_sel,
        turno_sel,
        via_sel
    )
    st.sidebar.divider()
    st.sidebar.caption(
        f"Registros carregados: {len(df_bruto):,}".replace(",", ".")
    )
    st.sidebar.caption(
        f"Registros após filtros: {len(df_filtrado):,}".replace(",", ".")
    )
    if df_filtrado.empty:
        st.warning(
            "Nenhum registro encontrado para os filtros selecionados."
        )
        st.stop()
    tab1, tab2, tab3 = st.tabs([
        "Indicadores e gráficos",
        "Mapa de concentração",
        "Classificação por Machine Learning"
    ])
    with tab1:
        st.subheader("Indicadores consolidados")
        MetricsVisualizer.renderizar_kpis(df_filtrado)
        st.divider()
        col_grafico1, col_grafico2 = st.columns(
            [1, 1],
            gap="large"
        )
        with col_grafico1:
            fig_temporal = (
                MetricsVisualizer.gerar_grafico_temporal(
                    df_filtrado
                )
            )
            if fig_temporal is not None:
                st.plotly_chart(
                    fig_temporal,
                    use_container_width=True
                )
            else:
                st.info(
                    "Não há dados suficientes para gerar o gráfico temporal."
                )
        with col_grafico2:
            fig_turno = MetricsVisualizer.gerar_grafico_turno(
                df_filtrado
            )
            if fig_turno is not None:
                st.plotly_chart(
                    fig_turno,
                    use_container_width=True
                )
            else:
                st.info(
                    "Não há dados de turno suficientes para gerar o gráfico."
                )
        st.divider()
        MetricsVisualizer.renderizar_resumo(df_filtrado)
        with st.expander("Como interpretar os indicadores"):
            st.markdown(
                """

                - **Sinistros confirmados:** registros classificados como

                  fatais ou não fatais.

                - **Participação dos fatais:** proporção dos registros fatais

                  em relação ao total confirmado na seleção.

                - **Evolução temporal:** compara a quantidade de registros

                  fatais e não fatais por ano.

                - **Distribuição por turno:** apresenta a quantidade de

                  sinistros registrados em cada turno.



                Os valores representam registros da base. Isoladamente,

                eles não constituem taxas de risco por habitante, veículo

                ou volume de tráfego.

                """
            )
    with tab2:
        st.subheader("Mapa de concentração de sinistros")
        st.write(
            "O mapa de calor apresenta a concentração espacial dos "
            "registros confirmados que possuem coordenadas válidas."
        )
        mapa, total_validos, total_exibidos = (
            MapVisualizer.gerar_mapa_calor(df_filtrado)
        )
        if mapa is None:
            st.warning(
                "Não foi possível gerar o mapa. Verifique se a base contém "
                "coordenadas válidas para os filtros selecionados."
            )
        else:
            col_mapa1, col_mapa2 = st.columns(2)
            col_mapa1.metric(
                "Registros com coordenadas válidas",
                f"{total_validos:,}".replace(",", ".")
            )
            col_mapa2.metric(
                "Pontos representados",
                f"{total_exibidos:,}".replace(",", ".")
            )
            if total_validos > total_exibidos:
                st.caption(
                    "Para melhorar o desempenho, o mapa utiliza uma amostra "
                    "aleatória fixa de até 5.000 registros. A intensidade "
                    "visual é aproximada e não representa todos os pontos."
                )
            else:
                st.caption(
                    "Todos os registros confirmados com coordenadas válidas "
                    "dentro dos limites geográficos definidos foram usados."
                )
            st_folium(
                mapa,
                width=None,
                height=650,
                key="mapa_concentracao_infosiga",
                returned_objects=[]
            )
        with st.expander("Limitações da visualização cartográfica"):
            st.markdown(
                """

                - A visualização depende da disponibilidade e da qualidade

                  das coordenadas da base.

                - Registros sem coordenadas válidas ou fora dos limites

                  geográficos definidos não são representados.

                - Quando há mais de 5.000 pontos, utiliza-se uma amostra

                  para melhorar o desempenho.

                - As manchas indicam concentração espacial dos registros

                  representados; não demonstram causalidade nem risco

                  individual de ocorrência.

                """
            )
    with tab3:
        ModelMetricsVisualizer.renderizar(
            "data/processed/metricas_modelos.json"
        )
        st.divider()
        st.subheader(
            "Classificação experimental quanto à fatalidade"
        )
        st.markdown(
            """

            O módulo utiliza um modelo Random Forest treinado previamente

            para classificar um cenário com base em características

            contextuais presentes nos dados históricos.



            O resultado é experimental. Não representa uma previsão

            determinística de acidentes futuros nem demonstra relações

            de causa e efeito.

            """
        )
        pipeline_ml = MLClassifier.carregar_modelo(
            caminho_modelo
        )
        if pipeline_ml is None:
            st.warning(
                f"Modelo não encontrado em `{caminho_modelo}`. "
                "Execute `train_model.py` para gerar o arquivo."
            )
            st.stop()
        colunas_modelo = [
            "tp_sinistro_primario",
            "tipo_via",
            "tipo_local",
            "administracao",
            "circunscricao",
            "turno"
        ]
        faltantes = [
            coluna for coluna in colunas_modelo
            if coluna not in df_bruto.columns
        ]
        if faltantes:
            st.error(
                "Colunas necessárias ausentes na base: "
                + ", ".join(faltantes)
            )
            st.stop()
        st.caption(
            "As opções são obtidas dos valores disponíveis na base carregada."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            input_tp_sinistro = st.selectbox(
                "Tipo de sinistro primário",
                obter_opcoes(
                    df_bruto,
                    "tp_sinistro_primario"
                )
            )
            input_tipo_via = st.selectbox(
                "Tipo de via",
                obter_opcoes(
                    df_bruto,
                    "tipo_via"
                )
            )
        with col2:
            input_tipo_local = st.selectbox(
                "Tipo de local",
                obter_opcoes(
                    df_bruto,
                    "tipo_local"
                )
            )
            input_administracao = st.selectbox(
                "Administração",
                obter_opcoes(
                    df_bruto,
                    "administracao"
                )
            )
        with col3:
            input_circunscricao = st.selectbox(
                "Circunscrição",
                obter_opcoes(
                    df_bruto,
                    "circunscricao"
                )
            )
            input_turno = st.selectbox(
                "Turno do dia",
                obter_opcoes(
                    df_bruto,
                    "turno"
                )
            )
        if st.button(
            "Executar classificação",
            type="primary",
            use_container_width=True
        ):
            entradas = {
                "tp_sinistro_primario": input_tp_sinistro,
                "tipo_via": input_tipo_via,
                "tipo_local": input_tipo_local,
                "administracao": input_administracao,
                "circunscricao": input_circunscricao,
                "turno": input_turno
            }
            try:
                (
                    predicao,
                    prob_fatal,
                    prob_nao_fatal
                ) = MLClassifier.simular_classificacao(
                    pipeline_ml,
                    entradas
                )
            except Exception as erro:
                st.error(
                    "Não foi possível executar a classificação. "
                    "Verifique se o modelo foi treinado com as mesmas "
                    "variáveis do formulário."
                )
                st.exception(erro)
                st.stop()
            st.divider()
            st.markdown("#### Resultado da classificação")
            resultado_col, prob_col = st.columns(2)
            with resultado_col:
                if predicao == 1:
                    st.error(
                        "**Classe atribuída: sinistro fatal**"
                    )
                else:
                    st.info(
                        "**Classe atribuída: sinistro não fatal**"
                    )
                st.caption(
                    "Esta é a classe atribuída pelo modelo para as "
                    "características selecionadas."
                )
            with prob_col:
                st.write(
                    f"Probabilidade estimada da classe fatal: "
                    f"**{prob_fatal:.2f}%**"
                )
                st.progress(
                    max(
                        0,
                        min(100, int(round(prob_fatal)))
                    )
                )
                st.write(
                    f"Probabilidade estimada da classe não fatal: "
                    f"**{prob_nao_fatal:.2f}%**"
                )
            st.warning(
                "As probabilidades refletem a saída do modelo e não "
                "necessariamente uma probabilidade individual calibrada "
                "de ocorrência futura. O resultado não deve ser usado "
                "isoladamente para decisões de segurança viária."
            )
        with st.expander("Limitações do modelo"):
            st.markdown(
                """

                - O desempenho depende da qualidade e da representatividade

                  dos dados utilizados no treinamento.

                - O desequilíbrio entre as classes pode afetar a interpretação

                  da acurácia e de outras métricas.

                - A avaliação deve considerar precisão, recall e F1-score

                  da classe fatal, além da acurácia geral.

                - O modelo identifica padrões nos dados, mas não comprova

                  relações de causa e efeito.

                - A classificação é experimental e não constitui previsão

                  determinística de acidentes futuros.

                """
            )
if __name__ == "__main__":
    main()
