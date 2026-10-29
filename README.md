# Infosiga SP — Análise Espaço-Temporal e Classificação por Aprendizado de Máquina

Protótipo funcional desenvolvido como parte do Projeto Integrador do curso de Análise e Desenvolvimento de Sistemas (ADS) do Instituto Federal de Educação, Ciência e Tecnologia de São Paulo (IFSP).

O projeto tem como objetivo explorar dados públicos de sinistros de trânsito do Estado de São Paulo, por meio de análises estatísticas, visualizações espaciais e modelos de classificação por aprendizado de máquina.

## Funcionalidades

- Processamento e preparação dos microdados públicos do Infosiga SP.
- Análise exploratória dos registros de sinistros de trânsito.
- Consulta interativa por município, período, turno e tipo de via.
- Visualização de indicadores, gráficos e mapa de calor.
- Comparação experimental de modelos de classificação.
- Classificação experimental de cenários com apresentação da classe atribuída e das probabilidades estimadas.

## Tecnologias utilizadas

- **Linguagem:** Python
- **Manipulação e processamento de dados:** Pandas, NumPy e PyArrow
- **Armazenamento dos dados:** Apache Parquet
- **Aplicação web:** Streamlit
- **Visualização de dados:** Plotly
- **Visualização geográfica:** Folium e Streamlit-Folium
- **Aprendizado de máquina:** Scikit-learn
- **Persistência dos modelos:** Joblib

## Estrutura de arquivos

```text
prototipo_tcc/
├── dashboard.py
├── etl.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── processed/
│       ├── infosiga_dashboard_2021_2025.parquet
│       ├── infosiga_ml_2021_2025.parquet
│       └── metricas_modelos.json
└── models/
    └── random_forest_model.pkl
```

Os nomes apresentados representam a estrutura esperada do projeto. A lista deve ser ajustada caso os arquivos reais tenham nomes diferentes ou existam outros módulos necessários para a execução.

### Descrição dos principais arquivos

- **`dashboard.py`:** interface web interativa desenvolvida com Streamlit, contendo indicadores, gráficos, mapa de calor, filtros e funcionalidades de classificação.
- **`etl.py`:** pipeline de extração, transformação e carregamento (ETL) dos microdados públicos do Infosiga SP.
- **`train_model.py`:** script responsável pelo treinamento e pela avaliação comparativa dos modelos de Regressão Logística, Árvore de Decisão e Random Forest.
- **`requirements.txt`:** lista de bibliotecas Python necessárias para executar o projeto.
- **`data/processed/`:** diretório destinado aos conjuntos de dados processados e às métricas geradas durante o treinamento.
- **`models/`:** diretório destinado aos arquivos dos modelos treinados.

## Como executar a aplicação localmente

### 1. Pré-requisitos

- Python 3.x instalado.
- Arquivos do projeto disponíveis localmente.
- Conjuntos de dados processados e demais arquivos necessários à execução.

### 2. Instalar as dependências

Abra o terminal na pasta raiz do projeto e execute:

```bash
pip install -r requirements.txt
```

### 3. Executar o dashboard

No mesmo terminal, execute:

```bash
python -m streamlit run dashboard.py
```

O Streamlit disponibilizará o endereço local da aplicação no terminal. Abra esse endereço no navegador para utilizar o dashboard.

### 4. Processamento dos dados e treinamento dos modelos

Caso seja necessário gerar novamente os conjuntos de dados processados, execute:

```bash
python etl.py
```

Para treinar novamente os modelos de classificação, execute:

```bash
python train_model.py
```

Essas etapas dependem dos arquivos de entrada, das configurações e das dependências previstas nos respectivos scripts. Se os dados processados e os modelos necessários já estiverem disponíveis, poderá ser possível executar diretamente o dashboard.

## Observações

- O projeto utiliza dados públicos do Infosiga SP para fins acadêmicos e de análise exploratória.
- Os modelos de aprendizado de máquina têm finalidade experimental. Suas classificações e probabilidades estimadas não devem ser interpretadas como previsões determinísticas de sinistros futuros.
- Os resultados dependem da qualidade, da disponibilidade e das características dos dados utilizados.
