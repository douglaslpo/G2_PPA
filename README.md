# G2_PPA — Projeto Final de Séries Temporais

Previsão mensal de passageiros aéreos com análise exploratória, decomposição, testes de estacionariedade e comparação entre modelos clássicos e modernos, utilizando a série clássica **Air Passengers**.

**Autor:** Douglas Leonard Passos Olinto  
**Disciplina:** Séries Temporais  
**Tema:** Previsão de passageiros aéreos — Air Passengers (1949–1960)

---

## Sumário

- [Visão geral](#visão-geral)
- [Por que a série Air Passengers?](#por-que-a-série-air-passengers)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Metodologia](#metodologia)
- [Modelos comparados](#modelos-comparados)
- [Métricas de avaliação](#métricas-de-avaliação)
- [Principais conclusões](#principais-conclusões)
- [Limitações](#limitações)
- [Como executar o projeto](#como-executar-o-projeto)
- [Dependências](#dependências)
- [Referências](#referências)

---

## Visão geral

Este repositório reúne o trabalho completo do projeto final da disciplina de Séries Temporais. O objetivo é aplicar um **pipeline estruturado de análise e previsão temporal**, desde o carregamento e a exploração dos dados até a comparação de diferentes abordagens de forecasting.

O fluxo segue o roteiro acadêmico proposto:

1. Escolha de uma série temporal real;
2. Carregamento, limpeza e definição do índice temporal;
3. Análise exploratória (EDA);
4. Decomposição da série;
5. Verificação de estacionariedade;
6. Modelagem com baseline, modelo clássico e modelo moderno;
7. Avaliação por MAE, RMSE e MAPE;
8. Conclusão com justificativa do melhor modelo e limitações.

A implementação está disponível em **dois formatos equivalentes**:

- **Notebook Jupyter** — ideal para apresentação, revisão e execução interativa;
- **Script Python** — versão linear do mesmo pipeline, útil para execução direta ou adaptação.

---

## Por que a série Air Passengers?

A série **Air Passengers** registra o número mensal de passageiros internacionais de companhias aéreas entre **janeiro de 1949 e dezembro de 1960**, totalizando **144 observações** (12 anos × 12 meses).

Ela é amplamente utilizada em estudos introdutórios de séries temporais por combinar características didáticas e desafiadoras:

| Característica | Descrição |
|---|---|
| **Tendência** | Crescimento consistente no número de passageiros ao longo dos anos |
| **Sazonalidade** | Padrão anual recorrente, com meses de maior e menor movimento |
| **Amplitude sazonal crescente** | A variação sazonal aumenta conforme o nível da série cresce |
| **Aplicação prática** | Problema real de previsão de demanda em transporte aéreo |

Essas propriedades tornam a série adequada para demonstrar decomposição multiplicativa, transformações para estacionariedade e comparação entre métodos estatísticos e de aprendizado de máquina.

---

## Estrutura do repositório

```
G2_PPA/
├── README.md                                                          # Este arquivo
├── projeto_final_series_temporais_air_passengers_douglas.ipynb        # Notebook principal
├── projeto_final_series_temporais_air_passengers_douglas.py           # Script equivalente ao notebook
├── roteiro_pitch_projeto_final_series_temporais_douglas.txt           # Roteiro para apresentação oral
├── G2_PPA.pdf                                                         # Documento complementar da disciplina
└── .gitignore                                                         # Exclusões para Python/Jupyter
```

### Descrição dos arquivos

| Arquivo | Conteúdo |
|---|---|
| `.ipynb` | Pipeline completo com gráficos, comentários interpretativos e conclusões |
| `.py` | Mesma lógica do notebook, organizada como script executável |
| `roteiro_pitch_*.txt` | Roteiro resumido para apresentação do projeto (pitch) |
| `G2_PPA.pdf` | Material de referência ou enunciado da atividade |

---

## Metodologia

### 1. Preparação dos dados

- Valores da série embutidos diretamente no código (sem dependência de arquivo CSV externo);
- Criação de índice temporal mensal com frequência `MS` (Month Start);
- Conversão para `datetime` e definição como índice do DataFrame;
- Verificação de valores ausentes e inferência da frequência temporal.

### 2. Análise exploratória (EDA)

Foram produzidos os seguintes gráficos e análises:

- **Série temporal completa** — visualização da tendência e dos ciclos anuais;
- **Médias móveis** — janelas de 3 e 12 meses para suavizar oscilações e evidenciar tendência;
- **Boxplot por mês** — comparação da distribuição de passageiros entre os 12 meses do ano;
- **Histograma** — distribuição geral dos valores observados;
- **Heatmap ano × mês** — visualização da sazonalidade ao longo dos anos.

**Principais observações da EDA:**

- Há crescimento claro ao longo do tempo (tendência);
- Meses centrais do ano tendem a concentrar valores mais altos (sazonalidade anual);
- A amplitude das oscilações sazonais aumenta nos anos mais recentes da série.

### 3. Engenharia de features e detecção de outliers

Foram criadas defasagens temporais (`lag_1`, `lag_12`) e médias móveis deslocadas, sempre utilizando **apenas informação do passado** (via `.shift()`), para evitar vazamento de dados.

A detecção de outliers foi feita com a regra do **IQR (Interquartile Range)** aplicada ao resíduo:

```
resíduo = série original − média móvel de 12 meses
```

Pontos fora dos limites `Q1 − 1,5 × IQR` e `Q3 + 1,5 × IQR` foram sinalizados e destacados em gráfico.

### 4. Decomposição da série

Utilizou-se a decomposição clássica do `statsmodels` com:

- **Modelo:** multiplicativo (`model="multiplicative"`);
- **Período sazonal:** 12 meses.

A escolha do modelo multiplicativo é coerente porque a **amplitude da sazonalidade cresce junto com o nível da série** — comportamento típico de séries econômicas em expansão.

Componentes analisados:

| Componente | Interpretação |
|---|---|
| **Observed** | Série original |
| **Trend** | Tendência de longo prazo |
| **Seasonal** | Padrão que se repete a cada 12 meses |
| **Residual** | Variação não explicada por tendência ou sazonalidade |

### 5. Estacionariedade

Foi aplicado o **Teste ADF (Augmented Dickey-Fuller)** em duas etapas:

1. **Série original** → provavelmente **não estacionária** (tendência + sazonalidade);
2. **Log + diferenciação sazonal (12 meses)** → série mais adequada para análises que exigem estacionariedade.

A transformação logarítmica ajuda a **estabilizar a variância**, enquanto a diferenciação sazonal remove parte do padrão anual repetitivo.

### 6. Separação treino / teste

Para respeitar a ordem temporal e evitar vazamento de dados:

| Conjunto | Período | Observações |
|---|---|---|
| **Treino** | Jan/1949 – Dez/1958 | 120 meses |
| **Teste** | Jan/1959 – Dez/1960 | 24 meses (2 anos) |

O conjunto de teste corresponde sempre ao **período mais recente** da série.

---

## Modelos comparados

Três abordagens foram implementadas e comparadas no mesmo horizonte de previsão (24 meses):

### Modelo 1 — Baseline sazonal (referência)

Previsão simples: para cada mês do teste, utiliza o valor observado no **mesmo mês do ano anterior** (`shift(12)`).

> Função: servir como referência mínima. Qualquer modelo mais sofisticado precisa superar essa baseline para justificar sua complexidade.

### Modelo 2 — Holt-Winters multiplicativo (clássico)

Suavização exponencial com:

- Tendência multiplicativa (`trend="mul"`);
- Sazonalidade multiplicativa (`seasonal="mul"`);
- Período sazonal de 12 meses;
- Parâmetros otimizados automaticamente (`optimized=True`).

> Adequado para séries com crescimento e sazonalidade de amplitude crescente.

### Modelo 3 — XGBoost com features temporais (moderno)

Regressor baseado em gradient boosting com features derivadas exclusivamente do passado:

| Feature | Descrição |
|---|---|
| `lag_1`, `lag_2`, `lag_3` | Valores defasados em 1, 2 e 3 meses |
| `lag_12` | Valor do mesmo mês no ano anterior |
| `rolling_3`, `rolling_12` | Médias móveis deslocadas |
| `mes`, `ano` | Componentes calendário |

**Hiperparâmetros principais:** `n_estimators=300`, `learning_rate=0.03`, `max_depth=3`, `random_state=42`.

> Abordagem de aprendizado de máquina aplicada a forecasting univariado com engenharia manual de features temporais.

---

## Métricas de avaliação

Todos os modelos foram avaliados no conjunto de teste com três métricas:

| Métrica | Fórmula (conceitual) | Interpretação |
|---|---|---|
| **MAE** | Média do erro absoluto | Erro médio em unidades de passageiros |
| **RMSE** | Raiz da média do erro quadrático | Penaliza erros grandes com mais peso |
| **MAPE (%)** | Erro percentual absoluto médio | Facilita a interpretação relativa do erro |

A comparação final prioriza o **MAPE** por expressar o erro de forma percentual, mais intuitiva para comunicação dos resultados.

> **Nota:** Os valores exatos das métricas são gerados na execução do notebook/script e podem variar ligeiramente conforme versões das bibliotecas.

---

## Principais conclusões

Com base na análise completa do pipeline, as respostas às perguntas centrais do projeto são:

### a) Existe tendência?

**Sim.** A série apresenta crescimento claro e consistente no número de passageiros ao longo dos 12 anos analisados.

### b) Existe sazonalidade?

**Sim.** Há sazonalidade anual bem definida, com padrões mensais que se repetem a cada 12 meses. O heatmap ano × mês e o boxplot por mês confirmam esse comportamento.

### c) Há comportamento atípico?

**Sim.** A regra do IQR aplicada ao resíduo (série − média móvel de 12 meses) sinalizou observações que se afastam do comportamento suavizado. Vale ressaltar que, em séries com forte tendência, alguns pontos nos anos finais podem ser classificados como outliers sem representar anomalias reais.

### Escolha do melhor modelo

O melhor modelo é aquele com **menor MAE, RMSE e MAPE** no conjunto de teste. A comparação inclui sempre o baseline sazonal como referência obrigatória — garantindo que modelos mais complexos realmente agreguem valor preditivo.

---

## Limitações

Este projeto foi desenvolvido com finalidade **acadêmica e didática**. As principais limitações são:

- **Base pequena:** apenas 144 observações mensais;
- **Série univariada:** não inclui variáveis externas (preço de passagens, renda, feriados, eventos econômicos);
- **Período histórico:** dados de 1949–1960, sem representatividade para o contexto atual do setor aéreo;
- **XGBoost com poucos dados:** modelos de ML podem ter desempenho limitado em bases curtas;
- **Previsão multi-step:** o XGBoost utiliza features do passado disponíveis no teste; previsões recursivas de longo prazo exigiriam abordagem adicional.

---

## Como executar o projeto

### Pré-requisitos

- Python 3.9 ou superior (recomendado)
- Ambiente virtual (opcional, mas recomendado)

### Instalação das dependências

```bash
pip install numpy pandas matplotlib statsmodels scikit-learn xgboost jupyter
```

### Opção 1 — Notebook Jupyter (recomendado)

```bash
jupyter notebook projeto_final_series_temporais_air_passengers_douglas.ipynb
```

Execute as células sequencialmente. Gráficos e tabelas de resultados serão exibidos inline.

### Opção 2 — Script Python

```bash
python projeto_final_series_temporais_air_passengers_douglas.py
```

> O script exibirá gráficos em janelas separadas (requer backend gráfico compatível) e imprimirá resultados no terminal.

### Apresentação (pitch)

Consulte o arquivo `roteiro_pitch_projeto_final_series_temporais_douglas.txt` para um roteiro resumido de apresentação oral do projeto.

---

## Dependências

| Biblioteca | Uso no projeto |
|---|---|
| `numpy` | Operações numéricas e arrays |
| `pandas` | Manipulação de séries temporais e DataFrames |
| `matplotlib` | Visualizações e gráficos exploratórios |
| `statsmodels` | Decomposição sazonal, teste ADF e Holt-Winters |
| `scikit-learn` | Métricas MAE e RMSE |
| `xgboost` | Modelo de gradient boosting com features temporais |
| `jupyter` | Execução do notebook (opcional) |

---

## Referências

- Box, G. E. P.; Jenkins, G. M. — *Time Series Analysis: Forecasting and Control*
- Hyndman, R. J.; Athanasopoulos, G. — [*Forecasting: Principles and Practice*](https://otexts.com/fpp3/)
- Série Air Passengers — dataset clássico amplamente utilizado em forecasting e ensino de séries temporais

---

## Licença

Projeto acadêmico desenvolvido para fins educacionais. Consulte a instituição de ensino para orientações sobre uso e reprodução.
