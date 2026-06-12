# Projeto Final — Séries Temporais
# Aluno: Douglas Leonard Passos Olinto
# Tema: Previsão de passageiros aéreos — Air Passengers


# # Projeto Final — Séries Temporais
# 
# ## Previsão de Passageiros Aéreos com Análise Exploratória, Decomposição e Modelagem
# 
# **Aluno:** Douglas Leonard Passos Olinto  
# 
# Este notebook segue o pipeline solicitado no projeto final de Séries Temporais:
# 
# 1. Escolha de uma série temporal real;
# 2. Carregamento, limpeza e definição do índice temporal;
# 3. Análise exploratória;
# 4. Decomposição da série;
# 5. Verificação de estacionariedade;
# 6. Modelagem com baseline, modelo clássico e modelo moderno;
# 7. Avaliação por MAE, RMSE e MAPE;
# 8. Conclusão com justificativa do melhor modelo e limitações.
# 
# ## Dataset escolhido
# 
# Foi escolhida a série **Air Passengers**, uma série clássica mensal com o número de passageiros internacionais de companhias aéreas entre 1949 e 1960.
# 
# A série é adequada para este projeto porque apresenta:
# 
# - tendência clara de crescimento;
# - sazonalidade anual identificável;
# - amplitude sazonal crescente ao longo do tempo;
# - boa aplicação para forecasting.

# ## 1. Importação das bibliotecas

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["axes.grid"] = True

np.random.seed(42)

# ## 2. Carregamento do dataset

# Série Air Passengers: passageiros mensais internacionais, 1949-1960.
# Valores clássicos usados em estudos introdutórios de séries temporais.
passageiros = [
    112,118,132,129,121,135,148,148,136,119,104,118,
    115,126,141,135,125,149,170,170,158,133,114,140,
    145,150,178,163,172,178,199,199,184,162,146,166,
    171,180,193,181,183,218,230,242,209,191,172,194,
    196,196,236,235,229,243,264,272,237,211,180,201,
    204,188,235,227,234,264,302,293,259,229,203,229,
    242,233,267,269,270,315,364,347,312,274,237,278,
    284,277,317,313,318,374,413,405,355,306,271,306,
    315,301,356,348,355,422,465,467,404,347,305,336,
    340,318,362,348,363,435,491,505,404,359,310,337,
    360,342,406,396,420,472,548,559,463,407,362,405,
    417,391,419,461,472,535,622,606,508,461,390,432
]

datas = pd.date_range(start="1949-01-01", periods=len(passageiros), freq="MS")

df = pd.DataFrame({
    "data": datas,
    "passageiros": passageiros
})

df.head()

# ## 3. Preparação dos dados
# 
# A coluna `data` é convertida para o tipo datetime e definida como índice.  
# Isso é necessário para que o Pandas reconheça a estrutura temporal da série e permita o uso de operações como `resample()`, `shift()` e `rolling()`.

df["data"] = pd.to_datetime(df["data"])
df = df.set_index("data")

print(df.info())
df.head()

# Verificação de valores ausentes
print("Valores ausentes por coluna:")
print(df.isna().sum())

# Frequência da série
print("\nFrequência inferida:", pd.infer_freq(df.index))

# ## 4. Análise exploratória da série temporal

plt.figure(figsize=(14, 5))
plt.plot(df.index, df["passageiros"], marker="o")
plt.title("Série temporal — Air Passengers")
plt.xlabel("Data")
plt.ylabel("Número de passageiros")
plt.show()

# Comentário:
# A série apresenta crescimento ao longo do tempo, indicando tendência.
# Também é possível observar ciclos que se repetem anualmente, indicando sazonalidade.

# ### 4.1 Médias móveis

df["media_movel_3"] = df["passageiros"].rolling(window=3).mean()
df["media_movel_12"] = df["passageiros"].rolling(window=12).mean()

plt.figure(figsize=(14, 5))
plt.plot(df.index, df["passageiros"], label="Série original", alpha=0.55)
plt.plot(df.index, df["media_movel_3"], label="Média móvel 3 meses")
plt.plot(df.index, df["media_movel_12"], label="Média móvel 12 meses")
plt.title("Série com médias móveis")
plt.xlabel("Data")
plt.ylabel("Número de passageiros")
plt.legend()
plt.show()

# Comentário:
# A média móvel de 12 meses evidencia a tendência de crescimento.
# A média móvel de 3 meses suaviza oscilações curtas, mas ainda preserva parte da sazonalidade.

# ### 4.2 Boxplot por mês

df_eda = df.copy()
df_eda["mes"] = df_eda.index.month

dados_por_mes = [df_eda[df_eda["mes"] == mes]["passageiros"] for mes in range(1, 13)]

plt.figure(figsize=(12, 5))
plt.boxplot(dados_por_mes, labels=range(1, 13))
plt.title("Distribuição de passageiros por mês")
plt.xlabel("Mês")
plt.ylabel("Número de passageiros")
plt.show()

# Comentário:
# O boxplot por mês mostra que alguns meses concentram valores maiores,
# indicando um padrão sazonal anual.
# Meses de meio do ano tendem a ter maior movimento de passageiros.

# ### 4.3 Histograma

plt.figure(figsize=(10, 5))
plt.hist(df["passageiros"], bins=20)
plt.title("Histograma do número de passageiros")
plt.xlabel("Número de passageiros")
plt.ylabel("Frequência")
plt.show()

# Comentário:
# O histograma mostra a distribuição dos valores da série.
# Como há tendência de crescimento, os valores aparecem espalhados entre faixas menores e maiores.

# ### 4.4 Heatmap ano × mês para visualizar sazonalidade

tabela_heatmap = df.copy()
tabela_heatmap["ano"] = tabela_heatmap.index.year
tabela_heatmap["mes"] = tabela_heatmap.index.month

matriz = tabela_heatmap.pivot_table(
    values="passageiros",
    index="ano",
    columns="mes",
    aggfunc="mean"
)

plt.figure(figsize=(12, 6))
plt.imshow(matriz, aspect="auto")
plt.colorbar(label="Passageiros")
plt.title("Heatmap — ano × mês")
plt.xlabel("Mês")
plt.ylabel("Ano")
plt.xticks(range(12), range(1, 13))
plt.yticks(range(len(matriz.index)), matriz.index)
plt.show()

# Comentário:
# O heatmap reforça a presença de sazonalidade anual.
# Em vários anos, os meses centrais apresentam valores mais altos que os meses iniciais e finais.

# ## 5. Criação de defasagens e cuidados com vazamento de dados

df_features = df[["passageiros"]].copy()

# Defasagens: sempre usamos valores do passado para evitar vazamento de dados.
df_features["lag_1"] = df_features["passageiros"].shift(1)
df_features["lag_12"] = df_features["passageiros"].shift(12)

# Média móvel deslocada: usa somente valores anteriores ao mês atual.
df_features["rolling_3_shifted"] = df_features["passageiros"].shift(1).rolling(window=3).mean()

# Comentário:
# Surgem valores NaN nas primeiras linhas porque não existem observações passadas suficientes.
# O lag_12, por exemplo, precisa de 12 meses anteriores.
# A média móvel deslocada também precisa de uma janela completa de valores passados.

df_features.head(15)

# ## 6. Detecção de outliers com IQR aplicado ao resíduo

# Resíduo: série menos média móvel de 12 meses
df["residuo_mm12"] = df["passageiros"] - df["media_movel_12"]

residuos_validos = df["residuo_mm12"].dropna()

Q1 = residuos_validos.quantile(0.25)
Q3 = residuos_validos.quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

df["outlier"] = (df["residuo_mm12"] < limite_inferior) | (df["residuo_mm12"] > limite_superior)

datas_outliers = df.index[df["outlier"]].strftime("%Y-%m").tolist()

print("Q1:", round(Q1, 2))
print("Q3:", round(Q3, 2))
print("IQR:", round(IQR, 2))
print("Limite inferior:", round(limite_inferior, 2))
print("Limite superior:", round(limite_superior, 2))
print("\nDatas sinalizadas como outliers:")
for data in datas_outliers:
    print("-", data)

plt.figure(figsize=(14, 5))
plt.plot(df.index, df["passageiros"], label="Série original")
plt.scatter(
    df.index[df["outlier"]],
    df.loc[df["outlier"], "passageiros"],
    s=80,
    label="Outliers detectados"
)
plt.title("Outliers detectados pelo IQR aplicado ao resíduo")
plt.xlabel("Data")
plt.ylabel("Passageiros")
plt.legend()
plt.show()

# Comentário:
# Os pontos destacados são observações que se afastam do comportamento suavizado pela média móvel.
# Como a série tem forte crescimento e sazonalidade, alguns valores altos nos anos finais podem ser sinalizados.

# ## 7. Decomposição da série temporal

# Como os dados são mensais e há sazonalidade anual, usamos period=12.
# Como a amplitude sazonal cresce junto com o nível da série, o modelo multiplicativo é mais adequado.
decomposicao = seasonal_decompose(
    df["passageiros"],
    model="multiplicative",
    period=12
)

fig = decomposicao.plot()
fig.set_size_inches(12, 9)
plt.tight_layout()
plt.show()

# Comentários:
# Observed: mostra a série original, com crescimento e ciclos anuais.
# Trend: mostra a tendência de crescimento ao longo dos anos.
# Seasonal: mostra o padrão que se repete a cada 12 meses.
# Residual: mostra o que não foi explicado pela tendência nem pela sazonalidade.

# ## 8. Estacionariedade

def teste_adf(serie, nome):
    resultado = adfuller(serie.dropna())
    print(f"Teste ADF — {nome}")
    print("Estatística ADF:", round(resultado[0], 4))
    print("p-valor:", round(resultado[1], 4))
    print("Valores críticos:")
    for chave, valor in resultado[4].items():
        print(f"  {chave}: {round(valor, 4)}")

    if resultado[1] < 0.05:
        print("Conclusão: a série é provavelmente estacionária.")
    else:
        print("Conclusão: a série provavelmente NÃO é estacionária.")
    print("-" * 60)

teste_adf(df["passageiros"], "série original")

# Transformação logarítmica e diferenciação sazonal para reduzir tendência e sazonalidade
df["log_passageiros"] = np.log(df["passageiros"])
df["log_diff_12"] = df["log_passageiros"].diff(12)

teste_adf(df["log_diff_12"], "log + diferenciação sazonal de 12 meses")

plt.figure(figsize=(14, 5))
plt.plot(df.index, df["log_diff_12"])
plt.title("Série após log + diferenciação sazonal")
plt.xlabel("Data")
plt.ylabel("Diferença sazonal do log")
plt.show()

# Comentário:
# A transformação logarítmica ajuda a estabilizar a variância.
# A diferenciação sazonal remove parte do padrão anual.

# ## 9. Separação treino/teste no tempo
# 
# Para evitar vazamento de dados, o teste deve ser o período mais recente.  
# Aqui, os últimos 24 meses são reservados para teste.

serie = df["passageiros"].copy()

h = 24
train = serie.iloc[:-h]
test = serie.iloc[-h:]

print("Período de treino:", train.index.min().date(), "até", train.index.max().date())
print("Período de teste:", test.index.min().date(), "até", test.index.max().date())
print("Tamanho treino:", len(train))
print("Tamanho teste:", len(test))

plt.figure(figsize=(14, 5))
plt.plot(train.index, train, label="Treino")
plt.plot(test.index, test, label="Teste")
plt.title("Separação temporal entre treino e teste")
plt.xlabel("Data")
plt.ylabel("Passageiros")
plt.legend()
plt.show()

# ## 10. Funções de avaliação

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def avaliar_modelo(nome, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape_valor = mape(y_true, y_pred)
    return {
        "modelo": nome,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape_valor
    }

# ## 11. Modelo 1 — Baseline sazonal

# Baseline sazonal:
# para cada mês do teste, usa o valor observado no mesmo mês do ano anterior.
baseline_pred = serie.shift(12).loc[test.index]

plt.figure(figsize=(14, 5))
plt.plot(test.index, test, label="Real", marker="o")
plt.plot(test.index, baseline_pred, label="Baseline sazonal", marker="o")
plt.title("Baseline sazonal — previsão no conjunto de teste")
plt.xlabel("Data")
plt.ylabel("Passageiros")
plt.legend()
plt.show()

resultado_baseline = avaliar_modelo("Baseline sazonal", test, baseline_pred)
resultado_baseline

# ## 12. Modelo 2 — Holt-Winters multiplicativo

# Modelo clássico: Holt-Winters com tendência e sazonalidade multiplicativas.
# Essa escolha é coerente porque a amplitude da sazonalidade cresce com o nível da série.
hw_model = ExponentialSmoothing(
    train,
    trend="mul",
    seasonal="mul",
    seasonal_periods=12
).fit(optimized=True)

hw_pred = hw_model.forecast(h)
hw_pred.index = test.index

plt.figure(figsize=(14, 5))
plt.plot(train.index, train, label="Treino")
plt.plot(test.index, test, label="Real", marker="o")
plt.plot(test.index, hw_pred, label="Holt-Winters multiplicativo", marker="o")
plt.title("Holt-Winters multiplicativo — previsão")
plt.xlabel("Data")
plt.ylabel("Passageiros")
plt.legend()
plt.show()

resultado_hw = avaliar_modelo("Holt-Winters multiplicativo", test, hw_pred)
resultado_hw

# ## 13. Modelo 3 — XGBoost com features temporais

# Modelo moderno: XGBoost usando defasagens temporais.
# Importante: as features usam apenas valores passados da série.
df_model = pd.DataFrame({
    "y": serie
})

df_model["lag_1"] = df_model["y"].shift(1)
df_model["lag_2"] = df_model["y"].shift(2)
df_model["lag_3"] = df_model["y"].shift(3)
df_model["lag_12"] = df_model["y"].shift(12)
df_model["rolling_3"] = df_model["y"].shift(1).rolling(3).mean()
df_model["rolling_12"] = df_model["y"].shift(1).rolling(12).mean()
df_model["mes"] = df_model.index.month
df_model["ano"] = df_model.index.year

df_model = df_model.dropna()

train_model = df_model.loc[df_model.index < test.index.min()]
test_model = df_model.loc[df_model.index >= test.index.min()]

X_train = train_model.drop(columns=["y"])
y_train = train_model["y"]

X_test = test_model.drop(columns=["y"])
y_test = test_model["y"]

xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    objective="reg:squarederror"
)

xgb.fit(X_train, y_train)

xgb_pred = pd.Series(xgb.predict(X_test), index=y_test.index)

plt.figure(figsize=(14, 5))
plt.plot(y_test.index, y_test, label="Real", marker="o")
plt.plot(xgb_pred.index, xgb_pred, label="XGBoost", marker="o")
plt.title("XGBoost — previsão no conjunto de teste")
plt.xlabel("Data")
plt.ylabel("Passageiros")
plt.legend()
plt.show()

resultado_xgb = avaliar_modelo("XGBoost com lags", y_test, xgb_pred)
resultado_xgb

# ## 14. Comparação dos modelos

resultados = pd.DataFrame([
    resultado_baseline,
    resultado_hw,
    resultado_xgb
]).sort_values("MAPE (%)")

resultados

plt.figure(figsize=(10, 5))
plt.bar(resultados["modelo"], resultados["MAPE (%)"])
plt.title("Comparação dos modelos por MAPE")
plt.xlabel("Modelo")
plt.ylabel("MAPE (%)")
plt.xticks(rotation=20)
plt.show()

# Comentário:
# O melhor modelo é aquele com menor erro.
# O baseline é importante porque qualquer modelo sofisticado precisa superar uma referência simples.

# ## 15. Gráfico final comparando previsões

plt.figure(figsize=(14, 6))
plt.plot(test.index, test, label="Real", marker="o")
plt.plot(test.index, baseline_pred, label="Baseline sazonal", marker="o")
plt.plot(test.index, hw_pred, label="Holt-Winters multiplicativo", marker="o")
plt.plot(xgb_pred.index, xgb_pred, label="XGBoost com lags", marker="o")
plt.title("Comparação das previsões no período de teste")
plt.xlabel("Data")
plt.ylabel("Passageiros")
plt.legend()
plt.show()

# ## 16. Conclusão
# 
# A série **Air Passengers** apresenta tendência de crescimento e sazonalidade anual bem definida.  
# A análise exploratória mostrou aumento no número de passageiros ao longo dos anos, além de padrões mensais recorrentes.
# 
# A decomposição multiplicativa foi escolhida porque a amplitude sazonal aumenta conforme o nível da série cresce.  
# O teste ADF indicou que a série original não é estacionária, o que é esperado em séries com tendência e sazonalidade. Após transformação logarítmica e diferenciação sazonal, a série se torna mais adequada para modelagens que exigem estacionariedade.
# 
# Na modelagem, foram comparados três métodos:
# 
# 1. **Baseline sazonal**, como referência simples;
# 2. **Holt-Winters multiplicativo**, como modelo clássico de séries temporais;
# 3. **XGBoost com lags**, como modelo moderno baseado em aprendizado de máquina.
# 
# O melhor modelo deve ser escolhido com base na menor combinação de MAE, RMSE e MAPE, dando prioridade ao MAPE por facilitar a interpretação percentual do erro.
# 
# ## Limitações
# 
# - A base é pequena, com apenas 144 observações mensais;
# - O XGBoost pode ter desempenho limitado por haver poucos dados;
# - A série é univariada, sem variáveis externas como preço, renda, feriados ou eventos econômicos;
# - O período histórico é antigo, então a conclusão é adequada para fins didáticos, não para uso operacional atual.
# 
# ## Resposta final às perguntas do projeto
# 
# **a) Existe tendência?**  
# Sim. Há crescimento claro no número de passageiros ao longo dos anos.
# 
# **b) Existe sazonalidade?**  
# Sim. Existe sazonalidade anual, com padrões que se repetem a cada 12 meses.
# 
# **c) Há comportamento atípico?**  
# Sim. A regra do IQR aplicada ao resíduo sinalizou alguns pontos fora do comportamento esperado. Esses pontos foram destacados no gráfico de outliers.

# # Roteiro curto para apresentação/pitch
# 
# 1. **Problema escolhido:** previsão mensal de passageiros aéreos usando a série Air Passengers.
# 2. **Por que essa série?** Ela possui tendência e sazonalidade clara, sendo adequada para aplicar o pipeline do curso.
# 3. **EDA:** os gráficos mostram crescimento ao longo dos anos e padrão sazonal anual.
# 4. **Decomposição:** o modelo multiplicativo foi escolhido porque a sazonalidade cresce junto com o nível da série.
# 5. **Modelos:** foram comparados baseline sazonal, Holt-Winters multiplicativo e XGBoost com lags.
# 6. **Avaliação:** a comparação foi feita por MAE, RMSE e MAPE.
# 7. **Conclusão:** o melhor modelo é aquele com menor erro, mas o baseline foi mantido como referência obrigatória para validar se os modelos mais sofisticados realmente agregaram valor.
