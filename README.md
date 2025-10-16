# 📈 Time Series Forecasting of Bike Demand with Machine Learning
This project presents a complete time series forecasting pipeline for a bike-sharing demand dataset, developed in Python. The work focuses on modeling, feature engineering, and performance evaluation using multiple regression algorithms.

The goal was to identify the model that best captures temporal dependencies and generalizes effectively to unseen data. After extensive experimentation, the HistGradientBoostingRegressor (HGB) was selected as the final model.

---

## 🧩 Project Overview

This project implements a **machine learning-based approach** for forecasting a time-dependent variable.  
The main objectives were:

- To perform robust **feature engineering** for time series data.
- To encode cyclical features (e.g., hours, days, months) effectively.
- To generate **lagged** and **rolling features** to capture temporal dependencies.
- To evaluate several **regression models** using **cross-validation with `TimeSeriesSplit`**.
- To identify the model with the best generalization performance.
- To train and validate the final model (HGB) on real data.

---

## ⚙️ Workflow Summary

### 1. Data Preprocessing

Before modeling, data cleaning and preparation steps were applied to ensure reliability and consistency:

- Calculate the coefficiente of variation of some features, to understand their variability.
- Encoding of Cyclical Features from temporal columns, such as hour, day of the week, and month, to help the model capture temporal patterns.
- Consolidated the rare heavy_rain category (only 3 instances) into the broader rain category to reduce noise and prevent model overfitting.
- Normalization of numeric values where applicable.

---

### 2. Feature Engineering

To enrich the dataset and improve predictive power, several feature transformations were performed.

#### 🔁 Lagged Features

Lagged features were created to allow the model to capture temporal dependencies by incorporating past values of the target variable. The specific lags were selected based on insights from ACF and PACF plots. The following lagged features were generated:

- 25-hour lag for "temp", "feel_temp", "humidity", and "windspeed".
- 143-hour lag for the same set of features.

#### 📉 Rolling Features

Rolling window statistics were applied to create smoothed representations of the series and capture recent variability. The window sizes were determined using the same analytical method as the lags. The generated features include:

- 48-hour rolling mean and standard deviation for "temp", "feel_temp", "humidity", and "windspeed".
- 168-hour rolling mean and standard deviation for the same columns.

These features are designed to capture medium-term trends and volatility in the data.

### 3. Data Splitting and Cross-Validation

Instead of a random split, TimeSeriesSplit was used to maintain the chronological order of the data and ensure realistic model evaluation.

This technique prevents data leakage and mimics real-world forecasting scenarios where future data is never available during training.

### 4. Model Evaluation

Multiple algorithms and metrics were tested to assess performance. Each model was evaluated using cross-validation, and the following metrics were calculated for both training and testing sets:

- **NMAE Negative Mean Absolute Error**

- RIDGE: -0.093172 (0.008721)
- LASSO: -0.092691 (0.008623)
- ENET: -0.092693 (0.008624)
- NN (MLP): -0.063116 (0.006311)
- RF: -0.076823 (0.007322)
- XGBoost: -0.062710 (0.004100)
- LGBM: -0.058437 (0.004295)
- HGB: -0.058410 (0.004338)

- **R² Coefficient of Determination**

- RIDGE: 0.601190 (0.085044)
- LASSO: 0.603184 (0.087896)
- ENET: 0.603186 (0.087893)
- NN: 0.808309 (0.043714)
- RF: 0.733283 (0.056201)
- XG: 0.814273 (0.048524)
- LGBM: 0.823377 (0.058416)
- HGB: 0.823732 (0.048395)

### 5. Final Model: HistGradientBoostingRegressor (HGB)

The HGB model was trained on the full training dataset and evaluated on the test set.
It showed excellent performance with low errors and strong predictive power.

### 📊 Model Performance Analysis

#### High R² (Train: 0.9886, Test: 0.8317):
- The model captures most of the variance while maintaining good generalization.

#### Low MAE and RMSE:
- The predictions remain close to real values both in training and testing.

#### Minimal Bias:
- Slight underestimation observed in the test set, which can be addressed with calibration or bias correction.

#### No Overfitting:
- The small gap between training and testing performance suggests that the model learned general patterns rather than memorizing data.


### 🔍 Residuals Analysis

Residuals (prediction errors) are centered around zero and show no visible trend or autocorrelation.
This indicates that the model made balanced predictions and did not systematically overestimate or underestimate the target variable.


### 🧠 Why HistGradientBoostingRegressor?

The HGB model was chosen because it combines:

- Strong predictive performance on structured tabular data.

- Ability to model nonlinear relationships.

- Smooth regularization mechanisms, reducing overfitting.

- It achieved the best trade-off between accuracy, interpretability, and computational cost.


### 🧑‍💻 Author

##### **Bernardo Costa**

- Data Analyst & Data Scientist in graduation
- 📍 Brazil
- 💼 LinkedIn -> https://www.linkedin.com/in/bernardobadc/
- 📊 GitHub -> https://github.com/bernardobadc

-------------------------------------------------------------------------------------------------------------------------------------------------

# PORTUGUESE

# 📈 Previsão de Séries Temporais da Demanda de Bicicletas com Machine Learning

Este projeto apresenta um pipeline completo de previsão de séries temporais para um conjunto de dados de demanda de bicicletas compartilhadas, desenvolvido em Python. O trabalho foca na modelagem, engenharia de features e avaliação de desempenho usando múltiplos algoritmos de regressão.

O objetivo foi identificar o modelo que melhor captura dependências temporais e generaliza efetivamente para dados não vistos. Após extensiva experimentação, o **HistGradientBoostingRegressor (HGB)** foi selecionado como modelo final.

---

## 🧩 Visão Geral do Projeto

Este projeto implementa uma **abordagem baseada em machine learning** para prever uma variável dependente do tempo.  
Os principais objetivos foram:

- Realizar **engenharia de features** robusta para dados de séries temporais.
- Codificar features cíclicas (como hora, dia, mês) de forma efetiva.
- Gerar **features defasadas (lagged)** e **features de rolling** para capturar dependências temporais.
- Avaliar vários **modelos de regressão** usando **validação cruzada com `TimeSeriesSplit`**.
- Identificar o modelo com melhor desempenho de generalização.
- Treinar e validar o modelo final (HGB) em dados reais.

---

## ⚙️ Resumo do Fluxo de Trabalho

### 1. Pré-processamento de Dados

Antes da modelagem, etapas de limpeza e preparação de dados foram aplicadas para garantir confiabilidade e consistência:

- Cálculo do coeficiente de variação de algumas features para entender sua variabilidade.
- Codificação de Features Cíclicas a partir de colunas temporais, como hora, dia da semana e mês, para ajudar o modelo a capturar padrões temporais.
- Consolidação da categoria rara `heavy_rain` (apenas 3 instâncias) na categoria mais ampla `rain` para reduzir ruído e prevenir overfitting do modelo.
- Normalização de valores numéricos quando aplicável.

---

### 2. Engenharia de Features

Para enriquecer o conjunto de dados e melhorar o poder preditivo, várias transformações de features foram realizadas.

#### 🔁 Features Defasadas (Lagged Features)

Features defasadas foram criadas para permitir que o modelo capture dependências temporais incorporando valores passados da variável target. Os lags específicos foram selecionados com base em insights dos gráficos ACF e PACF. As seguintes features defasadas foram geradas:

- Lag de 25 horas para "temp", "feel_temp", "humidity" e "windspeed".
- Lag de 143 horas para o mesmo conjunto de features.

#### 📉 Features de Rolling

Estatísticas de janela móvel (rolling) foram aplicadas para criar representações suavizadas da série e capturar variabilidade recente. Os tamanhos das janelas foram determinados usando o mesmo método analítico dos lags. As features geradas incluem:

- Média móvel e desvio padrão de 48 horas para "temp", "feel_temp", "humidity" e "windspeed".
- Média móvel e desvio padrão de 168 horas para as mesmas colunas.

Essas features foram projetadas para capturar tendências de médio prazo e volatilidade nos dados.

### 3. Divisão de Dados e Validação Cruzada

Em vez de uma divisão aleatória, o `TimeSeriesSplit` foi usado para manter a ordem cronológica dos dados e garantir uma avaliação realista do modelo.

Esta técnica previne vazamento de dados e simula cenários reais de previsão onde dados futuros nunca estão disponíveis durante o treinamento.

### 4. Avaliação dos Modelos

Múltiplos algoritmos e métricas foram testados para avaliar o desempenho. Cada modelo foi avaliado usando validação cruzada, e as seguintes métricas foram calculadas para conjuntos de treino e teste:

- **NMAE (Negative Mean Absolute Error)**

- RIDGE: -0.093172 (0.008721)
- LASSO: -0.092691 (0.008623)
- ENET: -0.092693 (0.008624)
- NN (MLP): -0.063116 (0.006311)
- RF: -0.076823 (0.007322)
- XGBoost: -0.062710 (0.004100)
- LGBM: -0.058437 (0.004295)
- HGB: -0.058410 (0.004338)

- **R² (Coeficiente de Determinação)**

- RIDGE: 0.601190 (0.085044)
- LASSO: 0.603184 (0.087896)
- ENET: 0.603186 (0.087893)
- NN: 0.808309 (0.043714)
- RF: 0.733283 (0.056201)
- XG: 0.814273 (0.048524)
- LGBM: 0.823377 (0.058416)
- HGB: 0.823732 (0.048395)

### 5. Modelo Final: HistGradientBoostingRegressor (HGB)

O modelo HGB foi treinado em todo o conjunto de dados de treinamento e avaliado no conjunto de teste. Ele mostrou excelente desempenho com baixos erros e forte poder preditivo.

### 📊 Análise de Desempenho do Modelo

#### Alto R² (Treino: 0.9886, Teste: 0.8317):
- O modelo captura a maior parte da variância enquanto mantém boa generalização.

#### Baixo MAE e RMSE:
- As previsões permanecem próximas dos valores reais tanto no treinamento quanto no teste.

#### Viés Mínimo:
- Subestimação leve observada no conjunto de teste, que pode ser abordada com calibração ou correção de viés.

#### Sem Overfitting:
- A pequena diferença entre o desempenho de treino e teste sugere que o modelo aprendeu padrões gerais em vez de memorizar os dados.

### 🔍 Análise de Resíduos

Os resíduos (erros de previsão) estão centrados em torno de zero e não mostram tendência visível ou autocorrelação. Isso indica que o modelo fez previsões balanceadas e não superestimou ou subestimou sistematicamente a variável target.

### 🧠 Por que HistGradientBoostingRegressor?

O modelo HGB foi escolhido porque combina:

- Forte desempenho preditivo em dados tabulares estruturados.
- Capacidade de modelar relações não lineares.
- Mecanismos de regularização suaves, reduzindo overfitting.
- Conseguiu o melhor equilíbrio entre acurácia, interpretabilidade e custo computacional.

### 🧑‍💻 Autor

##### **Bernardo Costa**

- Data Analyst & Data Scientist em formação
- 📍 Brasil
- 💼 LinkedIn -> https://www.linkedin.com/in/bernardobadc/
- 📊 GitHub -> https://github.com/bernardobadc