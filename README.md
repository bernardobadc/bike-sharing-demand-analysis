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

**Bernardo Costa**
Data Analyst & Data Scientist in graduation
📍 Brazil
💼 LinkedIn -> https://www.linkedin.com/in/bernardobadc/
📊 GitHub -> https://github.com/bernardobadc