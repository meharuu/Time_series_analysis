Household Power Consumption Time Series Analysis & Forecasting
Overview

This repository presents an end-to-end time series analysis pipeline applied to the UCI Household Power Consumption dataset, focusing on hourly global active power usage for a single household.

The project covers:

Exploratory data analysis and preprocessing

Stationarity diagnostics and decomposition

Forecasting with SARIMA and Prophet

Anomaly detection using statistical and machine-learning methods

Re-forecasting after anomaly cleaning

Comparative evaluation before vs after anomaly handling

The goal is not only predictive accuracy, but also model interpretability, diagnostics, and assumption validation, consistent with master’s-level time-series analysis standards.

Dataset

Source:
UCI Machine Learning Repository – Individual Household Electric Power Consumption

Key Characteristics:

Originally recorded at 1-minute resolution

Converted to hourly frequency

Contains real-world noise, missing values, and anomalies

Target Variable:

Global_active_power (kilowatts)

Project Pipeline
1. Data Loading & Initial Exploration

Load raw .txt dataset with proper missing value handling (? → NaN)

Inspect shape, dtypes, summary statistics, and missing values

Visualize raw power consumption series

2. Datetime Processing & Resampling

Combine Date and Time into a unified DatetimeIndex

Sort chronologically

Resample to hourly frequency

Time-weighted interpolation for missing values

Why this matters:
Time-based interpolation preserves physical continuity in energy usage and avoids artificial step changes.

3. Stationarity Analysis

Augmented Dickey–Fuller (ADF) test

First-order differencing if non-stationary

ACF / PACF plots to inspect temporal dependence

4. Time Series Decomposition

Two decomposition techniques are applied:

Classical Additive Decomposition

STL (Seasonal–Trend decomposition using Loess)

Both extract:

Trend

Daily seasonality (24-hour cycle)

Residual (noise + anomalies)

STL is preferred for robustness to outliers.

5. Forecasting Models
SARIMA

Explicit modeling of:

Trend (differencing)

Daily seasonality (24-hour cycle)

Autoregressive and moving-average dynamics

Model selection supported by AIC/BIC

Prophet

Additive model with:

Piecewise linear trend

Daily and weekly seasonalities

Robust to missing data and moderate outliers

6. Forecast Evaluation

Models are evaluated using a time-aware train–test split:

Last 168 hours (7 days) reserved for testing

Metrics:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

AIC / BIC (SARIMA only)

This ensures out-of-sample evaluation with no information leakage.

7. Residual Diagnostics

To explain why one model performs better:

Residual time-series plots

Residual distributions (histograms)

Residual autocorrelation (ACF)

Error by hour-of-day analysis

These diagnostics demonstrate how effectively each model removes temporal dependence.

8. Anomaly Detection

At least two anomaly detection methods are applied:

Statistical

Z-score–based reasoning (implicitly via residual analysis)

SARIMA residual thresholding

Machine Learning

Isolation Forest (unsupervised, distribution-free)

Anomalies are visualized directly on the time series.

9. Anomaly Cleaning

Detected anomalies are:

Removed (set to NaN)

Re-imputed using time-based interpolation

This produces a cleaned time series while preserving temporal continuity.

10. Re-Forecasting After Cleaning

Both SARIMA and Prophet are:

Re-trained on cleaned data

Re-evaluated on the same forecast horizon

Performance is compared before vs after anomaly handling, demonstrating the impact of data quality on forecasting accuracy.
