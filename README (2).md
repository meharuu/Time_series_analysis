# ⚡ Household Energy Forecasting & Anomaly Detection

Forecasting hourly household electricity demand with **SARIMA** and **Prophet**, and measuring how **anomaly detection and cleaning** affect forecast accuracy.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![statsmodels](https://img.shields.io/badge/statsmodels-SARIMA-informational) ![Prophet](https://img.shields.io/badge/Prophet-1.2-informational) ![scikit--learn](https://img.shields.io/badge/scikit--learn-Isolation%20Forest-orange)

---

## Overview

This project is an end-to-end time-series pipeline built on **2+ million minute-level readings** from a real household (Dec 2006 – Nov 2010). It goes from raw, noisy sensor data to validated forecasts:

1. **Clean and prepare** the raw series (missing values, datetime indexing, hourly frequency)
2. **Diagnose** it (stationarity tests, autocorrelation, seasonal decomposition)
3. **Forecast** the next 7 days with SARIMA and Prophet
4. **Explain** the results with residual diagnostics, not just a single error score
5. **Detect and clean anomalies**, then retrain both models and compare

The aim is interpretable, well-diagnosed forecasting: understanding *why* one model beats another.

## Key Results

Out-of-sample forecast of the final **168 hours (7 days)**:

| Model | RMSE (kW) | MAE (kW) |
|---|---|---|
| **SARIMA (1,1,1)(1,1,1)₂₄** | **0.966** | **0.607** |
| Prophet (daily + weekly seasonality) | 1.016 | 0.722 |

**SARIMA outperformed Prophet, with about 16% lower MAE.** Its explicit 24-hour seasonal terms capture the household's strong daily cycle, and the residual diagnostics show where each model still leaves structure unexplained.

After Isolation Forest anomaly cleaning, both models were retrained and re-evaluated:

| Model | RMSE before → after | MAE before → after |
|---|---|---|
| SARIMA | 0.966 → 0.744 | 0.607 → 0.572 |
| Prophet | 1.016 → 0.810 | 0.722 → 0.669 |

> **Note:** In the current version, anomaly cleaning is applied to the full series, including the test window, so the "after" scores are measured against a cleaned ground truth. See [Limitations & Next Steps](#limitations--next-steps).

## Dataset

**[UCI – Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)**

| Property | Value |
|---|---|
| Records | 2,075,259 minute-level readings |
| Period | 16 Dec 2006 – 26 Nov 2010 (~4 years) |
| Missing values | 25,979 rows (~1.25%), encoded as `?` |
| Target | `Global_active_power` (kW) |

## Pipeline

### 1. Data preparation
- Parsed `?` as missing values and merged `Date` + `Time` into a sorted `DatetimeIndex`
- Filled gaps with **time-weighted interpolation**, which respects the actual time between readings and avoids artificial jumps in energy usage
- Converted the minute-level series to an **hourly** frequency

### 2. Stationarity & structure
- **Augmented Dickey–Fuller test**: ADF statistic −14.82, p ≈ 2e-27
- First-order differencing, with **ACF / PACF** plots over 48 lags to inspect temporal dependence

### 3. Decomposition
- **Classical additive decomposition** and **STL** (Seasonal-Trend decomposition using Loess), both with a 24-hour period
- STL is preferred because it is more robust to outliers

### 4. Forecasting
- **SARIMA (1,1,1)(1,1,1)₂₄**: explicit modeling of trend, daily seasonality and AR/MA dynamics (AIC 84,881, BIC 84,923)
- **Prophet**: additive model with a piecewise-linear trend plus daily and weekly seasonality
- **Chronological train/test split**: the last 168 hours are held out, so the models never see the test period during training

### 5. Residual diagnostics
- Residuals over time, residual distributions and residual ACF
- **Forecast error by hour of day**, which shows *when* each model struggles, not only how much

### 6. Anomaly detection & cleaning
- **Isolation Forest** (unsupervised, 1% contamination) to flag anomalous readings
- Flagged points are removed and **re-imputed with time-based interpolation**, keeping the series continuous

### 7. Re-forecasting
- Both models are retrained on the cleaned series and compared with their original scores

## Tech Stack

**Python 3.12** · pandas · NumPy · statsmodels (SARIMAX, STL, ADF) · Prophet · scikit-learn · Matplotlib · Seaborn · Jupyter

## Getting Started

```bash
git clone https://github.com/meharuu/Time_series_analysis.git
cd Time_series_analysis

pip install pandas numpy matplotlib seaborn statsmodels scikit-learn prophet jupyter
```

1. Download `household_power_consumption.txt` from the [UCI repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) and place it in the project root
2. Run `jupyter notebook pipeline.ipynb`

## Repository Structure

```
Time_series_analysis/
├── pipeline.ipynb   # Full pipeline: EDA → diagnostics → forecasting → anomalies → re-forecasting
└── README.md
```

## Limitations & Next Steps

- **Hourly aggregation:** switch from point sampling (`asfreq`) to `resample('h').mean()`, so each hour reflects the average load across all 60 readings
- **Leakage-free anomaly evaluation:** fit the Isolation Forest and clean data on the training window only, then score against the *original* test values
- **Contextual anomalies:** Isolation Forest on a single value mostly flags unusually high loads. Adding hour-of-day and day-of-week features, or thresholding SARIMA residuals, would catch readings that are unusual *for their time*
- **Model selection:** grid-search SARIMA orders using AIC/BIC, and use rolling-origin cross-validation instead of a single 7-day window
- **More models:** gradient-boosted trees with lag and calendar features (LightGBM/XGBoost), and a deep-learning baseline (LSTM / N-BEATS)

## Author

**[@meharuu](https://github.com/meharuu)** – AI Engineer & Data Scientist · meharoo261@gmail.com
