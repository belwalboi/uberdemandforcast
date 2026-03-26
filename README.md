# Uber Demand Forecasting

## Problem Statement
Accurately predicting ride-request volume for a specific area and time is critical for marketplace balance.

- If demand is **under-predicted**, riders may face longer wait times and higher surge pricing.
- If demand is **over-predicted**, too many drivers may be positioned in low-demand zones, causing idle time and reduced earnings.

This project analyzes historical Uber pickup data to learn recurring spatio-temporal patterns (for example, weekday commute peaks) and forecast expected ride volume in advance.

## Objectives
1. **Preprocess and aggregate raw spatio-temporal data**
   - Convert raw pickup timestamps into engineered temporal signals (hour, day of week, month, weekend/rush-hour flags).
   - Represent space using borough-based clusters.
   - Aggregate records into discrete hourly windows by cluster.

2. **Train three forecasting models**
   - Linear Regression
   - Random Forest Regressor
   - XGBoost Regressor

3. **Evaluate and compare model performance**
   - Use Root Mean Square Error (RMSE) as the primary evaluation metric.
   - Track train/test RMSE and R² to compare fit quality and generalization.

## Project Workflow
1. Download the enriched Uber NYC dataset.
2. Engineer temporal and weather-related features.
3. Aggregate pickup counts by spatial cluster and time bucket.
4. Vectorize features with `DictVectorizer`.
5. Train and evaluate all three models.
6. Save trained artifacts (`model_*.pkl`, `vectorizer.pkl`, `metrics.json`, `borough_map.json`).

## Repository Files
- `train.py` – end-to-end training pipeline and artifact export.
- `app.py` – Streamlit app for model training (first run) and demand prediction.
- `uber_nyc_enriched.csv` – local copy of the dataset (if present).
- `requirements.txt` – Python dependencies.

## Quick Start
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Evaluation Output
After training, the pipeline writes:
- `metrics.json` containing per-model:
  - `train_rmse`
  - `test_rmse`
  - `train_r2`
  - `test_r2`

Use these metrics to identify the best-performing model for deployment in the demand allocation workflow.
