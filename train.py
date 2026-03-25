# ==============================================================================
# Uber Demand Forecasting - Model Training Pipeline
# Author: Harshit Belwal | Institution: KIIT - DU
# ==============================================================================

import os
import pickle
import json

import kagglehub
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


# ─────────────────────────────────────────────
# 1. DATA DOWNLOAD
# ─────────────────────────────────────────────

def download_data() -> str:
    """Download the Uber NYC enriched dataset from Kaggle."""
    print("1. Downloading dataset...")
    folder_path = kagglehub.dataset_download("yannisp/uber-pickups-enriched")
    csv_path = os.path.join(folder_path, "uber_nyc_enriched.csv")
    return csv_path


# ─────────────────────────────────────────────
# 2. PREPROCESSING & AGGREGATION
# ─────────────────────────────────────────────

def preprocess(csv_path: str):
    """
    Load raw data, engineer spatio-temporal features, and aggregate
    pickups into discrete time windows (hourly) per spatial cluster.
    """
    print("2. Preprocessing & aggregating spatio-temporal data...")
    df = pd.read_csv(csv_path)

    # ── Temporal features ──────────────────────────────────────────
    df["pickup_dt"] = pd.to_datetime(df["pickup_dt"])
    df["Hour"]      = df["pickup_dt"].dt.hour
    df["DayOfWeek"] = df["pickup_dt"].dt.dayofweek
    df["Month"]     = df["pickup_dt"].dt.month
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

    # Rush-hour flag: 7-9 AM or 5-7 PM on weekdays
    df["IsRushHour"] = (
        (df["IsWeekend"] == 0) &
        (df["Hour"].isin([7, 8, 9, 17, 18, 19]))
    ).astype(int)

    # ── Holiday encoding ───────────────────────────────────────────
    if "hday" in df.columns:
        df["hday"] = df["hday"].map({"Y": 1, "N": 0, 1: 1, 0: 0}).fillna(0).astype(int)

    # ── Spatial clustering (borough as spatial proxy) ──────────────
    borough_map = {b: i for i, b in enumerate(df["borough"].dropna().unique())}
    df["cluster_id"] = df["borough"].map(borough_map)

    with open("borough_map.json", "w") as f:
        json.dump(borough_map, f)

    # ── Aggregate into hourly time windows per spatial cluster ─────
    agg_cols = ["temp", "spd", "vsb", "pcp01"]
    grp_keys = ["cluster_id", "Hour", "DayOfWeek", "Month"]

    weather_agg  = df.groupby(grp_keys)[agg_cols].mean().reset_index()
    flag_agg     = df.groupby(grp_keys)[["hday", "IsWeekend", "IsRushHour"]].max().reset_index()
    pickup_agg   = df.groupby(grp_keys)["pickups"].sum().reset_index()

    df_agg = pickup_agg.merge(weather_agg, on=grp_keys).merge(flag_agg, on=grp_keys)
    print(f"   Aggregated dataset shape: {df_agg.shape}")
    return df_agg, borough_map


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING & VECTORISATION
# ─────────────────────────────────────────────

FEATURES = [
    "cluster_id", "Hour", "DayOfWeek", "Month",
    "temp", "spd", "vsb", "pcp01",
    "hday", "IsWeekend", "IsRushHour",
]


def build_features(df_agg: pd.DataFrame):
    """Return X matrix, y target, and fitted DictVectorizer."""
    print("3. Vectorizing features...")
    df_clean = df_agg[FEATURES + ["pickups"]].dropna()
    y = df_clean["pickups"].values
    records = df_clean[FEATURES].to_dict(orient="records")
    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(records)
    return X, y, vec


# ─────────────────────────────────────────────
# 4. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_and_evaluate(X_train, X_test, y_train, y_test) -> dict:
    """Train Linear Regression, Random Forest, and XGBoost; return models + metrics."""
    print("4. Training models...")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost":           XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }

    results = {}
    for name, model in models.items():
        print(f"   Training {name}...")
        model.fit(X_train, y_train)

        results[name] = {
            "model":      model,
            "train_rmse": round(rmse(y_train, model.predict(X_train)), 4),
            "test_rmse":  round(rmse(y_test,  model.predict(X_test)),  4),
            "train_r2":   round(model.score(X_train, y_train),          4),
            "test_r2":    round(model.score(X_test,  y_test),           4),
        }
        print(f"   ✓ {name}: test RMSE={results[name]['test_rmse']:.2f}, R²={results[name]['test_r2']:.4f}")

    return results


# ─────────────────────────────────────────────
# 5. SAVE ARTIFACTS
# ─────────────────────────────────────────────

def save_artifacts(results: dict, vec: DictVectorizer) -> None:
    """Persist models, vectorizer, and metrics to disk."""
    print("5. Saving artifacts...")

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vec, f)

    metrics = {}
    for name, res in results.items():
        safe = name.lower().replace(" ", "_")
        with open(f"model_{safe}.pkl", "wb") as f:
            pickle.dump(res["model"], f)
        metrics[name] = {k: v for k, v in res.items() if k != "model"}

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Artifacts saved: model_*.pkl  |  vectorizer.pkl  |  metrics.json  |  borough_map.json")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    csv_path            = download_data()
    df_agg, borough_map = preprocess(csv_path)
    X, y, vec           = build_features(df_agg)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    save_artifacts(results, vec)


if __name__ == "__main__":
    main()
