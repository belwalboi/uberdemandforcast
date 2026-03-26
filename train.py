# ==============================================================================
# Uber Demand Forecasting - Training Script (LOCAL DATASET VERSION)
# Author: Harshit Belwal
# Run: python train.py
# ==============================================================================

import os
import pickle
import json

import numpy as np
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


CSV_PATH = "uber_nyc_enriched.csv"

FEATURES = [
    "cluster_id", "Hour", "DayOfWeek", "Month",
    "temp", "spd", "vsb", "pcp01",
    "hday", "IsWeekend", "IsRushHour",
]


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")
    print("📂 Loading dataset...")
    return pd.read_csv(CSV_PATH)


# ─────────────────────────────────────────────
# 2. PREPROCESS
# ─────────────────────────────────────────────
def preprocess(df):
    print("⚙️ Preprocessing...")

    df["pickup_dt"] = pd.to_datetime(df["pickup_dt"])
    df["Hour"] = df["pickup_dt"].dt.hour
    df["DayOfWeek"] = df["pickup_dt"].dt.dayofweek
    df["Month"] = df["pickup_dt"].dt.month
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

    df["IsRushHour"] = (
        (df["IsWeekend"] == 0) &
        (df["Hour"].isin([7,8,9,17,18,19]))
    ).astype(int)

    if "hday" in df.columns:
        df["hday"] = df["hday"].map({"Y":1,"N":0}).fillna(0).astype(int)

    # Spatial encoding
    borough_map = {b:i for i,b in enumerate(df["borough"].dropna().unique())}
    df["cluster_id"] = df["borough"].map(borough_map)

    with open("borough_map.json", "w") as f:
        json.dump(borough_map, f)

    # Aggregation
    grp = ["cluster_id","Hour","DayOfWeek","Month"]

    weather = df.groupby(grp)[["temp","spd","vsb","pcp01"]].mean().reset_index()
    flags = df.groupby(grp)[["hday","IsWeekend","IsRushHour"]].max().reset_index()
    pickups = df.groupby(grp)["pickups"].sum().reset_index()

    df_agg = pickups.merge(weather,on=grp).merge(flags,on=grp)

    print(f"✅ Aggregated shape: {df_agg.shape}")
    return df_agg


# ─────────────────────────────────────────────
# 3. FEATURE BUILDING
# ─────────────────────────────────────────────
def build_features(df_agg):
    print("🔢 Building features...")

    df_clean = df_agg[FEATURES + ["pickups"]].dropna()
    y = df_clean["pickups"].values
    X_dict = df_clean[FEATURES].to_dict(orient="records")

    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(X_dict)

    return X, y, vec


# ─────────────────────────────────────────────
# 4. TRAIN MODELS
# ─────────────────────────────────────────────
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_models(X_train, y_train, X_test, y_test):
    print("🏋️ Training models...")

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "xgboost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }

    results = {}

    for name, model in models.items():
        print(f"➡️ Training {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        results[name] = {
            "model": model,
            "rmse": rmse(y_test, y_pred),
            "r2": model.score(X_test, y_test),
        }

        print(f"   ✅ RMSE: {results[name]['rmse']:.2f}, R²: {results[name]['r2']:.4f}")

    return results


# ─────────────────────────────────────────────
# 5. SAVE ARTIFACTS
# ─────────────────────────────────────────────
def save_artifacts(results, vec):
    print("💾 Saving models...")

    # Save vectorizer
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vec, f)

    metrics = {}

    for name, res in results.items():
        with open(f"model_{name}.pkl", "wb") as f:
            pickle.dump(res["model"], f)

        metrics[name] = {
            "rmse": res["rmse"],
            "r2": res["r2"]
        }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Saved: models + vectorizer + metrics")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    df = load_data()
    df_agg = preprocess(df)

    X, y, vec = build_features(df_agg)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = train_models(X_train, y_train, X_test, y_test)

    save_artifacts(results, vec)


if __name__ == "__main__":
    main()
