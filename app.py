import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import plotly.express as px

st.set_page_config(page_title="Uber Demand Forecasting", layout="wide")

st.title("🚖 Uber Demand Forecasting (Live Training App)")

# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────
st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader("Upload uber_nyc_enriched.csv", type=["csv"])

if uploaded_file is None:
    st.warning("Upload dataset to proceed")
    st.stop()

df = pd.read_csv(uploaded_file)

# ─────────────────────────────────────────────
# SHOW DATA
# ─────────────────────────────────────────────
st.subheader("📊 Raw Dataset Preview")
st.dataframe(df.head())

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
st.subheader("⚙️ Preprocessing")

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
    df["hday"] = df["hday"].map({"Y":1,"N":0}).fillna(0)

borough_map = {b:i for i,b in enumerate(df["borough"].dropna().unique())}
df["cluster_id"] = df["borough"].map(borough_map)

# Aggregation
grp = ["cluster_id","Hour","DayOfWeek","Month"]

weather = df.groupby(grp)[["temp","spd","vsb","pcp01"]].mean().reset_index()
flags = df.groupby(grp)[["hday","IsWeekend","IsRushHour"]].max().reset_index()
pickups = df.groupby(grp)["pickups"].sum().reset_index()

df_agg = pickups.merge(weather,on=grp).merge(flags,on=grp)

st.write("Aggregated Data")
st.dataframe(df_agg.head())

# ─────────────────────────────────────────────
# TRAIN MODELS
# ─────────────────────────────────────────────
FEATURES = [
    "cluster_id","Hour","DayOfWeek","Month",
    "temp","spd","vsb","pcp01",
    "hday","IsWeekend","IsRushHour"
]

if st.button("🚀 Train Models"):

    df_clean = df_agg[FEATURES + ["pickups"]].dropna()
    y = df_clean["pickups"].values
    X_dict = df_clean[FEATURES].to_dict(orient="records")

    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(X_dict)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100),
        "XGBoost": XGBRegressor(n_estimators=100)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        results[name] = {
            "RMSE": rmse(y_test, pred),
            "R2": model.score(X_test, y_test)
        }

    # Show metrics
    st.subheader("📊 Model Performance")

    res_df = pd.DataFrame(results).T
    st.dataframe(res_df)

    fig = px.bar(res_df, y="RMSE", title="RMSE Comparison")
    st.plotly_chart(fig, use_container_width=True)

    st.success("✅ Models trained successfully!")

# ─────────────────────────────────────────────
# PREDICTION SECTION
# ─────────────────────────────────────────────
st.sidebar.header("🔮 Predict Demand")

hour = st.sidebar.slider("Hour", 0, 23, 8)
temp = st.sidebar.slider("Temperature", 0, 100, 60)
spd = st.sidebar.slider("Wind Speed", 0, 40, 10)

st.info("Train models first to enable prediction")
