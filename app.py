# ==============================================================================
# Uber Demand Forecasting - Streamlit Dashboard (Self-contained)
# Author: Harshit Belwal | Institution: KIIT - DU
# Run: streamlit run app.py
# ==============================================================================

import json
import os
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Uber Demand Forecasting",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.5px; }

  [data-testid="stSidebar"] { background: #0d0d0d !important; }
  [data-testid="stSidebar"] * { color: #e5e5e5 !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stNumberInput label {
    color: #a0a0a0 !important; font-size: 0.78rem;
    text-transform: uppercase; letter-spacing: 1px;
  }
  [data-testid="metric-container"] {
    background: #f7f7f5; border: 1px solid #e8e8e8;
    border-radius: 12px; padding: 1rem 1.2rem;
  }
  [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important; font-weight: 800 !important; color: #111 !important;
  }
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #f0f0ee; border-radius: 10px; padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px; font-family: 'Syne', sans-serif;
    font-weight: 600; font-size: 0.85rem; color: #666;
  }
  .stTabs [aria-selected="true"] { background: #111 !important; color: #fff !important; }

  .pred-box {
    background: linear-gradient(135deg, #111 0%, #2d2d2d 100%);
    color: #fff; border-radius: 16px; padding: 2rem;
    text-align: center; margin-top: 1rem;
  }
  .pred-box .label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 2px; color: #aaa; margin-bottom: 0.5rem; }
  .pred-box .value { font-family: 'Syne', sans-serif; font-size: 3.5rem; font-weight: 800; color: #f5c518; }
  .pred-box .unit  { font-size: 1rem; color: #888; margin-top: 0.25rem; }

  .section-title {
    font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700;
    color: #111; border-left: 4px solid #f5c518;
    padding-left: 10px; margin-bottom: 1rem;
  }
  .train-box {
    background: #0d0d0d; border-radius: 16px; padding: 2.5rem;
    text-align: center; color: #fff; margin: 2rem auto; max-width: 600px;
  }
  .train-box h2 { font-family: 'Syne', sans-serif; font-size: 1.8rem; color: #f5c518; margin-bottom: 0.5rem; }
  .train-box p  { color: #aaa; font-size: 0.95rem; }

  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
CSV_PATH = "uber_nyc_enriched.csv"

FEATURES = [
    "cluster_id", "Hour", "DayOfWeek", "Month",
    "temp", "spd", "vsb", "pcp01",
    "hday", "IsWeekend", "IsRushHour",
]
MODEL_FILES = {
    "Linear Regression": "model_linear_regression.pkl",
    "Random Forest":     "model_random_forest.pkl",
    "XGBoost":           "model_xgboost.pkl",
}
MODEL_COLORS = {
    "Linear Regression": "#4e79a7",
    "Random Forest":     "#59a14f",
    "XGBoost":           "#f5c518",
}


# ─── Training pipeline ────────────────────────────────────────────────────────

def run_training_pipeline():
    st.markdown("""
    <div class="train-box">
      <h2>🚀 First-time Setup</h2>
      <p>No trained models found. Running the full pipeline now — this takes a few minutes.</p>
    </div>
    """, unsafe_allow_html=True)

    if not os.path.exists(CSV_PATH):
        st.error(f"❌ Dataset not found: `{CSV_PATH}`\n\nPlease upload `uber_nyc_enriched.csv` to your GitHub repo root.")
        st.stop()

    progress = st.progress(0, text="Initialising…")
    log_area = st.empty()
    logs     = []

    def log(msg, pct):
        logs.append(msg)
        log_area.code("\n".join(logs), language="bash")
        progress.progress(pct, text=msg)

    # 1. Load & preprocess
    log("⚙️  Loading local dataset…", 10)
    df = pd.read_csv(CSV_PATH)
    log(f"✓  Loaded {len(df):,} rows", 15)

    log("⚙️  Engineering features…", 20)
    df["pickup_dt"]  = pd.to_datetime(df["pickup_dt"])
    df["Hour"]       = df["pickup_dt"].dt.hour
    df["DayOfWeek"]  = df["pickup_dt"].dt.dayofweek
    df["Month"]      = df["pickup_dt"].dt.month
    df["IsWeekend"]  = (df["DayOfWeek"] >= 5).astype(int)
    df["IsRushHour"] = (
        (df["IsWeekend"] == 0) & (df["Hour"].isin([7, 8, 9, 17, 18, 19]))
    ).astype(int)
    if "hday" in df.columns:
        df["hday"] = df["hday"].map({"Y": 1, "N": 0, 1: 1, 0: 0}).fillna(0).astype(int)

    borough_map      = {b: i for i, b in enumerate(df["borough"].dropna().unique())}
    df["cluster_id"] = df["borough"].map(borough_map)
    with open("borough_map.json", "w") as f:
        json.dump(borough_map, f)

    grp         = ["cluster_id", "Hour", "DayOfWeek", "Month"]
    weather_agg = df.groupby(grp)[["temp", "spd", "vsb", "pcp01"]].mean().reset_index()
    flag_agg    = df.groupby(grp)[["hday", "IsWeekend", "IsRushHour"]].max().reset_index()
    pickup_agg  = df.groupby(grp)["pickups"].sum().reset_index()
    df_agg      = pickup_agg.merge(weather_agg, on=grp).merge(flag_agg, on=grp)
    log(f"✓  Aggregated shape: {df_agg.shape}", 35)

    # 2. Vectorise
    log("🔢  Vectorising features…", 40)
    df_clean = df_agg[FEATURES + ["pickups"]].dropna()
    y        = df_clean["pickups"].values
    records  = df_clean[FEATURES].to_dict(orient="records")
    vec      = DictVectorizer(sparse=False)
    X        = vec.fit_transform(records)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log(f"✓  Train: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows", 45)

    # 3. Train
    def rmse(yt, yp): return float(np.sqrt(mean_squared_error(yt, yp)))

    specs = {
        "Linear Regression": (LinearRegression(), 60),
        "Random Forest":     (RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), 75),
        "XGBoost":           (XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42), 88),
    }

    trained   = {}
    metrics_d = {}
    for name, (model, pct) in specs.items():
        log(f"🏋️  Training {name}…", pct - 5)
        model.fit(X_train, y_train)
        te  = round(rmse(y_test,  model.predict(X_test)),  4)
        tr  = round(rmse(y_train, model.predict(X_train)), 4)
        r2  = round(model.score(X_test,  y_test),  4)
        r2t = round(model.score(X_train, y_train), 4)
        trained[name]   = model
        metrics_d[name] = {"train_rmse": tr, "test_rmse": te, "train_r2": r2t, "test_r2": r2}
        log(f"   ✓  {name}: RMSE={te}  R²={r2}", pct)

    # 4. Save
    log("💾  Saving artifacts…", 92)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vec, f)
    for name, model in trained.items():
        with open(f"model_{name.lower().replace(' ','_')}.pkl", "wb") as f:
            pickle.dump(model, f)
    with open("metrics.json", "w") as f:
        json.dump(metrics_d, f, indent=2)

    progress.progress(100, text="✅  Training complete!")
    log("✅  All artifacts saved. Launching dashboard…", 100)

    import time; time.sleep(1.5)
    st.rerun()


# ─── Load artifacts ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_artifacts():
    for fname in list(MODEL_FILES.values()) + ["vectorizer.pkl"]:
        if not os.path.exists(fname):
            return None, None, None, None
    vec        = pickle.load(open("vectorizer.pkl", "rb"))
    models_out = {n: pickle.load(open(p, "rb")) for n, p in MODEL_FILES.items()}
    metrics    = json.load(open("metrics.json"))     if os.path.exists("metrics.json")     else {}
    bmap       = json.load(open("borough_map.json")) if os.path.exists("borough_map.json") else {}
    return models_out, vec, metrics, bmap


# ─── Auto-train if needed ─────────────────────────────────────────────────────
models, vec, metrics, borough_map = load_artifacts()
if models is None:
    run_training_pipeline()
    st.stop()


# ─── Header ───────────────────────────────────────────────────────────────────
c1, c2 = st.columns([1, 8])
with c1:
    st.markdown("<h1 style='font-size:2.8rem;margin:0'>🚖</h1>", unsafe_allow_html=True)
with c2:
    st.markdown("""
    <h1 style='margin:0;font-size:2rem;font-weight:800'>Uber Demand Forecasting</h1>
    <p style='margin:0;color:#888;font-size:0.9rem'>2305702 · Harshit Belwal · CSE 47</p>
    """, unsafe_allow_html=True)
st.markdown("---")


# ─── Helpers ──────────────────────────────────────────────────────────────────
def predict_demand(model, vec, d):
    return float(max(0, model.predict(vec.transform([d]))[0]))


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Prediction Inputs")
    st.markdown("---")

    selected_model_name = st.selectbox("Model", list(MODEL_FILES.keys()))
    borough_options     = list(borough_map.keys()) if borough_map else ["Manhattan"]
    borough             = st.selectbox("Borough", borough_options)
    cluster_id          = borough_map.get(borough, 0)

    hour        = st.slider("Hour of Day", 0, 23, 8)
    day_of_week = st.selectbox("Day of Week", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    dow         = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}[day_of_week]
    month       = st.selectbox("Month", list(range(1, 13)), index=0)

    st.markdown("#### 🌤 Weather")
    temp = st.number_input("Temperature (°F)", value=60.0, min_value=-10.0, max_value=110.0, step=0.5)
    spd  = st.number_input("Wind Speed (mph)", value=10.0, min_value=0.0,  max_value=60.0,  step=0.5)
    vsb  = st.number_input("Visibility (miles)", value=10.0, min_value=0.0, max_value=10.0, step=0.1)
    pcp  = st.number_input("Precipitation (in/hr)", value=0.0, min_value=0.0, max_value=2.0, step=0.01)
    hday = st.checkbox("Public Holiday?")

    is_weekend = 1 if dow >= 5 else 0
    is_rush    = 1 if (is_weekend == 0 and hour in [7,8,9,17,18,19]) else 0

    input_dict = {
        "cluster_id": cluster_id, "Hour": hour, "DayOfWeek": dow, "Month": month,
        "temp": temp, "spd": spd, "vsb": vsb, "pcp01": pcp,
        "hday": int(hday), "IsWeekend": is_weekend, "IsRushHour": is_rush,
    }

    st.markdown("---")
    if st.button("🔄 Retrain Models", use_container_width=True):
        for f in list(MODEL_FILES.values()) + ["vectorizer.pkl","metrics.json","borough_map.json"]:
            if os.path.exists(f): os.remove(f)
        st.cache_resource.clear()
        st.rerun()


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict Demand", "📊 Model Comparison", "🔍 Feature Insights"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_pred, col_all = st.columns([1, 1], gap="large")

    with col_pred:
        st.markdown('<div class="section-title">Single Model Prediction</div>', unsafe_allow_html=True)
        prediction = predict_demand(models[selected_model_name], vec, input_dict)
        st.markdown(f"""
        <div class="pred-box">
          <div class="label">{selected_model_name}</div>
          <div class="value">{prediction:,.0f}</div>
          <div class="unit">estimated pickups</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        flags = []
        if is_rush:    flags.append("🏃 Rush Hour")
        if is_weekend: flags.append("🏖️ Weekend")
        if hday:       flags.append("🎉 Holiday")
        if pcp > 0.1:  flags.append("🌧️ Rain")
        if flags:
            st.info("  ·  ".join(flags))

    with col_all:
        st.markdown('<div class="section-title">All Models Comparison</div>', unsafe_allow_html=True)
        all_preds = {n: predict_demand(m, vec, input_dict) for n, m in models.items()}
        fig = go.Figure()
        for name, val in all_preds.items():
            fig.add_trace(go.Bar(
                x=[name], y=[val], name=name,
                marker_color=MODEL_COLORS[name],
                text=[f"{val:,.0f}"], textposition="outside",
            ))
        fig.update_layout(
            showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
            yaxis_title="Predicted Pickups",
            margin=dict(t=20, b=20, l=20, r=20),
            font=dict(family="DM Sans"), height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Predicted Demand Across the Day</div>', unsafe_allow_html=True)
    hours      = list(range(24))
    hour_preds = {name: [] for name in models}
    for h in hours:
        d = {**input_dict, "Hour": h,
             "IsRushHour": 1 if (is_weekend == 0 and h in [7,8,9,17,18,19]) else 0}
        for name, m in models.items():
            hour_preds[name].append(predict_demand(m, vec, d))

    fig2 = go.Figure()
    for name, vals in hour_preds.items():
        fig2.add_trace(go.Scatter(
            x=hours, y=vals, mode="lines+markers", name=name,
            line=dict(color=MODEL_COLORS[name], width=2.5), marker=dict(size=5),
        ))
    fig2.add_vline(x=hour, line_dash="dash", line_color="#f5c518", annotation_text="Selected hour")
    fig2.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Hour of Day", yaxis_title="Predicted Pickups",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, b=20, l=20, r=20),
        font=dict(family="DM Sans"), height=340,
    )
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not metrics:
        st.warning("No metrics found.")
    else:
        st.markdown('<div class="section-title">Test-set Performance</div>', unsafe_allow_html=True)
        cols = st.columns(len(metrics))
        for i, (name, m) in enumerate(metrics.items()):
            with cols[i]:
                st.metric(label=name, value=f"RMSE {m['test_rmse']:,.1f}")
                st.caption(f"R² = {m['test_r2']:.4f}  |  Train RMSE = {m['train_rmse']:,.1f}")

        st.markdown("---")
        col_a, col_b = st.columns(2, gap="large")
        names = list(metrics.keys())

        with col_a:
            st.markdown('<div class="section-title">RMSE — Train vs Test</div>', unsafe_allow_html=True)
            fig3 = go.Figure()
            for split, color in [("train_rmse", "#c9d4dc"), ("test_rmse", None)]:
                fig3.add_trace(go.Bar(
                    name=split.replace("_", " ").title(), x=names,
                    y=[metrics[n][split] for n in names],
                    marker_color=color if color else [MODEL_COLORS[n] for n in names],
                    text=[f"{metrics[n][split]:.1f}" for n in names], textposition="outside",
                ))
            fig3.update_layout(
                barmode="group", plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                margin=dict(t=30, b=20, l=10, r=10), font=dict(family="DM Sans"), height=320,
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-title">R² Score — Train vs Test</div>', unsafe_allow_html=True)
            fig4 = go.Figure()
            for split, color in [("train_r2", "#c9d4dc"), ("test_r2", None)]:
                fig4.add_trace(go.Bar(
                    name=split.replace("_", " ").title(), x=names,
                    y=[metrics[n][split] for n in names],
                    marker_color=color if color else [MODEL_COLORS[n] for n in names],
                    text=[f"{metrics[n][split]:.4f}" for n in names], textposition="outside",
                ))
            fig4.update_layout(
                barmode="group", plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(range=[0, 1.15]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                margin=dict(t=30, b=20, l=10, r=10), font=dict(family="DM Sans"), height=320,
            )
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown('<div class="section-title">Summary Table</div>', unsafe_allow_html=True)
        df_m = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})
        st.dataframe(
            df_m.style
                .highlight_min(subset=["test_rmse"], color="#d4edda")
                .highlight_max(subset=["test_r2"],   color="#d4edda")
                .format({"train_rmse": "{:.2f}", "test_rmse": "{:.2f}",
                         "train_r2": "{:.4f}",  "test_r2": "{:.4f}"}),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FEATURE INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Demand Sensitivity Analysis</div>', unsafe_allow_html=True)
    st.caption("How predicted pickups change as a single feature varies — all else held constant")

    c1, c2 = st.columns(2)
    with c1:
        sweep_feature = st.selectbox("Sweep feature", ["Hour","temp","spd","vsb","pcp01"])
    with c2:
        sweep_model = st.selectbox("Model", list(models.keys()), key="sweep_model")

    ranges     = {"Hour":(0,23,24),"temp":(10,100,30),"spd":(0,40,30),"vsb":(0,10,30),"pcp01":(0,1.5,30)}
    lo, hi, n  = ranges[sweep_feature]
    sweep_vals = np.linspace(lo, hi, n)
    preds = []
    for v in sweep_vals:
        d = {**input_dict, sweep_feature: float(v)}
        if sweep_feature == "Hour":
            d["IsRushHour"] = 1 if (is_weekend == 0 and int(v) in [7,8,9,17,18,19]) else 0
        preds.append(predict_demand(models[sweep_model], vec, d))

    fig5 = px.line(x=sweep_vals, y=preds,
                   labels={"x": sweep_feature, "y": "Predicted Pickups"},
                   color_discrete_sequence=[MODEL_COLORS[sweep_model]])
    fig5.add_scatter(
        x=[input_dict.get(sweep_feature, sweep_vals[0])],
        y=[predict_demand(models[sweep_model], vec, input_dict)],
        mode="markers", marker=dict(size=12, color="#f5c518", symbol="star"),
        name="Current selection",
    )
    fig5.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                       font=dict(family="DM Sans"), height=350,
                       margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<div class="section-title">Predicted Demand: Hour × Borough</div>', unsafe_allow_html=True)
    heat_model    = st.selectbox("Model", list(models.keys()), key="heat_model")
    boroughs_list = list(borough_map.keys()) if borough_map else ["Manhattan"]
    heat_data     = np.zeros((24, len(boroughs_list)))
    for bi, b in enumerate(boroughs_list):
        for ho in range(24):
            d = {**input_dict, "Hour": ho, "cluster_id": borough_map[b],
                 "IsRushHour": 1 if (is_weekend == 0 and ho in [7,8,9,17,18,19]) else 0}
            heat_data[ho, bi] = predict_demand(models[heat_model], vec, d)

    fig6 = px.imshow(heat_data, x=boroughs_list, y=list(range(24)),
                     color_continuous_scale="YlOrBr",
                     labels=dict(x="Borough", y="Hour of Day", color="Pickups"), aspect="auto")
    fig6.update_layout(font=dict(family="DM Sans"), height=420,
                       margin=dict(t=20, b=20, l=20, r=20), paper_bgcolor="white")
    st.plotly_chart(fig6, use_container_width=True)
