# ==============================================================================
# Uber Demand Forecasting - Retro Terminal Edition (Clean)
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

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Uber Demand Forecasting",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');

  /* 1. Global Font and Dark CRT Backgrounds */
  html, body, [class*="css"] {
    font-family: 'VT323', monospace !important;
    font-size: 1.15rem; 
  }
  
  [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #000000 !important; 
  }
  
  [data-testid="stSidebar"] {
    background-color: #0A0A0A !important; 
    border-right: 4px solid #39FF14 !important; 
  }

  /* 2. Retro Text Coloring */
  h1, h2, h3, h4, h5, h6, p, label, li, .stMarkdown {
    color: #39FF14 !important; 
    letter-spacing: 1px;
  }
  h1, h2, h3, .section-title {
    text-transform: uppercase;
  }

  /* 3. Aggressive Fix for Input Boxes & Dropdowns */
  input, [data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="input"] > div {
    background-color: #000000 !important;
    color: #39FF14 !important;
    border: 2px solid #39FF14 !important;
    border-radius: 0px !important; 
  }
  
  input::placeholder, [data-baseweb="select"] span {
    color: #008800 !important; 
  }

  div[data-baseweb="popover"], 
  div[data-baseweb="popover"] > div, 
  div[data-baseweb="popover"] ul {
    background-color: #000000 !important;
  }
  
  ul[data-baseweb="menu"] {
    background-color: #000000 !important;
    border: 2px solid #39FF14 !important;
    border-radius: 0px !important;
  }
  
  ul[data-baseweb="menu"] li {
    background-color: #000000 !important;
    color: #39FF14 !important;
    font-size: 1.2rem;
  }
  
  ul[data-baseweb="menu"] li span {
    color: #39FF14 !important;
  }
  
  ul[data-baseweb="menu"] li:hover {
    background-color: #39FF14 !important; 
    color: #000000 !important;
  }
  ul[data-baseweb="menu"] li:hover span {
    color: #000000 !important;
  }

  [data-testid="stNumberInputStepUp"], 
  [data-testid="stNumberInputStepDown"] {
      background-color: #000000 !important;
      color: #39FF14 !important;
      border-left: 2px solid #39FF14 !important;
  }
  [data-testid="stNumberInputStepUp"] svg, 
  [data-testid="stNumberInputStepDown"] svg {
      fill: #39FF14 !important;
  }

  /* 4. Fix Sliders & Toggles */
  [data-baseweb="slider"] div[data-testid="stTickBar"] > div,
  [data-baseweb="slider"] div[role="slider"] {
    background-color: #FF00FF !important; 
  }
  
  [data-testid="stCheckbox"] label span:first-child {
    border: 2px solid #39FF14 !important;
    background-color: #000000 !important;
    border-radius: 0px !important;
  }

  /* 5. Custom UI Elements - 8-Bit Style */
  [data-testid="metric-container"] {
    background-color: #000000 !important;
    border: 2px solid #FF00FF !important; 
    border-radius: 0px;
    padding: 1rem 1.2rem;
    box-shadow: 4px 4px 0px #FF00FF; 
  }
  [data-testid="stMetricValue"] {
      font-size: 2.5rem !important;
      color: #00FFFF !important; 
  }
  
  .stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
    border-bottom: 2px solid #39FF14 !important;
    padding: 0;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 0px;
    font-weight: normal;
    font-size: 1.2rem;
    padding: 10px 16px;
    color: #008800 !important;
    border-bottom: 4px solid transparent;
    text-transform: uppercase;
  }
  .stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #00FFFF !important; 
    border-bottom: 4px solid #00FFFF !important;
  }

  /* 8-Bit Prediction Box */
  .pred-box {
    background: #000000 !important;
    border: 4px solid #00FFFF !important;
    border-radius: 0px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-top: 1rem;
    box-shadow: 8px 8px 0px #00FFFF;
  }
  .pred-box * {
    color: #00FFFF !important;
  }
  .pred-box .label {
    font-size: 1.2rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.5rem;
  }
  .pred-box .value {
    font-size: 4.5rem;
    line-height: 1.1;
  }
  .pred-box .unit {
    font-size: 1.2rem;
    margin-top: 0.5rem;
  }

  .section-title {
    font-size: 1.5rem;
    color: #39FF14 !important;
    border-bottom: 2px dashed #39FF14; 
    padding-bottom: 5px;
    margin-bottom: 1rem;
    margin-top: 1.5rem;
  }

  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
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

# Arcade Palette for Charts
MODEL_COLORS = {
    "Linear Regression": "#00FFFF", # Cyan
    "Random Forest":     "#FF00FF", # Magenta
    "XGBoost":           "#FFFF00", # Yellow
}


@st.cache_resource(show_spinner=False)
def load_artifacts():
    missing = []
    for fname in list(MODEL_FILES.values()) + ["vectorizer.pkl"]:
        if not os.path.exists(fname):
            missing.append(fname)
    if missing:
        return None, None, None, None

    vec = pickle.load(open("vectorizer.pkl", "rb"))
    models = {name: pickle.load(open(path, "rb")) for name, path in MODEL_FILES.items()}

    metrics = {}
    if os.path.exists("metrics.json"):
        metrics = json.load(open("metrics.json"))

    borough_map = {}
    if os.path.exists("borough_map.json"):
        borough_map = json.load(open("borough_map.json"))

    return models, vec, metrics, borough_map


def predict_demand(model, vec: DictVectorizer, input_dict: dict) -> float:
    X = vec.transform([input_dict])
    return float(max(0, model.predict(X)[0]))


# ─── Load ─────────────────────────────────────────────────────────────────────
models, vec, metrics, borough_map = load_artifacts()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='margin:0;font-size:2.8rem;color:#39FF14 !important;'>Uber Demand Forcasting </h1>
<p style='margin:0;font-size:1.2rem;color:#00FFFF !important;margin-bottom:1.5rem;'>HARSHIT BELWAL |2305702 |CSE_47</p>
""", unsafe_allow_html=True)

if models is None:
    st.error("[!] SYSTEM ERROR: MODEL ARTIFACTS NOT FOUND. EXECUTE 'python train.py' TO COMPILE.")
    st.code("python train.py", language="bash")
    st.stop()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='section-title' style='margin-top:0;'>&gt; INPUT_PARAMS</div>", unsafe_allow_html=True)

    selected_model_name = st.selectbox("[ ALGO ] SELECT_ENGINE", list(MODEL_FILES.keys()))

    borough_options = list(borough_map.keys()) if borough_map else ["Manhattan", "Brooklyn", "Queens", "Bronx", "EWR"]
    borough = st.selectbox("[ LOC ] ZONE_SELECT", borough_options)
    cluster_id = borough_map.get(borough, 0)

    hour       = st.slider("[ TIME ] HOUR_INDEX", 0, 23, 8)
    day_of_week = st.selectbox("[ DATE ] DAY_VECTOR", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    dow_map    = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
    dow        = dow_map[day_of_week]
    month      = st.selectbox("[ DATE ] MONTH_CYCLE", list(range(1, 13)), index=0)

    st.markdown("<div class='section-title'>&gt; WEATHER_DATA</div>", unsafe_allow_html=True)
    temp = st.number_input("TEMP_F", value=60.0, min_value=-10.0, max_value=110.0, step=0.5)
    spd  = st.number_input("WIND_SPD", value=10.0, min_value=0.0, max_value=60.0, step=0.5)
    vsb  = st.number_input("VISIBILITY", value=10.0, min_value=0.0, max_value=10.0, step=0.1)
    pcp  = st.number_input("PRECIP_LVL", value=0.0, min_value=0.0, max_value=2.0, step=0.01)
    hday = st.checkbox("[ FLAG ] PUBLIC_HOLIDAY")

    is_weekend  = 1 if dow >= 5 else 0
    is_rush     = 1 if (is_weekend == 0 and hour in [7,8,9,17,18,19]) else 0

    input_dict = {
        "cluster_id": cluster_id, "Hour": hour, "DayOfWeek": dow, "Month": month,
        "temp": temp, "spd": spd, "vsb": vsb, "pcp01": pcp,
        "hday": int(hday), "IsWeekend": is_weekend, "IsRushHour": is_rush,
    }

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["[ EXEC_PREDICT ]", "[ BENCHMARKS ]", "[ DEBUG_FEATURES ]"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_pred, col_all = st.columns([1, 1], gap="large")

    with col_pred:
        st.markdown('<div class="section-title">&gt; PRIMARY_OUTPUT</div>', unsafe_allow_html=True)
        prediction = predict_demand(models[selected_model_name], vec, input_dict)
        st.markdown(f"""
        <div class="pred-box">
          <div class="label">:: {selected_model_name}_RESULT ::</div>
          <div class="value">{prediction:,.0f}</div>
          <div class="unit">UNITS: EST_PICKUPS</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        flags = []
        if is_rush:   flags.append("[!] RUSH_HR")
        if is_weekend: flags.append("[*] WEEKEND")
        if hday:      flags.append("[*] HOLIDAY")
        if pcp > 0.1: flags.append("[!] RAIN_WARN")
        if flags:
            st.info(f"**ENV_FLAGS:** {' | '.join(flags)}")

    with col_all:
        st.markdown('<div class="section-title">&gt; ALGO_COMPARISON</div>', unsafe_allow_html=True)
        all_preds = {name: predict_demand(m, vec, input_dict) for name, m in models.items()}
        fig = go.Figure()
        for name, val in all_preds.items():
            fig.add_trace(go.Bar(
                x=[name], y=[val], name=name, marker_color=MODEL_COLORS[name],
                text=[f"{val:,.0f}"], textposition="outside",
            ))
        fig.update_layout(
            showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis_title="EST_PICKUPS", margin=dict(t=20, b=20, l=20, r=20),
            font=dict(family="VT323", color="#39FF14", size=16), height=320,
        )
        fig.update_traces(marker_line_color="#000", marker_line_width=2)
        fig.update_xaxes(showgrid=False, linecolor="#39FF14")
        fig.update_yaxes(showgrid=True, gridcolor="#004400", linecolor="#39FF14")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">&gt; 24HR_PROJECTION_MATRIX</div>', unsafe_allow_html=True)
    hours = list(range(24))
    hour_preds = {name: [] for name in models}
    for h in hours:
        d = {**input_dict, "Hour": h, "IsRushHour": 1 if (is_weekend == 0 and h in [7,8,9,17,18,19]) else 0}
        for name, m in models.items():
            hour_preds[name].append(predict_demand(m, vec, d))

    fig2 = go.Figure()
    for name, vals in hour_preds.items():
        fig2.add_trace(go.Scatter(
            x=hours, y=vals, mode="lines+markers", name=name, 
            line=dict(color=MODEL_COLORS[name], width=3, shape='vh'), marker=dict(size=8, symbol='square'),
        ))
    fig2.add_vline(x=hour, line_dash="dash", line_color="#FF00FF", annotation_text="< CURR_T", annotation_position="top left", annotation_font_color="#FF00FF")
    fig2.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="TIME_INDEX", yaxis_title="EST_PICKUPS",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, b=20, l=20, r=20), font=dict(family="VT323", color="#39FF14", size=16), height=380,
    )
    fig2.update_xaxes(showgrid=False, linecolor="#39FF14")
    fig2.update_yaxes(showgrid=True, gridcolor="#004400", linecolor="#39FF14")
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not metrics:
        st.warning("[!] ERR: 'metrics.json' NOT FOUND")
    else:
        st.markdown('<div class="section-title">&gt; TEST_SET_METRICS</div>', unsafe_allow_html=True)
        cols = st.columns(len(metrics))
        for i, (name, m) in enumerate(metrics.items()):
            with cols[i]:
                st.metric(label=f":: {name}", value=f"RMSE {m['test_rmse']:,.1f}")
                st.caption(f"**R²** = {m['test_r2']:.4f}  |  **TRN_RMSE** = {m['train_rmse']:,.1f}")

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            st.markdown('<div class="section-title">&gt; RMSE: TRN vs TST</div>', unsafe_allow_html=True)
            names = list(metrics.keys())
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                name="TRN_RMSE", x=names, y=[metrics[n]["train_rmse"] for n in names],
                marker_color="#008800", text=[f"{metrics[n]['train_rmse']:.1f}" for n in names], textposition="outside",
            ))
            fig3.add_trace(go.Bar(
                name="TST_RMSE", x=names, y=[metrics[n]["test_rmse"] for n in names],
                marker_color=[MODEL_COLORS[n] for n in names], text=[f"{metrics[n]['test_rmse']:.1f}" for n in names], textposition="outside",
            ))
            fig3.update_layout(
                barmode="group", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                margin=dict(t=30, b=20, l=10, r=10), font=dict(family="VT323", color="#39FF14", size=16), height=320,
            )
            fig3.update_traces(marker_line_color="#000", marker_line_width=2)
            fig3.update_xaxes(showgrid=False, linecolor="#39FF14")
            fig3.update_yaxes(showgrid=True, gridcolor="#004400", linecolor="#39FF14")
            st.plotly_chart(fig3, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-title">&gt; R_SQUARED_SCORES</div>', unsafe_allow_html=True)
            fig4 = go.Figure()
            for split in ["train", "test"]:
                fig4.add_trace(go.Bar(
                    name=f"{split[:3].upper()}_R2", x=names, y=[metrics[n][f"{split}_r2"] for n in names],
                    marker_color="#008800" if split == "train" else [MODEL_COLORS[n] for n in names],
                    text=[f"{metrics[n][f'{split}_r2']:.4f}" for n in names], textposition="outside",
                ))
            fig4.update_layout(
                barmode="group", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[0, 1.1]), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                margin=dict(t=30, b=20, l=10, r=10), font=dict(family="VT323", color="#39FF14", size=16), height=320,
            )
            fig4.update_traces(marker_line_color="#000", marker_line_width=2)
            fig4.update_xaxes(showgrid=False, linecolor="#39FF14")
            fig4.update_yaxes(showgrid=True, gridcolor="#004400", linecolor="#39FF14")
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown('<div class="section-title">&gt; MEMORY_DUMP (TABLE)</div>', unsafe_allow_html=True)
        df_metrics = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "MODEL_ID"})
        st.dataframe(
            df_metrics.style.format({"train_rmse": "{:.2f}", "test_rmse": "{:.2f}", "train_r2": "{:.4f}", "test_r2": "{:.4f}"}),
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FEATURE INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">&gt; SENSITIVITY_ANALYSIS</div>', unsafe_allow_html=True)
    st.caption("> SIMULATING SINGLE VARIABLE SWEEP WHILE HOLDING OTHERS CONSTANT")

    sweep_feature = st.selectbox("[ VAR ] TARGET_VAR", ["Hour", "temp", "spd", "vsb", "pcp01"])
    sweep_model   = st.selectbox("[ ALGO ] ENGINE", list(models.keys()), key="sweep_model")

    ranges = {"Hour": (0, 23, 24), "temp": (10, 100, 30), "spd": (0, 40, 30), "vsb": (0, 10, 30), "pcp01": (0, 1.5, 30)}
    lo, hi, n = ranges[sweep_feature]
    sweep_vals = np.linspace(lo, hi, n)

    preds = []
    for v in sweep_vals:
        d = {**input_dict, sweep_feature: float(v)}
        if sweep_feature == "Hour":
            d["IsRushHour"] = 1 if (is_weekend == 0 and int(v) in [7,8,9,17,18,19]) else 0
        preds.append(predict_demand(models[sweep_model], vec, d))

    fig5 = px.line(x=sweep_vals, y=preds, labels={"x": sweep_feature, "y": "EST_PICKUPS"}, color_discrete_sequence=[MODEL_COLORS[sweep_model]])
    fig5.add_scatter(
        x=[input_dict.get(sweep_feature, sweep_vals[0])], y=[predict_demand(models[sweep_model], vec, input_dict)],
        mode="markers", marker=dict(size=14, color="#FF00FF", symbol="cross"), name="CURR_STATE",
    )
    fig5.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="VT323", color="#39FF14", size=16), height=350, margin=dict(t=20, b=20, l=20, r=20),
    )
    fig5.update_traces(line=dict(width=4, shape='vh')) 
    fig5.update_xaxes(showgrid=False, linecolor="#39FF14")
    fig5.update_yaxes(showgrid=True, gridcolor="#004400", linecolor="#39FF14")
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<div class="section-title">&gt; ZONE_MATRIX: HR x BOROUGH</div>', unsafe_allow_html=True)
    heat_model = st.selectbox("[ ALGO ] ENGINE_HEATMAP", list(models.keys()), key="heat_model")
    boroughs_list = list(borough_map.keys()) if borough_map else ["Manhattan"]
    heat_data = np.zeros((24, len(boroughs_list)))
    for hi_, b in enumerate(boroughs_list):
        for ho_ in range(24):
            d = {**input_dict, "Hour": ho_, "cluster_id": borough_map[b], "IsRushHour": 1 if (is_weekend == 0 and ho_ in [7,8,9,17,18,19]) else 0}
            heat_data[ho_, hi_] = predict_demand(models[heat_model], vec, d)

    fig6 = px.imshow(
        heat_data, x=boroughs_list, y=list(range(24)), color_continuous_scale="Turbo",
        labels=dict(x="ZONE", y="TIME_INDEX", color="PICKUPS"), aspect="auto",
    )
    fig6.update_layout(
        font=dict(family="VT323", color="#39FF14", size=16), height=450, margin=dict(t=20, b=20, l=20, r=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig6, use_container_width=True)