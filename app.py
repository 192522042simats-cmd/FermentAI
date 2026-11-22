from pathlib import Path
from datetime import datetime
import itertools
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# FERMENTAI | Random Forest fermentation analysis dashboard
# Model feature names were inspected from the supplied .pkl:
# Initial_pH, Temperature_C, Agitation_rpm, Incubation_Time_h
# ============================================================

st.set_page_config(
    page_title="FermentAI | α-Amylase Optimization",
    page_icon="🧫",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "random_forest_model.pkl"
DATA_FILE = BASE_DIR / "BBD_29_Runs_For_Verification_FIXED.csv"

FEATURES = [
    "Initial_pH",
    "Temperature_C",
    "Agitation_rpm",
    "Incubation_Time_h",
]
LABELS = {
    "Initial_pH": "Initial pH",
    "Temperature_C": "Temperature",
    "Agitation_rpm": "Agitation speed",
    "Incubation_Time_h": "Incubation time",
}
UNITS = {
    "Initial_pH": "pH",
    "Temperature_C": "°C",
    "Agitation_rpm": "rpm",
    "Incubation_Time_h": "hours",
}
RANGES = {
    "Initial_pH": (4.0, 8.0, 6.0, 0.1),
    "Temperature_C": (25.0, 45.0, 35.0, 1.0),
    "Agitation_rpm": (120.0, 140.0, 130.0, 1.0),
    "Incubation_Time_h": (12.0, 48.0, 24.0, 1.0),
}
REFERENCE_ACTIVITY = 145.32
REFERENCE_CONDITION = {
    "Initial_pH": 6.0,
    "Temperature_C": 35.0,
    "Agitation_rpm": 130.0,
    "Incubation_Time_h": 24.0,
}

# --------------------------- Theme ---------------------------
st.markdown("""
<style>
:root { --navy:#12263F; --teal:#087F8C; --cyan:#DDF3F3; --ink:#26384A; }
.stApp { background:#F4F7FB; color:var(--ink); }
[data-testid="stSidebar"] { background:#10243A; }
[data-testid="stSidebar"] * { color:#F3F7FB !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:5px 2px; }
.main .block-container { max-width:1500px; padding-top:1.25rem; padding-bottom:3rem; }
h1,h2,h3 { color:var(--navy); letter-spacing:-.02em; }
.hero {
  background:linear-gradient(115deg,#10243A 0%,#126B78 65%,#1A9A92 100%);
  padding:26px 30px; border-radius:18px; color:white; margin-bottom:18px;
  box-shadow:0 8px 24px rgba(16,36,58,.12);
}
.hero h1 { color:white !important; margin:0 0 7px 0; font-size:2.1rem; }
.hero p { color:#E6F5F6 !important; margin:0; font-size:1rem; }
.kpi {
  background:white; border:1px solid #E4EAF1; border-radius:15px;
  padding:17px 18px; min-height:112px; box-shadow:0 3px 12px rgba(18,38,63,.04);
}
.kpi-label { color:#718096; font-size:.83rem; font-weight:600; }
.kpi-value { color:#142D48; font-size:1.75rem; font-weight:750; margin-top:5px; }
.kpi-sub { color:#7A8A9A; font-size:.76rem; margin-top:3px; }
.info {
  background:#E8F5F5; border-left:4px solid #07858A; padding:13px 16px;
  border-radius:9px; margin:8px 0 14px 0; color:#203B4A;
}
.warning {
  background:#FFF4E5; border-left:4px solid #D88B2A; padding:13px 16px;
  border-radius:9px; margin:8px 0 14px 0; color:#503B1C;
}
.section-card {
  background:white; border:1px solid #E4EAF1; border-radius:15px;
  padding:17px 19px; margin-bottom:12px;
}
div[data-testid="stMetric"] {
  background:white; border:1px solid #E4EAF1; border-radius:13px;
  padding:12px 15px;
}
.stButton button, .stDownloadButton button {
  border-radius:10px; font-weight:650; min-height:42px;
}
[data-testid="stTabs"] button { font-weight:650; }
</style>
""", unsafe_allow_html=True)


# --------------------------- Helpers ---------------------------
@st.cache_resource(show_spinner="Loading the Random Forest model...")
def load_model(path_string):
    # The supplied model was serialized with scikit-learn 1.6.1.
    # requirements.txt pins that version to avoid pickle version drift.
    return joblib.load(path_string)

@st.cache_data(show_spinner=False)
def load_data(path_string):
    return pd.read_csv(path_string)

def hero(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True
    )

def kpi(label, value, sub=""):
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True
    )

def note(message, warning=False):
    cls = "warning" if warning else "info"
    st.markdown(f'<div class="{cls}">{message}</div>', unsafe_allow_html=True)

def check_model_schema(model):
    found = list(getattr(model, "feature_names_in_", []))
    if found and found != FEATURES:
        raise ValueError(
            "The model feature columns do not match the app. "
            f"Expected {FEATURES}, but model reports {found}."
        )
    count = getattr(model, "n_features_in_", len(FEATURES))
    if int(count) != 4:
        raise ValueError(f"The model expects {count} inputs; this app supplies 4.")
    return True

def input_frame(values):
    # Always use the exact feature names and order stored in the model.
    return pd.DataFrame(
        [[float(values[f]) for f in FEATURES]],
        columns=FEATURES
    )

def predict_one(model, values):
    result = np.asarray(model.predict(input_frame(values))).reshape(-1)
    if result.size != 1 or not np.isfinite(float(result[0])):
        raise ValueError("The model returned an invalid prediction.")
    return float(result[0])

def predict_many(model, frame):
    x = frame.loc[:, FEATURES].astype(float)
    y = np.asarray(model.predict(x)).reshape(-1)
    if len(y) != len(x) or not np.isfinite(y).all():
        raise ValueError("The model returned invalid predictions.")
    return y

def download_csv(frame, filename, label):
    st.download_button(
        label=label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )

def parameter_control(feature, key, value=None):
    lo, hi, default, step = RANGES[feature]
    if value is None:
        value = default
    return st.slider(
        LABELS[feature],
        min_value=float(lo),
        max_value=float(hi),
        value=float(np.clip(float(value), lo, hi)),
        step=float(step),
        key=key,
        help=f"Study range: {lo:g}–{hi:g} {UNITS[feature]}"
    )

def make_reference_values():
    return {f: float(REFERENCE_CONDITION[f]) for f in FEATURES}

# --------------------------- Load model/data ---------------------------
model = None
model_status = None
data = None
data_status = None

try:
    if MODEL_FILE.exists():
        model = load_model(str(MODEL_FILE))
        check_model_schema(model)
        model_status = "Model loaded"
    else:
        model_status = f"Missing file: {MODEL_FILE.name}"
except Exception as exc:
    model = None
    model_status = f"Model loading error: {exc}"

try:
    if DATA_FILE.exists():
        data = load_data(str(DATA_FILE))
        required = FEATURES + ["Amylase_Activity_Experimental_U_mL"]
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")
        data_status = f"{len(data)} published BBD runs loaded"
    else:
        data_status = f"Missing file: {DATA_FILE.name}"
except Exception as exc:
    data = None
    data_status = f"Dataset loading error: {exc}"

# --------------------------- Sidebar ---------------------------
st.sidebar.markdown("# 🧫 FermentAI")
st.sidebar.caption("SMART FERMENTATION ANALYTICS")
st.sidebar.markdown("---")
pages = [
    "Overview",
    "Smart Optimizer",
    "Parameter Analysis",
    "Model Performance",
    "Dataset Explorer",
    "Optimization Lab",
    "Scientific Insights",
    "Report Center",
]
page = st.sidebar.radio("WORKSPACE", pages, index=0)
st.sidebar.markdown("---")
st.sidebar.markdown("**SYSTEM STATUS**")
st.sidebar.caption(f"Model: {model_status or 'Not checked'}")
st.sidebar.caption(f"Dataset: {data_status or 'Not checked'}")
st.sidebar.markdown("---")
st.sidebar.caption("B.Tech Biotechnology research prototype")
st.sidebar.caption("Submerged fermentation • α-Amylase")

def require_model():
    if model is None:
        st.error(
            "The model could not be loaded. Confirm that random_forest_model.pkl "
            "is in the same GitHub root folder as app.py and that scikit-learn "
            "is pinned to 1.6.1 in requirements.txt."
        )
        return False
    return True


# ============================================================
# 1. OVERVIEW
# ============================================================
if page == "Overview":
    hero("FermentAI", "Explore, predict and analyze microbial α-amylase production under submerged fermentation.")

    st.markdown(
        "A student-friendly machine-learning dashboard that connects fermentation "
        "process variables with model-estimated enzyme activity."
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("ML algorithm", "Random Forest", "Regression model")
    with c2: kpi("Process factors", "04", "pH • temperature • agitation • time")
    with c3: kpi("Reference design", "29 runs", "Box–Behnken design")
    with c4: kpi("Reference activity", "145.32", "U/mL • published experiment")

    st.subheader("Your workflow")
    flow = st.columns(4)
    cards = [
        ("01", "Set conditions", "Choose the four fermentation parameters."),
        ("02", "Predict response", "Estimate α-amylase activity with Random Forest."),
        ("03", "Explore patterns", "Study one-factor curves and two-factor response maps."),
        ("04", "Screen candidates", "Compare combinations and export your results."),
    ]
    for col, (num, title, body) in zip(flow, cards):
        with col:
            st.markdown(
                f'<div class="section-card"><div style="color:#07858A;font-weight:800;font-size:1.3rem">{num}</div>'
                f'<h3 style="margin:5px 0">{title}</h3><div>{body}</div></div>',
                unsafe_allow_html=True
            )

    st.subheader("Fermentation process at a glance")
    process = go.Figure()
    nodes = ["Medium preparation", "Parameter setting", "Bacillus cultivation", "Enzyme production", "Activity estimation"]
    x = [0, 1, 2, 3, 4]
    for i in range(4):
        process.add_annotation(
            x=i + 0.5, y=0.5, ax=i + 0.18, ay=0.5,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=3, arrowsize=1.2,
            arrowwidth=2, arrowcolor="#0B8990"
        )
    process.add_trace(go.Scatter(
        x=x, y=[0.5]*5, mode="markers+text",
        marker=dict(size=54, color=["#16324F","#1A6073","#16858A","#20A39E","#75C9BE"]),
        text=nodes, textposition="bottom center",
        textfont=dict(size=11, color="#20384A"),
        hoverinfo="text"
    ))
    process.update_layout(
        height=210, margin=dict(l=20,r=20,t=15,b=50),
        xaxis=dict(visible=False, range=[-0.4,4.4]),
        yaxis=dict(visible=False, range=[0,1]),
        showlegend=False, plot_bgcolor="white", paper_bgcolor="white"
    )
    st.plotly_chart(process, use_container_width=True)

    note(
        "<b>Scientific note:</b> The 29-run response table below is transcribed from "
        "Table 1 of Khusro et al. (2017), which reports experimental and RSM-predicted "
        "amylase activity. The Random Forest model file is a separate serialized model; "
        "its original training split/history is not embedded in the app. Do not treat "
        "dashboard estimates as new experimental measurements.",
        warning=True
    )
    st.markdown(
        "[Open the published reference paper](https://www.mdpi.com/2311-5637/3/1/7)"
    )


# ============================================================
# 2. SMART OPTIMIZER
# ============================================================
elif page == "Smart Optimizer":
    hero("Smart Optimizer", "Enter fermentation conditions and obtain an instant model estimate.")

    if require_model():
        left, right = st.columns([1, 1.05], gap="large")
        with left:
            st.markdown("### Fermentation conditions")
            values = {}
            for feature in FEATURES:
                values[feature] = parameter_control(feature, f"smart_{feature}")
            run_prediction = st.button("⚡ Predict enzyme activity", type="primary", use_container_width=True)

        with right:
            st.markdown("### Prediction dashboard")
            if run_prediction:
                try:
                    estimate = predict_one(model, values)
                    record = {
                        **values,
                        "Predicted_Activity_U_mL": estimate,
                        "Generated_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    st.session_state["last_prediction"] = record
                    st.metric("Estimated α-amylase activity", f"{estimate:.2f} U/mL")
                    delta = estimate - REFERENCE_ACTIVITY
                    st.metric("Difference from published maximum", f"{delta:+.2f} U/mL")
                    st.caption("The comparison is contextual only; it is not a validation score.")
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=estimate,
                        number={"suffix":" U/mL","font":{"size":27}},
                        gauge={
                            "axis":{"range":[0, max(160, estimate*1.15)]},
                            "bar":{"color":"#0B8990"},
                            "steps":[
                                {"range":[0,80],"color":"#EDF2F7"},
                                {"range":[80,120],"color":"#D8EEEE"},
                                {"range":[120,max(160,estimate*1.15)],"color":"#BDE5DF"},
                            ],
                        },
                        title={"text":"Model-estimated response"}
                    ))
                    gauge.update_layout(height=260, margin=dict(l=20,r=20,t=45,b=10))
                    st.plotly_chart(gauge, use_container_width=True)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
            elif "last_prediction" in st.session_state:
                saved = st.session_state["last_prediction"]
                st.metric("Last estimated activity", f"{saved['Predicted_Activity_U_mL']:.2f} U/mL")
                st.caption(f"Generated at {saved['Generated_At']}")
                st.info("Change any parameter and click Predict to update the result.")
            else:
                st.info("Set the four conditions and click **Predict enzyme activity**.")

        if "last_prediction" in st.session_state:
            saved = st.session_state["last_prediction"]
            st.markdown("### Prediction profile")
            profile = pd.DataFrame({
                "Parameter": [LABELS[f] for f in FEATURES],
                "Selected value": [saved[f] for f in FEATURES],
                "Unit": [UNITS[f] for f in FEATURES],
            })
            st.dataframe(profile, hide_index=True, use_container_width=True)
            download_csv(pd.DataFrame([saved]), "fermentai_prediction.csv", "Download prediction")


# ============================================================
# 3. PARAMETER ANALYSIS
# ============================================================
elif page == "Parameter Analysis":
    hero("Parameter Analysis", "See how each factor changes the model-estimated response.")

    if require_model():
        tab1, tab2 = st.tabs(["One-factor response", "Two-factor response map"])

        with tab1:
            chosen = st.selectbox(
                "Choose the factor to investigate",
                FEATURES,
                format_func=lambda f: f"{LABELS[f]} ({UNITS[f]})"
            )
            baseline = make_reference_values()
            st.markdown("#### Hold the other three factors constant")
            others = [f for f in FEATURES if f != chosen]
            cols = st.columns(3)
            for col, feature in zip(cols, others):
                with col:
                    baseline[feature] = parameter_control(
                        feature, f"analysis_base_{feature}", REFERENCE_CONDITION[feature]
                    )

            lo, hi, _, _ = RANGES[chosen]
            values = np.linspace(lo, hi, 51, dtype=float)
            rows_out = []
            for v in values:
                trial = baseline.copy()
                trial[chosen] = float(v)
                rows_out.append(predict_one(model, trial))
            curve = pd.DataFrame({
                LABELS[chosen]: values,
                "Predicted activity (U/mL)": rows_out
            })
            fig = px.line(
                curve, x=LABELS[chosen], y="Predicted activity (U/mL)",
                markers=True, template="plotly_white",
                title=f"Effect of {LABELS[chosen]} on predicted α-amylase activity"
            )
            fig.update_traces(line=dict(color="#07858A", width=3), marker=dict(size=5))
            fig.update_layout(height=420, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            peak = int(np.argmax(rows_out))
            m1, m2 = st.columns(2)
            m1.metric("Highest point in this sweep", f"{rows_out[peak]:.2f} U/mL")
            m2.metric(f"{LABELS[chosen]} at that point", f"{values[peak]:.2f} {UNITS[chosen]}")
            st.caption("One-factor-at-a-time model exploration. Other factors remain fixed at the values shown.")
            with st.expander("View and download response data"):
                st.dataframe(curve, use_container_width=True)
                download_csv(curve, "fermentai_one_factor_analysis.csv", "Download curve data")

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                x_factor = st.selectbox(
                    "Horizontal axis", FEATURES, index=0,
                    format_func=lambda f: LABELS[f], key="surface_x"
                )
            with c2:
                y_options = [f for f in FEATURES if f != x_factor]
                y_factor = st.selectbox(
                    "Vertical axis", y_options, index=0,
                    format_func=lambda f: LABELS[f], key="surface_y"
                )
            fixed = make_reference_values()
            remaining = [f for f in FEATURES if f not in [x_factor, y_factor]]
            st.markdown("#### Fixed values for the remaining factors")
            fixed_cols = st.columns(2)
            for col, feature in zip(fixed_cols, remaining):
                with col:
                    fixed[feature] = parameter_control(
                        feature, f"surface_fixed_{feature}", REFERENCE_CONDITION[feature]
                    )
            xlo, xhi, _, _ = RANGES[x_factor]
            ylo, yhi, _, _ = RANGES[y_factor]
            xvals = np.linspace(xlo, xhi, 35)
            yvals = np.linspace(ylo, yhi, 35)
            grid_rows = []
            z = []
            for yv in yvals:
                row_z = []
                for xv in xvals:
                    trial = fixed.copy()
                    trial[x_factor] = float(xv)
                    trial[y_factor] = float(yv)
                    row_z.append(predict_one(model, trial))
                z.append(row_z)
            surface = go.Figure(data=go.Contour(
                x=xvals, y=yvals, z=z, colorscale="Teal",
                contours=dict(showlabels=True),
                colorbar=dict(title="U/mL")
            ))
            surface.update_layout(
                title=f"Predicted response map: {LABELS[x_factor]} vs {LABELS[y_factor]}",
                xaxis_title=f"{LABELS[x_factor]} ({UNITS[x_factor]})",
                yaxis_title=f"{LABELS[y_factor]} ({UNITS[y_factor]})",
                template="plotly_white", height=510
            )
            st.plotly_chart(surface, use_container_width=True)
            st.caption("The heatmap is generated from the Random Forest model, not from new wet-lab measurements.")


# ============================================================
# 4. MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":
    hero("Model Performance", "Inspect model structure and compare its estimates with the published response table.")

    if require_model():
        a, b, c = st.columns(3)
        with a: kpi("Algorithm", type(model).__name__, "Regression")
        with b: kpi("Trees", str(getattr(model, "n_estimators", "—")), "Random Forest estimators")
        with c: kpi("Input variables", str(getattr(model, "n_features_in_", 4)), "Exact model inputs")

        st.subheader("Feature importance")
        if hasattr(model, "feature_importances_"):
            imp = pd.DataFrame({
                "Feature": [LABELS[f] for f in FEATURES],
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=True)
            fig = px.bar(
                imp, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Teal",
                template="plotly_white", title="Relative feature importance"
            )
            fig.update_layout(height=360, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Feature importance describes the fitted forest; it does not establish biological causation.")
        else:
            st.info("This model object does not expose feature_importances_.")

        if data is not None:
            st.subheader("Model estimate vs published experimental observations")
            comparison = data[FEATURES + ["Amylase_Activity_Experimental_U_mL"]].copy()
            try:
                comparison["Random_Forest_Estimate_U_mL"] = predict_many(model, comparison)
                comparison["Absolute_Difference_U_mL"] = (
                    comparison["Random_Forest_Estimate_U_mL"] -
                    comparison["Amylase_Activity_Experimental_U_mL"]
                ).abs()
                mae = comparison["Absolute_Difference_U_mL"].mean()
                rmse = np.sqrt(np.mean(
                    comparison["Absolute_Difference_U_mL"] ** 2
                ))
                m1, m2 = st.columns(2)
                m1.metric("Mean absolute difference", f"{mae:.2f} U/mL")
                m2.metric("Root mean squared difference", f"{rmse:.2f} U/mL")
                fig = px.scatter(
                    comparison,
                    x="Amylase_Activity_Experimental_U_mL",
                    y="Random_Forest_Estimate_U_mL",
                    hover_data=["Initial_pH","Temperature_C","Agitation_rpm","Incubation_Time_h"],
                    template="plotly_white",
                    title="Published experimental values vs Random Forest estimates"
                )
                low = min(
                    comparison["Amylase_Activity_Experimental_U_mL"].min(),
                    comparison["Random_Forest_Estimate_U_mL"].min()
                )
                high = max(
                    comparison["Amylase_Activity_Experimental_U_mL"].max(),
                    comparison["Random_Forest_Estimate_U_mL"].max()
                )
                fig.add_trace(go.Scatter(
                    x=[low, high], y=[low, high], mode="lines",
                    name="Ideal agreement", line=dict(dash="dash", color="#667788")
                ))
                fig.update_layout(height=430)
                st.plotly_chart(fig, use_container_width=True)
                note(
                    "<b>Interpret carefully:</b> These are descriptive differences across "
                    "the 29 published design points. The original training membership of "
                    "the supplied pickle is not documented here, so these values must not "
                    "be labelled independent test-set performance or validation metrics.",
                    warning=True
                )
                with st.expander("View comparison table"):
                    st.dataframe(comparison, use_container_width=True)
            except Exception as exc:
                st.error(f"Comparison could not be calculated: {exc}")


# ============================================================
# 5. DATASET EXPLORER
# ============================================================
elif page == "Dataset Explorer":
    hero("Dataset Explorer", "Explore the 29-run Box–Behnken design and its reported responses.")

    if data is None:
        st.error(f"Dataset unavailable: {data_status}")
    else:
        d1, d2, d3, d4 = st.columns(4)
        with d1: kpi("Experimental runs", str(len(data)), "Published BBD table")
        with d2: kpi("Factors", "4", "Three-level design")
        with d3: kpi("Maximum reported activity", f"{data['Amylase_Activity_Experimental_U_mL'].max():.2f}", "U/mL")
        with d4: kpi("Center-point replicates", str(int((data[['A_code','B_code','C_code','D_code']].eq(0).all(axis=1)).sum())), "All coded factors = 0")

        tab1, tab2, tab3 = st.tabs(["Run table", "Response distribution", "Factor relationships"])
        with tab1:
            st.dataframe(data, use_container_width=True, hide_index=True)
            download_csv(data, "BBD_29_Runs_For_Verification_FIXED.csv", "Download 29-run dataset")
        with tab2:
            fig = px.histogram(
                data, x="Amylase_Activity_Experimental_U_mL", nbins=10,
                template="plotly_white", color_discrete_sequence=["#07858A"],
                title="Distribution of published experimental α-amylase activity"
            )
            fig.update_layout(height=380, xaxis_title="Experimental activity (U/mL)", yaxis_title="Number of runs")
            st.plotly_chart(fig, use_container_width=True)
            fig2 = px.box(
                data, y="Amylase_Activity_Experimental_U_mL",
                points="all", template="plotly_white",
                title="Spread of experimental activity values"
            )
            fig2.update_traces(marker_color="#07858A")
            st.plotly_chart(fig2, use_container_width=True)
        with tab3:
            selected = st.selectbox(
                "Compare activity against", FEATURES,
                format_func=lambda f: LABELS[f]
            )
            fig = px.scatter(
                data, x=selected, y="Amylase_Activity_Experimental_U_mL",
                color="Amylase_Activity_Experimental_U_mL",
                color_continuous_scale="Teal", size_max=12,
                hover_data=["Run","Temperature_C","Agitation_rpm","Incubation_Time_h"],
                template="plotly_white",
                title=f"Experimental activity vs {LABELS[selected]}"
            )
            fig.update_layout(height=430, xaxis_title=f"{LABELS[selected]} ({UNITS[selected]})", yaxis_title="Activity (U/mL)")
            st.plotly_chart(fig, use_container_width=True)

        note(
            "Dataset source: Table 1 and Table 2 of Khusro et al. (2017). "
            "The factor settings are converted from coded levels (−1, 0, +1) to "
            "the actual pH, temperature, agitation and incubation-time settings. "
            "The response values are transcribed from the paper's experimental column.",
            warning=False
        )


# ============================================================
# 6. OPTIMIZATION LAB
# ============================================================
elif page == "Optimization Lab":
    hero("Optimization Lab", "Screen all combinations of the three published factor levels.")

    if require_model():
        st.write(
            "This lab evaluates a discrete 3 × 3 × 3 × 3 grid (81 combinations) "
            "using the Random Forest model. It ranks candidates for further study."
        )
        run = st.button("🧪 Screen 81 candidate conditions", type="primary", use_container_width=True)
        if run:
            level_values = {
                "Initial_pH": [4.0, 6.0, 8.0],
                "Temperature_C": [25.0, 35.0, 45.0],
                "Agitation_rpm": [120.0, 130.0, 140.0],
                "Incubation_Time_h": [12.0, 24.0, 48.0],
            }
            combos = list(itertools.product(*(level_values[f] for f in FEATURES)))
            candidates = pd.DataFrame(combos, columns=FEATURES)
            try:
                candidates["Predicted_Activity_U_mL"] = predict_many(model, candidates)
                ranked = candidates.sort_values(
                    "Predicted_Activity_U_mL", ascending=False
                ).reset_index(drop=True)
                ranked.insert(0, "Rank", np.arange(1, len(ranked)+1))
                st.session_state["optimization_results"] = ranked
            except Exception as exc:
                st.error(f"Candidate screening failed: {exc}")

        if "optimization_results" in st.session_state:
            ranked = st.session_state["optimization_results"]
            best = ranked.iloc[0]
            st.subheader("Highest-ranked model candidate")
            x1, x2 = st.columns([1, 2])
            with x1:
                st.metric("Predicted activity", f"{best['Predicted_Activity_U_mL']:.2f} U/mL")
                st.metric("Candidate rank", "1 of 81")
            with x2:
                st.dataframe(ranked.head(10), hide_index=True, use_container_width=True)
            fig = px.bar(
                ranked.head(15).sort_values("Predicted_Activity_U_mL"),
                x="Predicted_Activity_U_mL", y="Rank", orientation="h",
                color="Predicted_Activity_U_mL", color_continuous_scale="Teal",
                template="plotly_white", title="Top 15 ranked candidate conditions"
            )
            fig.update_layout(height=480, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            download_csv(ranked, "fermentai_81_candidate_screening.csv", "Download all 81 candidates")
        note(
            "<b>Important:</b> This is model-based candidate screening, not proof of a "
            "biological optimum. Confirm any promising condition experimentally. "
            "The model may behave differently between and beyond the original design points.",
            warning=True
        )


# ============================================================
# 7. SCIENTIFIC INSIGHTS
# ============================================================
elif page == "Scientific Insights":
    hero("Scientific Insights", "Understand the fermentation factors and the published study behind the dashboard.")

    st.subheader("What is submerged fermentation?")
    st.write(
        "Submerged fermentation (SmF) is a microbial cultivation process in which "
        "microorganisms grow in a liquid nutrient medium. Process variables can influence "
        "microbial growth, oxygen transfer and enzyme production."
    )
    factor_text = {
        "Initial_pH": "pH affects enzyme expression, nutrient availability and microbial physiology.",
        "Temperature_C": "Temperature influences microbial growth rate, metabolism and enzyme stability.",
        "Agitation_rpm": "Agitation supports mixing and oxygen transfer in the fermentation broth.",
        "Incubation_Time_h": "Incubation time determines how long the organism can grow and accumulate enzyme."
    }
    for f in FEATURES:
        with st.expander(f"{LABELS[f]} — why it matters"):
            st.write(factor_text[f])

    st.subheader("Published reference condition")
    a,b,c,d = st.columns(4)
    with a: kpi("pH", "6.0", "Initial pH")
    with b: kpi("Temperature", "35 °C", "Cultivation temperature")
    with c: kpi("Agitation", "130 rpm", "Shaking speed")
    with d: kpi("Incubation", "24 h", "Incubation period")
    st.metric("Reported experimental α-amylase activity", "145.32 U/mL")
    st.write(
        "Khusro et al. (2017) reported this maximum activity for the selected "
        "Bacillus sp. under the stated submerged-cultivation conditions. Their paper "
        "also reports a 29-run Box–Behnken design and a response-surface model."
    )
    st.markdown("[Read the original article — Fermentation (2017), 3(1), 7](https://www.mdpi.com/2311-5637/3/1/7)")

    st.subheader("Model limitations")
    st.markdown("""
    - A 29-run design is a small dataset for machine learning.
    - Random Forest predictions are estimates, not laboratory measurements.
    - Feature importance is not proof of biological causation.
    - The original training membership of the supplied pickle is not recorded in this app.
    - Experimental confirmation is required before claiming a new optimum.
    """)


# ============================================================
# 8. REPORT CENTER
# ============================================================
elif page == "Report Center":
    hero("Report Center", "Review and export your current session's prediction and candidate-screening results.")

    st.subheader("Prediction report")
    if "last_prediction" in st.session_state:
        pred = pd.DataFrame([st.session_state["last_prediction"]])
        st.dataframe(pred, use_container_width=True)
        download_csv(pred, "fermentai_prediction_report.csv", "Download prediction report")
    else:
        st.info("No prediction has been generated in this session. Open Smart Optimizer first.")

    st.divider()
    st.subheader("Optimization report")
    if "optimization_results" in st.session_state:
        opt = st.session_state["optimization_results"]
        st.dataframe(opt.head(20), use_container_width=True)
        download_csv(opt, "fermentai_optimization_report.csv", "Download optimization report")
    else:
        st.info("No candidate screening has been completed yet. Open Optimization Lab first.")

    st.divider()
    st.subheader("Reference dataset")
    if data is not None:
        download_csv(data, "BBD_29_Runs_For_Verification_FIXED.csv", "Download reference 29-run dataset")

    note(
        "Exports contain published observations and/or model-generated estimates. "
        "Keep these categories separate in reports and presentations.",
        warning=True
    )

# --------------------------- Footer ---------------------------
st.markdown("---")
st.caption(
    "FermentAI • Student research prototype • Random Forest estimates are computational "
    "outputs and require experimental confirmation."
)
