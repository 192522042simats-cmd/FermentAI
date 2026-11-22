import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# ============================================================
# FermentAI | ML-Based Microbial alpha-Amylase Production Optimizer
# ============================================================

st.set_page_config(
    page_title="FermentAI | Fermentation Optimizer",
    page_icon="🧫",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "random_forest_model.pkl"
DATA_PATH = ROOT / "BBD_29_Runs_For_Verification_FIXED.csv"

FEATURES = ["Initial_pH", "Temperature_C", "Agitation_rpm", "Incubation_Time_h"]
FEATURE_LABELS = {
    "Initial_pH": "Initial pH",
    "Temperature_C": "Temperature (°C)",
    "Agitation_rpm": "Agitation (rpm)",
    "Incubation_Time_h": "Incubation time (h)"
}
DEFAULTS = {
    "Initial_pH": 6.0,
    "Temperature_C": 35,
    "Agitation_rpm": 130,
    "Incubation_Time_h": 24
}
RANGES = {
    "Initial_pH": (4.0, 8.0, 0.1),
    "Temperature_C": (25, 45, 1),
    "Agitation_rpm": (120, 140, 1),
    "Incubation_Time_h": (12, 48, 1)
}

# ------------------------- Styling -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f5f8fb; color: #172b3a; }
[data-testid="stHeader"] { background: rgba(245,248,251,0.92); }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1440px; }
h1, h2, h3 { font-family: 'Manrope', sans-serif; color: #142d40; }
.hero {
    background: linear-gradient(120deg, #12384a 0%, #126b70 100%);
    padding: 2rem 2.2rem; border-radius: 20px; color: white; margin-bottom: 1.4rem;
}
.hero h1 { color: white; font-size: 2.25rem; margin: 0 0 .4rem 0; }
.hero p { color: #d8eeee; margin: 0; font-size: 1.02rem; }
.kpi {
    background: white; border: 1px solid #e2eaf0; border-radius: 16px;
    padding: 1rem 1.2rem; min-height: 112px;
}
.kpi-label { color: #607687; font-size: .83rem; font-weight: 600; }
.kpi-value { color: #12384a; font-family: 'Manrope',sans-serif; font-size: 1.65rem; font-weight: 800; margin-top: .3rem; }
.kpi-note { color: #8193a0; font-size: .76rem; margin-top: .1rem; }
.panel {
    background: white; border: 1px solid #e2eaf0; border-radius: 16px;
    padding: 1.2rem 1.3rem; margin-bottom: 1rem;
}
.small-muted { color: #6b7f8d; font-size: .88rem; }
div[data-testid="stMetric"] {
    background: white; border: 1px solid #e2eaf0; border-radius: 14px; padding: 14px;
}
.stButton > button { border-radius: 10px; font-weight: 600; }
section[data-testid="stSidebar"] { background: #edf3f6; }
hr { border-color: #e1e9ee; }
</style>
""", unsafe_allow_html=True)

# ------------------------- Data/model loading -------------------------
@st.cache_resource
def load_model(path_string):
    return joblib.load(path_string)

@st.cache_data
def load_dataset(path_string):
    return pd.read_csv(path_string)

def find_target_column(df):
    candidates = [
        "Amylase_Activity_U_mL", "Amylase_Activity", "Activity_U_mL",
        "Enzyme_Activity_U_mL", "Alpha_Amylase_Activity", "Predicted_Activity",
        "Activity", "Response"
    ]
    for name in candidates:
        if name in df.columns:
            return name
    # Use the first non-feature numeric column as a best-effort target.
    numeric = [c for c in df.select_dtypes(include=np.number).columns if c not in FEATURES]
    return numeric[0] if numeric else None

def model_feature_order(model):
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        names = list(names)
        if all(name in FEATURES for name in names):
            return names
    return FEATURES

def predict_one(model, params):
    order = model_feature_order(model)
    row = pd.DataFrame([[params[name] for name in order]], columns=order)
    return float(np.asarray(model.predict(row)).reshape(-1)[0])

model = None
model_error = None
if MODEL_PATH.exists():
    try:
        model = load_model(str(MODEL_PATH))
    except Exception as exc:
        model_error = str(exc)
else:
    model_error = "random_forest_model.pkl was not found in the app folder."

dataset = None
dataset_error = None
if DATA_PATH.exists():
    try:
        dataset = load_dataset(str(DATA_PATH))
    except Exception as exc:
        dataset_error = str(exc)

# ------------------------- Sidebar -------------------------
st.sidebar.markdown("## 🧫 FermentAI")
st.sidebar.caption("Fermentation intelligence platform")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Fermentation Optimizer", "Parameter Analysis",
     "ML Insights", "Dataset Explorer", "Knowledge Center"]
)
st.sidebar.divider()
st.sidebar.caption("Microbial α-amylase • Submerged fermentation")
st.sidebar.caption("Prototype version")

# ------------------------- Helpers -------------------------
def hero(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True
    )

def kpi(label, value, note=""):
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True
    )

def require_model():
    if model is None:
        st.error("The Random Forest model could not be loaded.")
        st.code(model_error or "Unknown model-loading error")
        st.info("Check that random_forest_model.pkl is uploaded beside app.py and that joblib is listed in requirements.txt.")
        return False
    return True

# ------------------------- Overview -------------------------
if page == "Overview":
    hero("FermentAI", "Data-driven modelling and machine-learning support for microbial α-amylase production.")
    st.markdown("### Project overview")
    st.write(
        "FermentAI is a prototype decision-support application for exploring how selected "
        "submerged-fermentation conditions relate to predicted α-amylase activity."
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Model", "Random Forest", "Regression model")
    with c2:
        kpi("Input factors", "04", "pH, temperature, agitation, time")
    with c3:
        kpi("Design reference", "29 runs", "Box–Behnken design")
    with c4:
        kpi("Fermentation", "SmF", "Submerged cultivation")
    st.markdown("### Workflow")
    cols = st.columns(4)
    workflow = [
        ("01", "Set conditions", "Choose pH, temperature, agitation and incubation time."),
        ("02", "Run prediction", "The saved regression model estimates the response."),
        ("03", "Explore factors", "Review one-factor response curves and model insights."),
        ("04", "Inspect data", "Explore the available design dataset and export records.")
    ]
    for col, (num, title, desc) in zip(cols, workflow):
        with col:
            st.markdown(f'<div class="panel"><div class="small-muted">{num}</div><b>{title}</b><p class="small-muted">{desc}</p></div>', unsafe_allow_html=True)
    st.warning(
        "Scientific transparency: the CSV supplied with this prototype is a reconstructed/"
        "model-generated dataset for software demonstration, not the original raw experimental "
        "measurements from the reference publication. Predictions must not be presented as "
        "experimentally validated results."
    )
    if model is None:
        st.error(f"Model status: unavailable — {model_error}")
    else:
        st.success("Model file loaded successfully.")

# ------------------------- Fermentation Optimizer -------------------------
elif page == "Fermentation Optimizer":
    hero("Fermentation Optimizer", "Enter process conditions and estimate α-amylase activity.")
    if require_model():
        left, right = st.columns([1.05, 0.95], gap="large")
        with left:
            st.markdown("### Process parameters")
            ph = st.slider("Initial pH", 4.0, 8.0, 6.0, 0.1)
            temp = st.slider("Temperature (°C)", 25, 45, 35, 1)
            agitation = st.slider("Agitation speed (rpm)", 120, 140, 130, 1)
            incubation = st.slider("Incubation time (hours)", 12, 48, 24, 1)
            run_prediction = st.button("Predict enzyme activity", type="primary", use_container_width=True)
        with right:
            st.markdown("### Prediction result")
            if run_prediction or "last_prediction" not in st.session_state:
                params = {
                    "Initial_pH": float(ph),
                    "Temperature_C": float(temp),
                    "Agitation_rpm": float(agitation),
                    "Incubation_Time_h": float(incubation)
                }
                try:
                    st.session_state["last_prediction"] = predict_one(model, params)
                    st.session_state["last_params"] = params
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
            if "last_prediction" in st.session_state:
                st.metric("Predicted α-amylase activity", f'{st.session_state["last_prediction"]:.2f} U/mL')
                st.caption("Model estimate only — not a measured laboratory result.")
                st.markdown("**Conditions used**")
                for key, value in st.session_state["last_params"].items():
                    st.write(f"• {FEATURE_LABELS[key]}: {value}")
        st.info("The model estimates activity for the selected input combination. It does not experimentally optimize or validate fermentation conditions.")

# ------------------------- Parameter Analysis -------------------------
elif page == "Parameter Analysis":
    hero("Parameter Analysis", "Explore how changing one process factor affects the model prediction.")
    if require_model():
        selected = st.selectbox(
            "Parameter to explore",
            ["Initial_pH", "Temperature_C", "Agitation_rpm", "Incubation_Time_h"],
            format_func=lambda key: FEATURE_LABELS[key]
        )
        base = DEFAULTS.copy()
        low, high, step = RANGES[selected]
        if selected == "Initial_pH":
            values = np.linspace(low, high, 41)
        else:
            values = np.arange(low, high + 1, step)
        rows = []
        try:
            for value in values:
                params = base.copy()
                params[selected] = float(value)
                prediction = predict_one(model, params)
                rows.append({"Parameter value": float(value), "Predicted activity (U/mL)": prediction})
            curve = pd.DataFrame(rows)
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.plot(curve["Parameter value"], curve["Predicted activity (U/mL)"], marker="o", linewidth=2)
            ax.axvline(DEFAULTS[selected], linestyle="--", alpha=0.65, label="Reference setting")
            ax.set_xlabel(FEATURE_LABELS[selected])
            ax.set_ylabel("Predicted activity (U/mL)")
            ax.set_title(f"Predicted response across {FEATURE_LABELS[selected]}")
            ax.grid(True, alpha=0.25)
            ax.legend()
            st.pyplot(fig, use_container_width=True)
            st.dataframe(curve, use_container_width=True, hide_index=True)
            st.download_button(
                "Download response data",
                curve.to_csv(index=False).encode("utf-8"),
                file_name=f"FermentAI_{selected}_response.csv",
                mime="text/csv"
            )
        except Exception as exc:
            st.error(f"Could not generate the response curve: {exc}")

# ------------------------- ML Insights -------------------------
elif page == "ML Insights":
    hero("ML Insights", "Review the saved model and the information available for interpretation.")
    if require_model():
        st.markdown("### Model information")
        a, b = st.columns(2)
        with a:
            st.markdown('<div class="panel"><b>Algorithm</b><p class="small-muted">Random Forest Regressor</p></div>', unsafe_allow_html=True)
        with b:
            st.markdown('<div class="panel"><b>Model input variables</b><p class="small-muted">Initial pH, temperature, agitation speed and incubation time</p></div>', unsafe_allow_html=True)
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            order = model_feature_order(model)
            if len(order) == len(importances):
                fi = pd.DataFrame({"Parameter": [FEATURE_LABELS.get(x, x) for x in order],
                                   "Importance": np.asarray(importances, dtype=float)})
                fi = fi.sort_values("Importance", ascending=True)
                st.markdown("### Feature importance")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(fi["Parameter"], fi["Importance"])
                ax.set_xlabel("Importance score")
                ax.set_title("Random Forest feature importance")
                ax.grid(axis="x", alpha=0.2)
                st.pyplot(fig, use_container_width=True)
                st.caption("Feature importance indicates model reliance, not cause-and-effect or biological mechanism.")
            else:
                st.info("Feature importance could not be aligned with the model's input variables.")
        else:
            st.info("This model does not expose feature_importances_.")
        st.warning("The model was trained using a reconstructed/model-generated target dataset. Its performance is not independent experimental validation.")
        if hasattr(model, "n_estimators"):
            st.write(f"Number of trees: **{model.n_estimators}**")

# ------------------------- Dataset Explorer -------------------------
elif page == "Dataset Explorer":
    hero("Dataset Explorer", "Inspect, filter and export the fermentation design dataset.")
    uploaded = st.file_uploader("Upload a CSV dataset (optional)", type=["csv"])
    df = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success("Uploaded dataset loaded.")
        except Exception as exc:
            st.error(f"Could not read uploaded CSV: {exc}")
    elif dataset is not None:
        df = dataset.copy()
    elif dataset_error:
        st.error(f"Dataset could not be loaded: {dataset_error}")
    else:
        st.info("No default CSV found. Upload a CSV to explore it.")
    if df is not None:
        st.markdown("### Dataset summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows", f"{len(df):,}")
        m2.metric("Columns", f"{len(df.columns):,}")
        m3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        st.dataframe(df, use_container_width=True, height=380)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            st.markdown("### Filter numeric values")
            filter_col = st.selectbox("Choose a numeric column", numeric_cols)
            min_value = float(df[filter_col].min())
            max_value = float(df[filter_col].max())
            if np.isfinite(min_value) and np.isfinite(max_value) and min_value < max_value:
                selected_range = st.slider(
                    f"Range for {filter_col}",
                    min_value=min_value,
                    max_value=max_value,
                    value=(min_value, max_value)
                )
                filtered = df[df[filter_col].between(selected_range[0], selected_range[1])]
            else:
                filtered = df.copy()
            st.write(f"Showing **{len(filtered)}** of **{len(df)}** rows.")
            st.dataframe(filtered, use_container_width=True, hide_index=True)
            st.download_button(
                "Download displayed data",
                filtered.to_csv(index=False).encode("utf-8"),
                file_name="FermentAI_filtered_dataset.csv",
                mime="text/csv"
            )
        else:
            st.download_button("Download dataset", df.to_csv(index=False).encode("utf-8"),
                               file_name="FermentAI_dataset.csv", mime="text/csv")

# ------------------------- Knowledge Center -------------------------
elif page == "Knowledge Center":
    hero("Knowledge Center", "Key concepts behind the FermentAI prototype.")
    with st.expander("What is submerged fermentation?"):
        st.write("Submerged fermentation (SmF) is a microbial cultivation process in which microorganisms grow in a liquid nutrient medium. Process conditions such as pH, temperature, agitation and incubation time can influence growth and enzyme production.")
    with st.expander("What is α-amylase?"):
        st.write("α-Amylase is an enzyme that hydrolyses internal α-1,4-glycosidic bonds in starch, producing shorter dextrins and sugars.")
    with st.expander("What is a Box–Behnken design?"):
        st.write("A Box–Behnken design is a response-surface experimental design used to study the effects and interactions of factors with a relatively limited number of experimental runs.")
    with st.expander("What does the Random Forest model do?"):
        st.write("A Random Forest Regressor combines predictions from multiple decision trees to estimate a continuous response from the selected process parameters.")
    with st.expander("How should predictions be interpreted?"):
        st.write("Predictions are computational estimates generated by the saved model. They are not direct measurements, proof of causality, or experimentally confirmed optimum conditions.")
    st.markdown("### Reference study")
    st.write("Khusro et al. (2017), “Optimization of Thermo-Alkali Stable Amylase Production and Biomass Yield from Bacillus sp. Under Submerged Cultivation,” Fermentation, 3(1), 7.")
    st.caption("Reference-study context is provided for educational background. The demonstration dataset in this app is not the original raw experimental dataset.")

st.sidebar.divider()
st.sidebar.caption("FermentAI • Academic prototype")
