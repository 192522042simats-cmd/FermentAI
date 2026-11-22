import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from datetime import datetime

# ============================================================
# FermentAI | Professional fermentation decision-support app
# ============================================================

st.set_page_config(
    page_title="FermentAI | Smart Fermentation Platform",
    page_icon="🧫",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "random_forest_model.pkl"
DATA_PATH = ROOT / "BBD_29_Runs_For_Verification_FIXED.csv"

FEATURES = ["Initial_pH", "Temperature_C", "Agitation_rpm", "Incubation_Time_h"]
LABELS = {
    "Initial_pH": "Initial pH",
    "Temperature_C": "Temperature (°C)",
    "Agitation_rpm": "Agitation speed (rpm)",
    "Incubation_Time_h": "Incubation time (h)"
}
DEFAULTS = {
    "Initial_pH": 6.0,
    "Temperature_C": 35.0,
    "Agitation_rpm": 130.0,
    "Incubation_Time_h": 24.0
}
BOUNDS = {
    "Initial_pH": (4.0, 8.0),
    "Temperature_C": (25.0, 45.0),
    "Agitation_rpm": (120.0, 140.0),
    "Incubation_Time_h": (12.0, 48.0)
}

# --------------------------- Design ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:#f4f7fa;color:#172d3d;}
[data-testid="stHeader"] {background:rgba(244,247,250,.94);}
.block-container {padding-top:1.35rem;padding-bottom:3rem;max-width:1500px;}
h1,h2,h3 {font-family:'Manrope',sans-serif;color:#15364a;}
section[data-testid="stSidebar"] {background:#eaf1f5;border-right:1px solid #dce6ec;}
.hero {background:linear-gradient(115deg,#12364a 0%,#126d70 100%);padding:2rem 2.2rem;border-radius:20px;color:white;margin-bottom:1.25rem;}
.hero h1 {color:white;font-size:2.25rem;margin:0 0 .35rem 0;font-weight:800;}
.hero p {color:#d9eeee;margin:0;font-size:1rem;}
.eyebrow {color:#8fd6cc;text-transform:uppercase;letter-spacing:.13em;font-size:.72rem;font-weight:700;margin-bottom:.5rem;}
.kpi {background:white;border:1px solid #e1e9ef;border-radius:16px;padding:1rem 1.1rem;min-height:112px;box-shadow:0 3px 12px rgba(24,55,73,.035);}
.kpi-label {color:#667d8c;font-size:.8rem;font-weight:600;}
.kpi-value {font-family:'Manrope',sans-serif;color:#15394d;font-size:1.55rem;font-weight:800;margin-top:.35rem;}
.kpi-note {color:#8a9ba6;font-size:.75rem;margin-top:.1rem;}
.panel {background:white;border:1px solid #e1e9ef;border-radius:16px;padding:1.15rem 1.25rem;margin-bottom:.9rem;}
.muted {color:#6b808e;font-size:.88rem;}
.tag {display:inline-block;background:#e7f4f2;color:#176b68;padding:.22rem .55rem;border-radius:20px;font-size:.75rem;font-weight:700;}
.stButton>button,.stDownloadButton>button {border-radius:10px;font-weight:650;min-height:2.55rem;}
div[data-testid="stMetric"] {background:white;border:1px solid #e1e9ef;border-radius:14px;padding:13px;}
div[data-testid="stAlert"] {border-radius:12px;}
hr {border-color:#e1e9ef;}
</style>
""", unsafe_allow_html=True)

# ------------------------- Load assets -------------------------
@st.cache_resource
def load_model(file_path):
    return joblib.load(file_path)

@st.cache_data
def load_csv(file_path):
    return pd.read_csv(file_path)

model, model_error = None, None
if MODEL_PATH.exists():
    try:
        model = load_model(str(MODEL_PATH))
    except Exception as exc:
        model_error = str(exc)
else:
    model_error = "random_forest_model.pkl is missing from the repository root."

dataset, data_error = None, None
if DATA_PATH.exists():
    try:
        dataset = load_csv(str(DATA_PATH))
    except Exception as exc:
        data_error = str(exc)
else:
    data_error = "Default dataset CSV is missing. You can upload a CSV in Dataset Explorer."

# ------------------------- Core utilities -------------------------
def get_model_features():
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        names = list(names)
        if len(names) == len(FEATURES) and set(names) == set(FEATURES):
            return names
    return FEATURES

def predict_activity(params):
    if model is None:
        raise RuntimeError(model_error or "Model unavailable.")
    order = get_model_features()
    x = pd.DataFrame([[float(params[k]) for k in order]], columns=order)
    return float(np.asarray(model.predict(x)).reshape(-1)[0])

def target_column(df):
    preferred = [
        "Amylase_Activity_U_mL", "Amylase_Activity", "Activity_U_mL",
        "Enzyme_Activity_U_mL", "Alpha_Amylase_Activity",
        "Predicted_Activity", "Activity", "Response"
    ]
    for col in preferred:
        if col in df.columns:
            return col
    candidates = [c for c in df.select_dtypes(include=np.number).columns if c not in FEATURES]
    return candidates[0] if candidates else None

def hero(title, subtitle, eyebrow="FERMENTATION INTELLIGENCE"):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True
    )

def kpi(label, value, note=""):
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True
    )

def model_status():
    if model is None:
        st.error("The saved model could not be loaded.")
        st.code(model_error or "Unknown model-loading error")
        st.info("Confirm that random_forest_model.pkl is in the same GitHub folder as app.py and that joblib is in requirements.txt.")
        return False
    return True

def params_from_values(ph, temp, rpm, hours):
    return {
        "Initial_pH": float(ph),
        "Temperature_C": float(temp),
        "Agitation_rpm": float(rpm),
        "Incubation_Time_h": float(hours)
    }

def make_csv_download(df, filename, label="Download CSV"):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), filename,
                       "text/csv", use_container_width=False)

# ------------------------- Sidebar navigation -------------------------
st.sidebar.markdown("# 🧫 FermentAI")
st.sidebar.caption("SMART FERMENTATION PLATFORM")
st.sidebar.markdown("---")
pages = [
    "Overview", "Smart Optimizer", "Parameter Analysis", "Model Performance",
    "Dataset Explorer", "Optimization Lab", "Scientific Insights", "Report Center"
]
page = st.sidebar.radio("WORKSPACE", pages, index=0)
st.sidebar.markdown("---")
st.sidebar.markdown("**Project**")
st.sidebar.caption("Microbial α-amylase production")
st.sidebar.caption("Submerged fermentation • ML prototype")
st.sidebar.markdown("---")
if model is not None:
    st.sidebar.success("Model loaded")
else:
    st.sidebar.error("Model unavailable")
if dataset is not None:
    st.sidebar.caption(f"Dataset: {len(dataset)} rows")
else:
    st.sidebar.caption("Dataset: upload required")
st.sidebar.caption("Prototype • Educational/research use")

# ------------------------- 1. Overview -------------------------
if page == "Overview":
    hero("FermentAI", "A machine-learning interface for exploring microbial α-amylase production conditions.")
    st.markdown("### Platform at a glance")
    c = st.columns(4)
    with c[0]: kpi("Prediction engine", "Random Forest", "Regression model")
    with c[1]: kpi("Process variables", "04", "pH • temperature • agitation • time")
    with c[2]: kpi("Design framework", "BBD", "Box–Behnken reference design")
    with c[3]: kpi("Cultivation mode", "SmF", "Submerged fermentation")
    st.markdown("### Your workflow")
    workflow = [
        ("01", "Smart Optimizer", "Set operating conditions and estimate enzyme activity."),
        ("02", "Parameter Analysis", "Explore one-factor response curves."),
        ("03", "Optimization Lab", "Compare candidate settings within defined ranges."),
        ("04", "Report Center", "Download a summary of selected conditions and estimates.")
    ]
    cols = st.columns(4)
    for col, (n, title, desc) in zip(cols, workflow):
        with col:
            st.markdown(f'<div class="panel"><span class="tag">{n}</span><h4>{title}</h4><div class="muted">{desc}</div></div>', unsafe_allow_html=True)
    st.markdown("### System status")
    s1, s2 = st.columns(2)
    with s1:
        if model is not None: st.success("Random Forest model file loaded.")
        else: st.error(f"Model issue: {model_error}")
    with s2:
        if dataset is not None: st.success(f"Dataset loaded: {len(dataset)} rows × {len(dataset.columns)} columns.")
        else: st.warning(data_error or "Dataset not loaded.")
    st.warning(
        "**Research transparency:** the supplied BBD CSV is a reconstructed/model-generated "
        "demonstration dataset, not the original raw experimental measurements. The saved model "
        "was trained using that reconstructed target data. Therefore, predictions and optimization "
        "outputs are prototype estimates, not experimentally validated results."
    )

# ------------------------- 2. Smart Optimizer -------------------------
elif page == "Smart Optimizer":
    hero("Smart Optimizer", "Adjust fermentation settings and generate a model-based activity estimate.")
    if model_status():
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown("### Define process conditions")
            ph = st.slider("Initial pH", 4.0, 8.0, 6.0, 0.1, key="opt_ph")
            temp = st.slider("Temperature (°C)", 25, 45, 35, 1, key="opt_temp")
            rpm = st.slider("Agitation speed (rpm)", 120, 140, 130, 1, key="opt_rpm")
            hours = st.slider("Incubation time (hours)", 12, 48, 24, 1, key="opt_hours")
            predict_clicked = st.button("⚡ Generate prediction", type="primary", use_container_width=True)
            st.caption("Input limits follow the factor ranges used in this prototype.")
        with right:
            st.markdown("### Prediction output")
            if predict_clicked or "fermentai_prediction" not in st.session_state:
                try:
                    p = params_from_values(ph, temp, rpm, hours)
                    st.session_state["fermentai_prediction"] = predict_activity(p)
                    st.session_state["fermentai_params"] = p
                    st.session_state["fermentai_prediction_time"] = datetime.now().strftime("%d %b %Y, %H:%M")
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
            if "fermentai_prediction" in st.session_state:
                st.metric("Predicted α-amylase activity", f'{st.session_state["fermentai_prediction"]:.2f} U/mL')
                st.markdown("**Input conditions**")
                p = st.session_state["fermentai_params"]
                for key in FEATURES:
                    st.write(f"• {LABELS[key]}: **{p[key]:g}**")
                st.caption(f"Generated: {st.session_state.get('fermentai_prediction_time', '—')}")
                st.download_button(
                    "Download prediction summary",
                    pd.DataFrame([{
                        "Predicted_Activity_U_mL": st.session_state["fermentai_prediction"],
                        **p
                    }]).to_csv(index=False).encode("utf-8"),
                    "FermentAI_prediction.csv", "text/csv"
                )
        st.info("This is a model estimate, not a laboratory measurement or proof of an optimum.")

# ------------------------- 3. Parameter Analysis -------------------------
elif page == "Parameter Analysis":
    hero("Parameter Analysis", "Study the model response while changing one factor and holding the others constant.")
    if model_status():
        selected = st.selectbox("Select a parameter", FEATURES, format_func=lambda x: LABELS[x])
        st.markdown("### Reference settings for remaining factors")
        a, b, c = st.columns(3)
        other_features = [x for x in FEATURES if x != selected]
        base = DEFAULTS.copy()
        controls = {
            "Initial_pH": (a, "Initial pH", 4.0, 8.0, 0.1),
            "Temperature_C": (b, "Temperature (°C)", 25, 45, 1),
            "Agitation_rpm": (c, "Agitation (rpm)", 120, 140, 1),
            "Incubation_Time_h": (st, "Incubation time (h)", 12, 48, 1)
        }
        for feature in other_features:
            container, label, lo, hi, step = controls[feature]
            with container:
                base[feature] = st.slider(label, lo, hi, DEFAULTS[feature], step, key=f"analysis_{feature}")
        if selected == "Initial_pH":
            values = np.linspace(4, 8, 41)
        else:
            values = np.arange(int(BOUNDS[selected][0]), int(BOUNDS[selected][1]) + 1, 1)
        results = []
        try:
            for value in values:
                test_params = base.copy()
                test_params[selected] = float(value)
                results.append({"Parameter value": float(value),
                                "Predicted activity (U/mL)": predict_activity(test_params)})
            curve = pd.DataFrame(results)
            best_row = curve.loc[curve["Predicted activity (U/mL)"].idxmax()]
            st.metric("Highest estimate within displayed sweep",
                      f'{best_row["Predicted activity (U/mL)"]:.2f} U/mL',
                      f'{LABELS[selected]} = {best_row["Parameter value"]:g}')
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(curve["Parameter value"], curve["Predicted activity (U/mL)"], marker="o", linewidth=2)
            ax.axvline(DEFAULTS[selected], linestyle="--", alpha=.65, label="Reference setting")
            ax.set_xlabel(LABELS[selected])
            ax.set_ylabel("Predicted activity (U/mL)")
            ax.set_title(f"One-factor model response: {LABELS[selected]}")
            ax.grid(alpha=.25)
            ax.legend()
            st.pyplot(fig, use_container_width=True)
            with st.expander("View response table"):
                st.dataframe(curve, use_container_width=True, hide_index=True)
            make_csv_download(curve, f"FermentAI_{selected}_response.csv", "Download response curve data")
            st.caption("One-factor-at-a-time model exploration; interactions are not fully explored by this plot.")
        except Exception as exc:
            st.error(f"Could not calculate response curve: {exc}")

# ------------------------- 4. Model Performance -------------------------
elif page == "Model Performance":
    hero("Model Performance", "Inspect model properties and available dataset-level diagnostics.")
    if model_status():
        c1, c2, c3 = st.columns(3)
        with c1: kpi("Algorithm", "Random Forest", "Regression")
        with c2: kpi("Input features", str(len(get_model_features())), "Expected process variables")
        with c3: kpi("Trees", str(getattr(model, "n_estimators", "Not exposed")), "Estimator count")
        st.markdown("### Feature importance")
        importances = getattr(model, "feature_importances_", None)
        order = get_model_features()
        if importances is not None and len(importances) == len(order):
            imp = pd.DataFrame({"Feature": [LABELS.get(x, x) for x in order],
                                "Importance": np.asarray(importances, dtype=float)}).sort_values("Importance")
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.barh(imp["Feature"], imp["Importance"])
            ax.set_xlabel("Relative importance")
            ax.set_title("Random Forest feature importance")
            ax.grid(axis="x", alpha=.2)
            st.pyplot(fig, use_container_width=True)
            st.dataframe(imp.sort_values("Importance", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("Feature-importance values are not available for this model.")
        if dataset is not None:
            st.markdown("### Dataset availability")
            st.write(f"Loaded dataset: **{len(dataset)} rows** and **{len(dataset.columns)} columns**.")
            st.caption("This page does not report training or cross-validation scores as validated performance. The supplied target values are reconstructed/model-generated.")
        st.warning("Feature importance describes how the fitted model uses input variables. It does not establish biological causation or independent predictive validity.")

# ------------------------- 5. Dataset Explorer -------------------------
elif page == "Dataset Explorer":
    hero("Dataset Explorer", "Upload, inspect, filter and export fermentation datasets.")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"], key="dataset_upload")
    df = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success("Uploaded CSV loaded successfully.")
        except Exception as exc:
            st.error(f"Could not read the uploaded file: {exc}")
    elif dataset is not None:
        df = dataset.copy()
        st.caption("Showing the repository's default dataset.")
    else:
        st.info(data_error or "Upload a CSV file to begin.")
    if df is not None:
        m1, m2, m3, m4 = st.columns(4)
        with m1: kpi("Rows", f"{len(df):,}")
        with m2: kpi("Columns", f"{len(df.columns):,}")
        with m3: kpi("Missing values", f"{int(df.isna().sum().sum()):,}")
        with m4: kpi("Duplicate rows", f"{int(df.duplicated().sum()):,}")
        st.markdown("### Data preview")
        st.dataframe(df, use_container_width=True, height=330)
        numeric = df.select_dtypes(include=np.number).columns.tolist()
        if numeric:
            st.markdown("### Interactive numeric filter")
            col = st.selectbox("Column to filter", numeric)
            min_v, max_v = float(df[col].min()), float(df[col].max())
            if np.isfinite(min_v) and np.isfinite(max_v) and min_v < max_v:
                selected_range = st.slider("Select value range", min_v, max_v, (min_v, max_v), key=f"filter_{col}")
                filtered = df[df[col].between(selected_range[0], selected_range[1])]
            else:
                filtered = df.copy()
            st.caption(f"{len(filtered)} of {len(df)} rows match the filter.")
            st.dataframe(filtered, use_container_width=True, hide_index=True)
            make_csv_download(filtered, "FermentAI_filtered_dataset.csv", "Download filtered data")
            chart_col = st.selectbox("Optional distribution plot", numeric, key="distribution_col")
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.hist(df[chart_col].dropna(), bins=15)
            ax.set_xlabel(chart_col)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Distribution of {chart_col}")
            ax.grid(axis="y", alpha=.2)
            st.pyplot(fig, use_container_width=True)
        else:
            make_csv_download(df, "FermentAI_dataset.csv")

# ------------------------- 6. Optimization Lab -------------------------
elif page == "Optimization Lab":
    hero("Optimization Lab", "Compare candidate fermentation settings within the defined process-factor ranges.")
    if model_status():
        st.markdown("### Search configuration")
        left, right = st.columns(2)
        with left:
            n_candidates = st.select_slider("Number of candidate combinations", options=[50, 100, 250, 500, 1000], value=250)
            seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)
        with right:
            st.caption("Candidates are sampled within the available factor ranges. This is a computational search, not experimental optimization.")
            st.write("Search bounds")
            for feature in FEATURES:
                st.write(f"• {LABELS[feature]}: {BOUNDS[feature][0]:g} to {BOUNDS[feature][1]:g}")
        if st.button("🔬 Run candidate search", type="primary"):
            rng = np.random.default_rng(int(seed))
            candidates = pd.DataFrame({
                "Initial_pH": rng.uniform(4, 8, n_candidates),
                "Temperature_C": rng.integers(25, 46, n_candidates),
                "Agitation_rpm": rng.integers(120, 141, n_candidates),
                "Incubation_Time_h": rng.integers(12, 49, n_candidates)
            })
            try:
                order = get_model_features()
                x = candidates[order]
                candidates["Predicted_Activity_U_mL"] = np.asarray(model.predict(x), dtype=float)
                candidates = candidates.sort_values("Predicted_Activity_U_mL", ascending=False).reset_index(drop=True)
                st.session_state["optimization_results"] = candidates
                st.success(f"Evaluated {len(candidates)} candidate combinations.")
            except Exception as exc:
                st.error(f"Candidate search failed: {exc}")
        if "optimization_results" in st.session_state:
            result = st.session_state["optimization_results"]
            best = result.iloc[0]
            st.markdown("### Top-ranked model candidate")
            st.metric("Highest predicted activity", f'{best["Predicted_Activity_U_mL"]:.2f} U/mL')
            st.write("**Candidate settings**")
            cc = st.columns(4)
            for col, feature in zip(cc, FEATURES):
                with col:
                    st.metric(LABELS[feature], f'{best[feature]:.2f}' if feature == "Initial_pH" else f'{best[feature]:.0f}')
            st.markdown("### Top 10 candidates")
            st.dataframe(result.head(10), use_container_width=True, hide_index=True)
            make_csv_download(result, "FermentAI_optimization_candidates.csv", "Download all candidate results")
            st.warning("The top-ranked row is only the best among sampled model candidates. Confirm any proposed condition experimentally before making a scientific claim.")

# ------------------------- 7. Scientific Insights -------------------------
elif page == "Scientific Insights":
    hero("Scientific Insights", "Understand the biological and statistical context of the prototype.")
    st.markdown("### Fermentation factors")
    factor_info = [
        ("Initial pH", "Influences enzyme activity, nutrient availability and microbial physiology."),
        ("Temperature", "Affects microbial growth rate, metabolism and enzyme stability."),
        ("Agitation speed", "Influences mixing and oxygen transfer in submerged cultivation; excessive agitation can also introduce shear."),
        ("Incubation time", "Determines the duration available for growth and enzyme accumulation; production may rise or decline over time.")
    ]
    for title, desc in factor_info:
        with st.expander(title):
            st.write(desc)
    st.markdown("### Methods used in the application")
    st.write("**Box–Behnken design (BBD):** a response-surface design used to investigate factor effects and interactions with a structured set of experimental runs.")
    st.write("**Random Forest regression:** an ensemble learning method that averages predictions from multiple decision trees.")
    st.write("**One-factor response plot:** varies one input while holding the other inputs fixed. It is useful for visual exploration but does not represent all factor interactions.")
    st.markdown("### Scientific limitations")
    st.warning(
        "The demonstration CSV is reconstructed/model-generated rather than the original raw experimental "
        "dataset. The saved model's outputs, feature importance and candidate rankings are therefore "
        "exploratory software demonstrations. They do not establish validated enzyme yields, causal effects, "
        "or a confirmed fermentation optimum."
    )
    st.markdown("### Reference study")
    st.write("Khusro et al. (2017), “Optimization of Thermo-Alkali Stable Amylase Production and Biomass Yield from Bacillus sp. Under Submerged Cultivation,” *Fermentation*, 3(1), 7.")
    st.caption("The reference study provides background context; this app's demonstration data should not be treated as the paper's raw observations.")

# ------------------------- 8. Report Center -------------------------
elif page == "Report Center":
    hero("Report Center", "Generate a downloadable summary of model predictions and computational exploration.")
    st.markdown("### Prediction report")
    if "fermentai_prediction" in st.session_state:
        pred = st.session_state["fermentai_prediction"]
        params = st.session_state["fermentai_params"]
        st.metric("Most recent predicted activity", f"{pred:.2f} U/mL")
        report_df = pd.DataFrame([{
            "Generated_at": st.session_state.get("fermentai_prediction_time", ""),
            **params,
            "Predicted_Activity_U_mL": pred,
            "Interpretation": "Model estimate only; not experimentally validated"
        }])
        st.dataframe(report_df, use_container_width=True, hide_index=True)
        make_csv_download(report_df, "FermentAI_prediction_report.csv", "Download prediction report")
    else:
        st.info("Run a prediction in Smart Optimizer first. Your latest result will appear here.")
    st.markdown("### Optimization report")
    if "optimization_results" in st.session_state:
        opt = st.session_state["optimization_results"]
        st.write(f"Candidate combinations evaluated: **{len(opt)}**")
        st.dataframe(opt.head(10), use_container_width=True, hide_index=True)
        make_csv_download(opt, "FermentAI_optimization_report.csv", "Download optimization report")
    else:
        st.info("Run a candidate search in Optimization Lab to generate an optimization report.")
    st.markdown("### Reporting note")
    st.warning("All report outputs are computational model estimates based on the supplied reconstructed/model-generated training data. Do not describe them as experimentally confirmed results.")

st.sidebar.markdown("---")
st.sidebar.caption("FermentAI • Research prototype")
