import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ============================================================
# FermentAI
# Data-driven modelling and optimization of microbial
# alpha-amylase production using submerged fermentation.
# ============================================================

st.set_page_config(
    page_title="FermentAI | Fermentation Optimizer",
    page_icon="🧫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------- Styling ---------------------------
st.markdown("""
<style>
    .stApp { background: #F6F8FB; }
    [data-testid="stSidebar"] { background: #10243A; }
    [data-testid="stSidebar"] * { color: #F4F7FB !important; }
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; }
    h1, h2, h3 { color: #142B45; }
    .hero {
        background: linear-gradient(115deg, #10243A 0%, #176B72 100%);
        padding: 1.5rem 1.8rem; border-radius: 16px; color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1, .hero p { color: white !important; }
    .metric-card {
        background: white; border: 1px solid #E4EAF1; border-radius: 13px;
        padding: 1rem 1.1rem; min-height: 112px;
        box-shadow: 0 2px 8px rgba(20,43,69,.04);
    }
    .metric-label { color: #66788A; font-size: .86rem; }
    .metric-value { color: #142B45; font-size: 1.65rem; font-weight: 700; margin-top: .25rem; }
    .soft-note {
        background: #EAF3F4; border-left: 4px solid #1C8585;
        padding: .85rem 1rem; border-radius: 8px; color: #20384A;
    }
    .warning-note {
        background: #FFF4E5; border-left: 4px solid #D68A25;
        padding: .85rem 1rem; border-radius: 8px; color: #503A1B;
    }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #E4EAF1; padding: 12px 15px;
        border-radius: 12px;
    }
    .stButton button, .stDownloadButton button {
        border-radius: 9px; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------- Constants ---------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random_forest_model.pkl"
DATA_PATH = BASE_DIR / "BBD_29_Runs_For_Verification_FIXED.csv"

# Use floats consistently in Streamlit sliders. This prevents
# mixed int/float slider parameter errors across Streamlit versions.
FACTOR_CONFIG = {
    "Initial_pH": {
        "label": "Initial pH", "min": 4.0, "max": 8.0,
        "default": 6.0, "step": 0.1, "unit": "pH"
    },
    "Temperature": {
        "label": "Temperature", "min": 25.0, "max": 45.0,
        "default": 35.0, "step": 1.0, "unit": "°C"
    },
    "Agitation": {
        "label": "Agitation speed", "min": 120.0, "max": 140.0,
        "default": 130.0, "step": 1.0, "unit": "rpm"
    },
    "Incubation_time": {
        "label": "Incubation time", "min": 12.0, "max": 48.0,
        "default": 24.0, "step": 1.0, "unit": "hours"
    }
}
DEFAULTS = {key: cfg["default"] for key, cfg in FACTOR_CONFIG.items()}
REFERENCE_ACTIVITY = 145.32

# --------------------------- Helpers ---------------------------
@st.cache_resource
def load_model(path_string):
    return joblib.load(path_string)

@st.cache_data
def load_dataset(path_string):
    return pd.read_csv(path_string)

def show_metric(label, value):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True
    )

def resolve_model_features(model):
    """Return model input feature names in the exact training order."""
    if hasattr(model, "feature_names_in_"):
        names = list(model.feature_names_in_)
        if len(names) == 4:
            return names
    return list(FACTOR_CONFIG.keys())

def build_input(values, feature_names):
    """Build a one-row DataFrame in the model's expected feature order."""
    row = {}
    for feature in feature_names:
        if feature in values:
            row[feature] = float(values[feature])
            continue

        normalized = feature.lower().replace(" ", "").replace("_", "")
        matched = None
        for canonical in FACTOR_CONFIG:
            candidate = canonical.lower().replace("_", "")
            if normalized == candidate:
                matched = canonical
                break
        if matched is None:
            raise ValueError(
                f"Model expects an unrecognized input feature: {feature}. "
                "Check the feature columns used during model training."
            )
        row[feature] = float(values[matched])
    return pd.DataFrame([row], columns=feature_names)

def predict_activity(model, values, feature_names):
    x = build_input(values, feature_names)
    prediction = model.predict(x)
    return float(np.asarray(prediction).reshape(-1)[0])

def parameter_slider(feature, key_prefix=""):
    cfg = FACTOR_CONFIG[feature]
    # Explicit keyword arguments + all-float values avoid invalid parameter types.
    return st.slider(
        label=cfg["label"],
        min_value=float(cfg["min"]),
        max_value=float(cfg["max"]),
        value=float(cfg["default"]),
        step=float(cfg["step"]),
        key=f"{key_prefix}{feature}"
    )

def make_csv_download(df, filename, label):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )

# --------------------------- Load assets ---------------------------
model = None
dataset = None
model_error = None
data_error = None

try:
    if MODEL_PATH.exists():
        model = load_model(str(MODEL_PATH))
    else:
        model_error = f"Model file not found: {MODEL_PATH.name}"
except Exception as exc:
    model_error = f"Could not load the model: {exc}"

try:
    if DATA_PATH.exists():
        dataset = load_dataset(str(DATA_PATH))
    else:
        data_error = f"Dataset file not found: {DATA_PATH.name}"
except Exception as exc:
    data_error = f"Could not load the dataset: {exc}"

feature_names = resolve_model_features(model) if model is not None else list(FACTOR_CONFIG.keys())

# --------------------------- Sidebar ---------------------------
st.sidebar.markdown("## 🧫 FermentAI")
st.sidebar.caption("Fermentation intelligence platform")
st.sidebar.markdown("---")

pages = [
    "Overview",
    "Smart Optimizer",
    "Parameter Analysis",
    "Model Performance",
    "Dataset Explorer",
    "Optimization Lab",
    "Scientific Insights",
    "Report Center"
]
page = st.sidebar.radio("NAVIGATION", pages, index=0)
st.sidebar.markdown("---")
st.sidebar.caption("Submerged fermentation • α-Amylase")
st.sidebar.caption("Research prototype | Version 1.0")

if model_error:
    st.sidebar.error(model_error)
if data_error:
    st.sidebar.warning(data_error)

# --------------------------- Header ---------------------------
def page_header(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True
    )

def model_required():
    if model is None:
        st.error("The prediction model could not be loaded. Check that random_forest_model.pkl is in the same GitHub folder as app.py.")
        return False
    return True

# ============================================================
# 1. OVERVIEW
# ============================================================
if page == "Overview":
    page_header(
        "FermentAI",
        "Machine-learning-assisted exploration of fermentation parameters for microbial α-amylase production."
    )

    st.markdown("""
    FermentAI is a research prototype that uses a trained Random Forest regression model
    to estimate α-amylase activity from four submerged-fermentation parameters.
    """)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        show_metric("Model", "Random Forest")
    with c2:
        show_metric("Input factors", "4")
    with c3:
        show_metric("Reference design", "29 runs")
    with c4:
        show_metric("Process", "SmF")

    st.subheader("How FermentAI works")
    steps = st.columns(4)
    step_content = [
        ("01 · Input", "Enter pH, temperature, agitation and incubation time."),
        ("02 · Predict", "The trained model estimates the response."),
        ("03 · Explore", "Compare how changing one factor affects predictions."),
        ("04 · Optimize", "Screen candidate combinations within the study ranges.")
    ]
    for col, (heading, body) in zip(steps, step_content):
        with col:
            st.markdown(f"**{heading}**")
            st.write(body)

    st.markdown(
        '<div class="warning-note"><b>Scientific transparency:</b> '
        'The currently available CSV is a reconstructed/model-generated verification '
        'dataset, not the original raw experimental measurements. Therefore, model '
        'predictions and optimization rankings are exploratory estimates and must not '
        'be presented as experimentally validated results.</div>',
        unsafe_allow_html=True
    )
    st.caption(
        "Literature reference: Khusro et al. (2017), Optimization of Thermo-Alkali Stable "
        "Amylase Production and Biomass Yield from Bacillus sp. Under Submerged Cultivation, "
        "Fermentation, 3(1), 7."
    )

# ============================================================
# 2. SMART OPTIMIZER
# ============================================================
elif page == "Smart Optimizer":
    page_header("Smart Optimizer", "Estimate α-amylase activity for a selected fermentation condition.")

    if model_required():
        left, right = st.columns([1, 1.15], gap="large")
        with left:
            st.subheader("Fermentation inputs")
            values = {}
            for feature in FACTOR_CONFIG:
                values[feature] = parameter_slider(feature, "smart_")
            predict_clicked = st.button("Predict activity", type="primary", use_container_width=True)

        with right:
            st.subheader("Prediction result")
            if predict_clicked:
                try:
                    result = predict_activity(model, values, feature_names)
                    st.session_state["last_prediction"] = {
                        "values": values.copy(),
                        "activity": result,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M")
                    }
                    st.metric("Estimated α-amylase activity", f"{result:.2f} U/mL")
                    st.caption("Model estimate only — not a measured experimental result.")
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
            elif "last_prediction" in st.session_state:
                saved = st.session_state["last_prediction"]
                st.metric("Last estimated activity", f"{saved['activity']:.2f} U/mL")
                st.caption(f"Last prediction: {saved['time']}")
            else:
                st.info("Choose the fermentation parameters and click **Predict activity**.")

            st.markdown(
                '<div class="soft-note">Prediction is generated by the saved Random Forest '
                'model using the four input parameters.</div>',
                unsafe_allow_html=True
            )

# ============================================================
# 3. PARAMETER ANALYSIS
# ============================================================
elif page == "Parameter Analysis":
    page_header("Parameter Analysis", "Explore the model-estimated response as one fermentation factor changes.")

    if model_required():
        selected_feature = st.selectbox(
            "Select the parameter to analyze",
            options=list(FACTOR_CONFIG.keys()),
            format_func=lambda x: FACTOR_CONFIG[x]["label"]
        )

        st.markdown("#### Set the other three parameters")
        base = DEFAULTS.copy()
        other_features = [f for f in FACTOR_CONFIG if f != selected_feature]

        # The three controls are rendered explicitly. All values passed to
        # st.slider are floats (min/max/default/step), preventing type errors.
        control_cols = st.columns(3)
        for col, feature in zip(control_cols, other_features):
            cfg = FACTOR_CONFIG[feature]
            with col:
                base[feature] = st.slider(
                    label=cfg["label"],
                    min_value=float(cfg["min"]),
                    max_value=float(cfg["max"]),
                    value=float(cfg["default"]),
                    step=float(cfg["step"]),
                    key=f"analysis_control_{feature}"
                )

        cfg = FACTOR_CONFIG[selected_feature]
        st.markdown(f"#### Response across {cfg['label'].lower()} values")
        sweep_values = np.linspace(float(cfg["min"]), float(cfg["max"]), 41)
        results = []
        try:
            for value in sweep_values:
                trial = base.copy()
                trial[selected_feature] = float(value)
                results.append(predict_activity(model, trial, feature_names))

            chart_df = pd.DataFrame({
                cfg["label"]: sweep_values,
                "Predicted activity (U/mL)": results
            })
            st.line_chart(
                chart_df.set_index(cfg["label"]),
                use_container_width=True
            )
            best_index = int(np.argmax(results))
            m1, m2 = st.columns(2)
            m1.metric("Highest estimate in this sweep", f"{results[best_index]:.2f} U/mL")
            m2.metric(
                f"{cfg['label']} at highest estimate",
                f"{sweep_values[best_index]:.1f} {cfg['unit']}"
            )
            st.caption("This is a one-factor-at-a-time model exploration, not a new experiment.")
            with st.expander("View sweep data"):
                st.dataframe(chart_df, use_container_width=True)
                make_csv_download(chart_df, "fermentai_parameter_sweep.csv", "Download sweep data")
        except Exception as exc:
            st.error(f"Parameter analysis could not be completed: {exc}")

# ============================================================
# 4. MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":
    page_header("Model Performance", "Inspect the saved model and its available metadata.")

    if model_required():
        a, b, c = st.columns(3)
        with a:
            show_metric("Algorithm", type(model).__name__)
        with b:
            n_features = getattr(model, "n_features_in_", len(feature_names))
            show_metric("Input features", str(n_features))
        with c:
            show_metric("Training samples", "Not verified")

        st.subheader("Input features")
        st.write(", ".join(feature_names))

        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)
            st.subheader("Random Forest feature importance")
            st.bar_chart(importance.set_index("Feature"))
            st.dataframe(importance, use_container_width=True)
        else:
            st.info("Feature-importance values are not available for this model object.")

        st.markdown(
            '<div class="warning-note"><b>Important:</b> No accuracy, R², RMSE, MAE, '
            'or independent validation score is displayed unless it is calculated from '
            'a verified held-out experimental dataset. The current reconstructed dataset '
            'cannot establish experimental predictive validity.</div>',
            unsafe_allow_html=True
        )

# ============================================================
# 5. DATASET EXPLORER
# ============================================================
elif page == "Dataset Explorer":
    page_header("Dataset Explorer", "Inspect the reference dataset or upload a CSV for exploration.")

    uploaded = st.file_uploader("Upload a CSV dataset (optional)", type=["csv"])
    active_data = dataset
    if uploaded is not None:
        try:
            active_data = pd.read_csv(uploaded)
            st.success("Uploaded dataset loaded.")
        except Exception as exc:
            st.error(f"Could not read uploaded CSV: {exc}")
            active_data = None

    if active_data is None:
        st.warning("No dataset is available. Add BBD_29_Runs_For_Verification_FIXED.csv beside app.py or upload a CSV.")
    else:
        d1, d2, d3 = st.columns(3)
        with d1:
            show_metric("Rows", f"{active_data.shape[0]:,}")
        with d2:
            show_metric("Columns", str(active_data.shape[1]))
        with d3:
            show_metric("Missing cells", f"{int(active_data.isna().sum().sum())}")

        st.subheader("Dataset preview")
        st.dataframe(active_data.head(100), use_container_width=True)

        numeric_columns = active_data.select_dtypes(include=np.number).columns.tolist()
        if numeric_columns:
            selected_column = st.selectbox("Choose a numeric variable", numeric_columns)
            st.subheader(f"Distribution of {selected_column}")
            fig, ax = plt.subplots()
            ax.hist(active_data[selected_column].dropna(), bins=12, edgecolor="black")
            ax.set_xlabel(selected_column)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Distribution of {selected_column}")
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Descriptive statistics")
            st.dataframe(active_data[numeric_columns].describe().T, use_container_width=True)
        else:
            st.info("No numeric columns were found in this dataset.")

        make_csv_download(active_data, "fermentai_dataset_export.csv", "Download dataset")

        st.markdown(
            '<div class="warning-note"><b>Dataset note:</b> The project verification CSV '
            'is reconstructed/model-generated because the original 29-run raw response '
            'values were not available. Do not describe it as the original experimental dataset.</div>',
            unsafe_allow_html=True
        )

# ============================================================
# 6. OPTIMIZATION LAB
# ============================================================
elif page == "Optimization Lab":
    page_header("Optimization Lab", "Screen candidate parameter combinations using the saved model.")

    if model_required():
        st.write("Generate candidate combinations within the parameter ranges used in this project.")
        n_candidates = st.slider(
            "Number of candidate combinations",
            min_value=50, max_value=1000, value=300, step=50,
            key="optimization_candidate_count"
        )
        run_optimization = st.button("Run candidate screening", type="primary")

        if run_optimization:
            rng = np.random.default_rng(42)
            candidates = {}
            for feature, cfg in FACTOR_CONFIG.items():
                candidates[feature] = rng.uniform(
                    float(cfg["min"]), float(cfg["max"]), int(n_candidates)
                )
            candidate_df = pd.DataFrame(candidates)

            try:
                prediction_input = candidate_df[feature_names].copy()
                candidate_df["Predicted_Activity_U_mL"] = np.asarray(
                    model.predict(prediction_input)
                ).reshape(-1)
                ranked = candidate_df.sort_values(
                    "Predicted_Activity_U_mL", ascending=False
                ).reset_index(drop=True)
                ranked.insert(0, "Rank", np.arange(1, len(ranked) + 1))
                st.session_state["optimization_results"] = ranked
            except Exception as exc:
                st.error(f"Optimization screening failed: {exc}")

        if "optimization_results" in st.session_state:
            ranked = st.session_state["optimization_results"]
            st.subheader("Top 10 model-ranked candidates")
            st.dataframe(ranked.head(10), use_container_width=True)
            best = ranked.iloc[0]
            st.metric(
                "Highest predicted activity among screened candidates",
                f"{best['Predicted_Activity_U_mL']:.2f} U/mL"
            )
            make_csv_download(ranked, "fermentai_optimization_candidates.csv", "Download all ranked candidates")
            st.caption("Candidate ranking is model-based screening, not proof of a true optimum.")
        st.markdown(
            '<div class="warning-note"><b>Interpretation:</b> These combinations are '
            'computational candidates only. Confirm promising conditions experimentally '
            'before making biological or process-performance claims.</div>',
            unsafe_allow_html=True
        )

# ============================================================
# 7. SCIENTIFIC INSIGHTS
# ============================================================
elif page == "Scientific Insights":
    page_header("Scientific Insights", "Understand the fermentation system and the boundaries of the model.")

    st.subheader("What is submerged fermentation?")
    st.write(
        "Submerged fermentation (SmF) is a microbial cultivation process in which "
        "microorganisms grow in a liquid nutrient medium. Process conditions such as "
        "pH, temperature, agitation and incubation time can influence growth and enzyme production."
    )

    st.subheader("Why these four parameters?")
    explanations = [
        ("Initial pH", "Influences enzyme activity, nutrient availability and microbial physiology."),
        ("Temperature", "Affects microbial growth rate, metabolism and enzyme stability."),
        ("Agitation speed", "Influences mixing and oxygen transfer in the fermentation broth."),
        ("Incubation time", "Determines the duration available for growth and enzyme accumulation.")
    ]
    for heading, body in explanations:
        with st.expander(heading):
            st.write(body)

    st.subheader("Reference study")
    st.write(
        "Khusro et al. (2017) reported a maximum experimental α-amylase activity of "
        "145.32 U/mL at pH 6, 35°C, 130 rpm and 24 hours for the reported Bacillus sp. "
        "submerged cultivation conditions."
    )
    st.markdown("[Read the reference paper](https://www.mdpi.com/2311-5637/3/1/7)")

    st.markdown(
        '<div class="warning-note"><b>Limitations:</b> The original raw 29-run response '
        'table is unavailable in this project workspace. A reconstructed/model-generated '
        'CSV was used for the current demonstration model. Therefore, this application '
        'demonstrates a computational workflow; it does not establish experimentally '
        'validated prediction accuracy or a newly discovered optimum.</div>',
        unsafe_allow_html=True
    )

# ============================================================
# 8. REPORT CENTER
# ============================================================
elif page == "Report Center":
    page_header("Report Center", "Export prediction records and optimization screening results.")

    st.subheader("Latest prediction")
    if "last_prediction" in st.session_state:
        saved = st.session_state["last_prediction"]
        prediction_report = pd.DataFrame([{
            **saved["values"],
            "Predicted_Activity_U_mL": saved["activity"],
            "Generated_At": saved["time"],
            "Status": "Model estimate — not experimentally validated"
        }])
        st.dataframe(prediction_report, use_container_width=True)
        make_csv_download(
            prediction_report,
            "fermentai_prediction_report.csv",
            "Download prediction report"
        )
    else:
        st.info("No prediction has been generated in this session yet. Use Smart Optimizer first.")

    st.divider()
    st.subheader("Optimization screening report")
    if "optimization_results" in st.session_state:
        optimization_report = st.session_state["optimization_results"]
        st.dataframe(optimization_report.head(20), use_container_width=True)
        make_csv_download(
            optimization_report,
            "fermentai_optimization_report.csv",
            "Download optimization report"
        )
    else:
        st.info("No optimization screening has been run in this session yet.")

    st.caption(
        "Reports contain model-generated estimates. They should not be interpreted "
        "as experimental measurements."
    )

# --------------------------- Footer ---------------------------
st.markdown("---")
st.caption(
    "FermentAI • Academic research prototype • Results require experimental verification."
)
