import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="FermentAI | Process Intelligence", page_icon="🧬", layout="wide")

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "random_forest_model.pkl"
DATA_PATH = BASE / "BBD_29_Runs_For_Verification_FIXED.csv"

FEATURES = ["Initial_pH", "Temperature_C", "Agitation_rpm", "Incubation_Time_h"]
TARGET = "Amylase_Activity"

# ---------- visual system ----------
st.markdown("""
<style>
:root { --ink:#142b3b; --muted:#637587; --teal:#087f78; --mint:#e7f6f3; --line:#e3eaf0; }
.stApp { background:#f6f9fb; color:var(--ink); }
[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #e5edf2; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color:#123449; }
.block-container { padding-top:1.8rem; padding-bottom:2.5rem; max-width:1400px; }
h1,h2,h3 { color:#142b3b !important; letter-spacing:-.02em; }
p, label, li { color:#354c5e; }
div[data-testid="stMetric"] { background:white; border:1px solid var(--line); padding:16px 18px; border-radius:14px; box-shadow:0 2px 8px rgba(22,45,64,.035); }
div[data-testid="stMetricLabel"] { color:#627789; }
div[data-testid="stMetricValue"] { color:#087f78; }
.stButton>button, .stDownloadButton>button { border-radius:10px; border:1px solid #087f78; background:#087f78; color:white; font-weight:600; }
.stButton>button:hover, .stDownloadButton>button:hover { border-color:#06645f; background:#06645f; color:white; }
div[data-testid="stAlert"] { border-radius:12px; }
.hero { background:linear-gradient(110deg,#123449,#087f78); border-radius:20px; padding:28px 32px; color:#fff; margin-bottom:20px; }
.hero h1 { color:white !important; font-size:2.25rem; margin:0; }
.hero p { color:#e5f6f3; font-size:1.05rem; margin:.5rem 0 0; }
.section-card { background:white; border:1px solid var(--line); border-radius:16px; padding:18px 20px; }
.small-note { color:#667b8c; font-size:.88rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)

try:
    model = load_model()
except Exception as exc:
    model = None
    model_error = str(exc)
else:
    model_error = ""

data = load_data()
if model is None:
    st.error("The saved Random Forest model could not be loaded. Confirm random_forest_model.pkl is in the repository.")
    st.code(model_error)
    st.stop()
if data is None:
    st.error("Dataset file not found. Confirm BBD_29_Runs_For_Verification_FIXED.csv is in the repository.")
    st.stop()

# feature names guard: preserve trained model's feature order if present
model_features = list(getattr(model, "feature_names_in_", FEATURES))
if set(model_features) != set(FEATURES):
    st.error(f"Model input features do not match this app. Found: {model_features}")
    st.stop()

def predict_one(ph, temp, agitation, time_h):
    row = pd.DataFrame([[ph, temp, agitation, time_h]], columns=FEATURES)
    row = row[model_features]
    return float(model.predict(row)[0])

def plot_style(ax):
    ax.set_facecolor("#ffffff")
    ax.grid(axis="y", alpha=.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#dce5eb")
    ax.spines["bottom"].set_color("#dce5eb")
    ax.tick_params(colors="#5d7182")
    ax.xaxis.label.set_color("#40576a")
    ax.yaxis.label.set_color("#40576a")

# sidebar
st.sidebar.markdown("## 🧬 FermentAI")
st.sidebar.caption("MICROBIAL PROCESS INTELLIGENCE")
page = st.sidebar.radio("WORKSPACE", ["Overview", "Fermentation Optimizer", "Parameter Analysis", "ML Insights", "Dataset Explorer", "Knowledge Center"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Process**  Bacillus sp.")
st.sidebar.markdown("**Product**  α-Amylase")
st.sidebar.markdown("**Mode**  Submerged fermentation")
st.sidebar.caption("Prototype • model-generated dataset")

# header
st.markdown('<div class="hero"><h1>FermentAI</h1><p>Machine-learning exploration of microbial α-amylase production under submerged fermentation.</p></div>', unsafe_allow_html=True)
st.caption("Research prototype • predictions are model-based and require experimental validation")

if page == "Overview":
    st.title("Process overview")
    st.write("Explore fermentation conditions, inspect the dataset and estimate α-amylase activity using the saved Random Forest model.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Literature reference", "145.32 U/mL")
    c2.metric("Design runs", f"{len(data)}")
    c3.metric("Predictor", "Random Forest")
    c4.metric("Process", "Submerged")

    left,right = st.columns([1.25,1])
    with left:
        st.subheader("Activity distribution in the current dataset")
        fig,ax=plt.subplots(figsize=(7,3.4))
        ax.hist(data[TARGET].dropna(), bins=10, edgecolor="white")
        ax.set_xlabel("Amylase activity (U/mL)")
        ax.set_ylabel("Run count")
        plot_style(ax)
        st.pyplot(fig, use_container_width=True)
    with right:
        st.subheader("Process conditions")
        st.markdown("""<div class="section-card">
        <b>Organism</b>: Bacillus sp.<br><br>
        <b>Product</b>: α-Amylase<br><br>
        <b>Fermentation</b>: Submerged cultivation<br><br>
        <b>Experimental design</b>: Box–Behnken Design<br><br>
        <b>Inputs</b>: pH, temperature, agitation and incubation time
        </div>""", unsafe_allow_html=True)
    st.warning("Dataset integrity notice: the 29 activity values in the uploaded dataset are model-generated/reconstructed, not original measured experimental results. Model outputs are demonstrations—not validated experimental recommendations.")

elif page == "Fermentation Optimizer":
    st.title("Fermentation optimizer")
    st.write("Set process conditions within the study ranges. The saved Random Forest model returns an estimated activity.")
    a,b = st.columns(2)
    with a:
        ph = st.slider("Initial pH", 4.0, 8.0, 6.0, 0.1)
        temp = st.slider("Temperature (°C)", 25, 45, 35, 1)
    with b:
        agitation = st.slider("Agitation (rpm)", 120, 140, 130, 1)
        time_h = st.slider("Incubation (hours)", 12, 48, 24, 1)
    pred = predict_one(ph,temp,agitation,time_h)
    st.markdown("---")
    m1,m2 = st.columns([1,2])
    m1.metric("Predicted activity", f"{pred:,.2f} U/mL")
    with m2:
        st.subheader("Selected conditions")
        st.write(f"**pH {ph:.1f}** · **{temp} °C** · **{agitation} rpm** · **{time_h} h**")
        st.progress(max(0.0,min(pred/145.32,1.0)))
        st.caption("Progress bar is scaled to the 145.32 U/mL literature reference; it is not a probability or confidence score.")
    st.warning("This prediction is generated by the saved Random Forest trained on reconstructed/model-generated activity values. It is not experimentally validated.")
    st.download_button("Download this prediction", pd.DataFrame([{"Initial_pH":ph,"Temperature_C":temp,"Agitation_rpm":agitation,"Incubation_Time_h":time_h,"Predicted_Amylase_Activity_U_mL":pred}]).to_csv(index=False), "fermentai_prediction.csv", "text/csv")

elif page == "Parameter Analysis":
    st.title("Parameter response explorer")
    st.write("One-factor-at-a-time model predictions. Other conditions are held at the reference setting.")
    selected = st.selectbox("Parameter to explore", ["Initial pH","Temperature_C","Agitation_rpm","Incubation_Time_h"])
    ranges = {"Initial_pH":np.linspace(4,8,41),"Temperature_C":np.arange(25,46),"Agitation_rpm":np.arange(120,141),"Incubation_Time_h":np.arange(12,49)}
    base = {"Initial_pH":6.0,"Temperature_C":35,"Agitation_rpm":130,"Incubation_Time_h":24}
    rows=[]
    for val in ranges[selected]:
        params=base.copy(); params[selected]=val
        rows.append({"x":val,"pred":predict_one(params["Initial_pH"],params["Temperature_C"],params["Agitation_rpm"],params["Incubation_Time_h"])})
    curve=pd.DataFrame(rows)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(curve.x,curve.pred,marker="o",markersize=3)
    ax.set_xlabel(selected); ax.set_ylabel("Predicted activity (U/mL)")
    plot_style(ax)
    st.pyplot(fig,use_container_width=True)
    st.dataframe(curve.rename(columns={"x":selected,"pred":"Predicted activity (U/mL)"}), use_container_width=True, hide_index=True)
    st.download_button("Export curve data", curve.to_csv(index=False), f"{selected}_response_curve.csv", "text/csv")
    st.info("These are model-generated response curves, not experimental response surfaces.")

elif page == "ML Insights":
    st.title("Model insights")
    st.write("A transparent look at the model and the data available in this prototype.")
    st.metric("Estimator", type(model).__name__)
    st.metric("Model training rows represented in dataset", len(data))
    if hasattr(model,"feature_importances_"):
        fi=pd.DataFrame({"Parameter":list(getattr(model, "feature_names_in_", FEATURES)),"Importance":model.feature_importances_})
        fi=fi.sort_values("Importance",ascending=True)
        st.subheader("Feature importance")
        fig,ax=plt.subplots(figsize=(7,3.6))
        ax.barh(fi["Parameter"],fi["Importance"])
        ax.set_xlabel("Relative importance")
        plot_style(ax)
        st.pyplot(fig,use_container_width=True)
        st.dataframe(fi.sort_values("Importance",ascending=False),hide_index=True,use_container_width=True)
    st.subheader("Limitations")
    st.markdown("""
    - Only 29 design rows are available.
    - Activity values are reconstructed/model-generated rather than authentic measured outcomes.
    - This makes model performance metrics unsuitable as evidence of real predictive validity.
    - No independent experimental validation has been performed.
    - Predictions should not be interpreted as confirmed optimum conditions.
    """)
    st.warning("Current dataset provenance limits scientific conclusions. Experimental measurements are needed before claiming validated optimization.")

elif page == "Dataset Explorer":
    st.title("Dataset explorer")
    st.write("Review the bundled 29-run table or upload another CSV to inspect.")
    up=st.file_uploader("Optional: upload a CSV for inspection",type=["csv"])
    view=pd.read_csv(up) if up is not None else data.copy()
    st.caption(f"{view.shape[0]} rows · {view.shape[1]} columns")
    if TARGET in view.columns:
        lo,hi=float(view[TARGET].min()),float(view[TARGET].max())
        if lo < hi:
            filt=st.slider("Activity filter (U/mL)",lo,hi,(lo,hi))
            view=view[view[TARGET].between(filt[0],filt[1])]
    st.dataframe(view,use_container_width=True,hide_index=True)
    st.download_button("Download displayed data",view.to_csv(index=False),"fermentation_dataset.csv","text/csv")

elif page == "Knowledge Center":
    st.title("Knowledge center")
    with st.expander("What is α-amylase?"):
        st.write("α-Amylase is an enzyme that cleaves internal α-1,4 glycosidic bonds in starch, producing shorter carbohydrate chains.")
    with st.expander("What is submerged fermentation?"):
        st.write("Submerged fermentation cultivates microorganisms in a liquid nutrient medium with controlled process conditions such as temperature, pH, agitation and time.")
    with st.expander("What is Box–Behnken Design?"):
        st.write("Box–Behnken Design is a response-surface experimental design that samples combinations of factor levels to study process responses while avoiding corner combinations.")
    with st.expander("Why use machine learning?"):
        st.write("Machine-learning models can explore relationships between process parameters and measured outcomes. Their reliability depends on representative, authentic training data and independent validation.")
    with st.expander("FermentAI workflow"):
        st.markdown("**Input process factors → Dataset → Random Forest prediction → Candidate condition exploration → Experimental validation**")
    st.info("The bundled dataset is reconstructed/model-generated. The scientific workflow is a prototype for demonstrating data exploration and model-based prediction.")
