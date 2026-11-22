import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="FermentAI",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #07111f;
    color: white;
}
[data-testid="stMetric"] {
    background: #10243a;
    padding: 18px;
    border-radius: 14px;
}
h1, h2, h3 {
    color: #55e6d0 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🧬 FermentAI")
st.subheader("Intelligent Fermentation. Data-Driven Optimization.")
st.write("Machine learning based exploration of microbial α-amylase production.")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module",
    ["Overview", "Fermentation Optimizer", "Parameter Analysis",
     "ML Insights", "Dataset Explorer", "Knowledge Center"]
)

# Reference conditions
reference_activity = 145.32
optimal_ph = 6
optimal_temp = 35
optimal_agitation = 130
optimal_time = 24

# Simple deterministic response approximation
def predict_activity(ph, temp, agitation, time):
    score = (
        ((ph - optimal_ph) / 2) ** 2
        + ((temp - optimal_temp) / 10) ** 2
        + ((agitation - optimal_agitation) / 10) ** 2
        + ((time - optimal_time) / 24) ** 2
    )
    prediction = reference_activity * np.exp(-0.35 * score)
    return float(prediction)

if page == "Overview":
    st.header("Fermentation Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Experimental Reference", "145.32 U/mL")
    c2.metric("Experimental Runs", "29")
    c3.metric("ML Model", "Random Forest")
    c4.metric("Process", "Submerged Fermentation")

    st.markdown("---")
    st.markdown("### Project Information")

    st.write("""
    **Microorganism:** Bacillus sp.  
    **Product:** α-Amylase  
    **Method:** Submerged fermentation  
    **Optimization:** Box–Behnken Design  
    **Approach:** Machine Learning-based prediction
    """)

    st.info("Predictions are exploratory and are not experimentally validated.")

elif page == "Fermentation Optimizer":
    st.header("🧪 Fermentation Parameter Optimizer")

    ph = st.slider("Initial pH", 4.0, 8.0, 6.0, 0.1)
    temp = st.slider("Temperature (°C)", 25, 45, 35)
    agitation = st.slider("Agitation (rpm)", 120, 140, 130)
    time = st.slider("Incubation time (hours)", 12, 48, 24)

    prediction = predict_activity(ph, temp, agitation, time)

    st.markdown("---")
    st.metric("Estimated α-Amylase Activity", f"{prediction:.2f} U/mL")

    st.write("### Selected Fermentation Conditions")
    st.write(f"pH: **{ph}**")
    st.write(f"Temperature: **{temp} °C**")
    st.write(f"Agitation: **{agitation} rpm**")
    st.write(f"Incubation: **{time} hours**")

    st.progress(min(prediction / reference_activity, 1.0))

    st.warning("This is a mathematical demonstration predictor, not your trained Random Forest model.")

elif page == "Parameter Analysis":
    st.header("📊 Parameter Analysis")

    selected = st.selectbox(
        "Analyze parameter",
        ["Temperature", "pH", "Agitation", "Incubation Time"]
    )

    values = {
        "Temperature": np.arange(25, 46, 1),
        "pH": np.arange(4, 8.1, 0.2),
        "Agitation": np.arange(120, 141, 1),
        "Incubation Time": np.arange(12, 49, 1)
    }

    activities = []

    for value in values[selected]:
        if selected == "Temperature":
            activities.append(predict_activity(6, value, 130, 24))
        elif selected == "pH":
            activities.append(predict_activity(value, 35, 130, 24))
        elif selected == "Agitation":
            activities.append(predict_activity(6, 35, value, 24))
        else:
            activities.append(predict_activity(6, 35, 130, value))

    fig, ax = plt.subplots()
    ax.plot(values[selected], activities, marker="o")
    ax.set_xlabel(selected)
    ax.set_ylabel("Estimated activity (U/mL)")
    ax.set_title(f"Effect of {selected}")
    ax.grid(True)

    st.pyplot(fig)

elif page == "ML Insights":
    st.header("🤖 Machine Learning Insights")

    st.subheader("Grouped Cross-Validation Results")

    metrics = pd.DataFrame({
        "Model": ["Random Forest", "Ridge Regression"],
        "R²": [-0.0399, -0.3493],
        "MAE": [17.91, 22.20],
        "RMSE": [24.34, 27.72]
    })

    st.dataframe(metrics, use_container_width=True)

    st.warning("""
    Negative R² values indicate poor generalization on unseen experimental groups.
    The model is exploratory and needs additional experimental validation.
    """)

    st.write("### Model Limitations")
    st.write("""
    - Small dataset (29 runs)
    - Limited parameter ranges
    - No independent experimental validation
    - Predictions may not generalize to new fermentation conditions
    """)

elif page == "Dataset Explorer":
    st.header("📁 Dataset Explorer")

    uploaded = st.file_uploader("Upload fermentation CSV", type=["csv"])

    if uploaded:
        data = pd.read_csv(uploaded)
        st.success(f"Dataset loaded: {data.shape[0]} rows")
        st.dataframe(data, use_container_width=True)

        st.download_button(
            "Download dataset",
            data=data.to_csv(index=False),
            file_name="fermentation_dataset.csv",
            mime="text/csv"
        )
    else:
        st.info("Upload a CSV file to explore your fermentation dataset.")

elif page == "Knowledge Center":
    st.header("📚 Fermentation Knowledge Center")

    with st.expander("What is α-Amylase?"):
        st.write("α-Amylase is an enzyme that hydrolyses internal α-1,4 glycosidic bonds in starch.")

    with st.expander("What is submerged fermentation?"):
        st.write("Submerged fermentation is microbial cultivation in a liquid nutrient medium.")

    with st.expander("What is Box–Behnken Design?"):
        st.write("Box–Behnken Design is a response surface experimental design used to evaluate parameter interactions and optimize processes.")

    with st.expander("Why Machine Learning?"):
        st.write("Machine learning can help identify patterns between fermentation parameters and product yield.")
