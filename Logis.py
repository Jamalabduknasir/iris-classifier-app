# import lbraries
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page Config
st.set_page_config(page_title = "Iris Classifier", page_icon = "🌸", layout = "wide")

# Model Loading
@st.cache_resource
def load_model():
    with open("model.pkl",'rb') as f:
        model = pickle.load(f)
        return model

model = load_model()
SPECIES_MAP = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}
SPECIES_INFO = {
    "Setosa": "Small petals, easily distinguishable from the other two species.",
    "Versicolor": "Medium-sized petals, with more variation in color.",
    "Virginica": "The largest petals among the three species in this dataset."
}
# SIDEBAR — Inputs (Day 3 concepts)
st.sidebar.title("Flower measurements")
st.sidebar.write("Set the flower measurements using the sliders below")

sepal_length = st.sidebar.slider("Sepal Length (cm):", 4.0, 8.0, 5.8)
sepal_width = st.sidebar.slider("Sepal Width (cm):", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm):", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width (cm):", 0.1, 2.5, 1.2)
 
predict = st.sidebar.button("🔍 Predict Karein", use_container_width=True)
 
st.sidebar.markdown("---")
st.sidebar.markdown("**Developer:** Jamal Abdul Nasir")
st.sidebar.markdown("**Built with:** Streamlit + Scikit-learn ")

# MAIN TITLE
st.markdown("<h1 style = 'text-align:center;color:dark;'>🌸 Iris Flower Species Classifier", unsafe_allow_html = True)
st.markdown("<h1 style = 'text-align:center;color:dark;'>This app uses a Logistic Regression model to predict a flower's species based on its measurements",unsafe_allow_html = True)
st.markdown("---")
# Tabs
tab1,tab2,tab3 = st.tabs(["🔮 Prediction", "📊 DATA Input", "ℹ️ About This Project"])

with tab1:
    st.markdown("<h1 style = 'text-align:center;color:dark;'>Prediction Results", unsafe_allow_html = True)

    if predict:
        # Build the feature array in the exact order the model expects
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]
 
        species_name = SPECIES_MAP[prediction]
 
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Predicted Species", value=species_name)
        with col2:
            st.metric(label="Confidence", value=f"{max(prediction_proba)*100:.1f}%")
 
        with st.expander(f"ℹ️ About {species_name}"):
            st.write(SPECIES_INFO[species_name])
 
        st.subheader("Probability for Each Species")
        proba_df = pd.DataFrame({
            "Species": [SPECIES_MAP[i] for i in range(3)],
            "Probability": prediction_proba
        })
        st.bar_chart(proba_df.set_index("Species"))
 
    else:
        st.info("Set the measurements in the sidebar and click '🔍 Predict'.")

# Tabs 2 input 
with tab2:
    st.markdown("<h1 style = 'text-align:center;color:dark;'>Input Values", unsafe_allow_html = True)
    input_df = pd.DataFrame({
        "Measurement": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
        "Value (cm)": [sepal_length, sepal_width, petal_length, petal_width]
    })
 
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(input_df, use_container_width=True)
    with col2:
        st.bar_chart(input_df.set_index("Measurement"))

# About tabs
with tab3:
    st.markdown("<h1 style = 'text-align:center;color:dark;'>About This Project", unsafe_allow_html = True)
 
    with st.container():
        st.markdown("""
        This dashboard was built while learning Streamlit, using the following concepts:
 
        **Machine Learning:**
        - Logistic Regression model (trained on the Iris dataset)
        - Loading the model with `pickle`
        - Using `predict()` and `predict_proba()` for prediction and confidence
 
        **Streamlit Concepts:**
        - `st.sidebar`, `st.tabs()`, `st.columns()`, `st.expander()`
        - `st.slider()`, `st.button()`, `st.metric()`
        - `st.bar_chart()`, `st.dataframe()`
        - `@st.cache_resource` for model caching
        """)
 
    st.success("This project combines Machine Learning and Streamlit!")