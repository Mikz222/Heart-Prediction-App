import streamlit as st
import pandas as pd
import joblib

# ==========================
# Load Trained Model
# ==========================
@st.cache_resource
def load_model():
    return joblib.load("heart_nb_pipeline.pkl")  # match filename from notebook

model = load_model()

# ==========================
# UI Layout
# ==========================
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.markdown("Enter patient details below to predict the likelihood of heart disease.")

# ==========================
# Input Form
# ==========================
def user_input():
    age = st.number_input("Age", min_value=20, max_value=100, value=40)
    gender = st.selectbox("Gender", ["male", "female"])
    education = st.selectbox("Education", ["1", "2", "3", "4"])  # categorical
    currentSmoker = st.selectbox("Current Smoker", [0, 1])
    cigsPerDay = st.number_input("Cigarettes per Day", min_value=0, max_value=60, value=0)
    BPMeds = st.selectbox("On Blood Pressure Medication", [0, 1])
    prevalentStroke = st.selectbox("History of Stroke", [0, 1])
    prevalentHyp = st.selectbox("Hypertension", [0, 1])
    diabetes = st.selectbox("Diabetes", [0, 1])
    totChol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
    sysBP = st.number_input("Systolic BP", min_value=80, max_value=250, value=120)
    diaBP = st.number_input("Diastolic BP", min_value=50, max_value=150, value=80)
    BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    heartRate = st.number_input("Heart Rate", min_value=40, max_value=200, value=75)
    glucose = st.number_input("Glucose Level", min_value=40, max_value=300, value=80)

    data = {
        "age": age,
        "Gender": gender,
        "education": education,
        "currentSmoker": currentSmoker,
        "cigsPerDay": cigsPerDay,
        "BPMeds": BPMeds,
        "prevalentStroke": prevalentStroke,
        "prevalentHyp": prevalentHyp,
        "diabetes": diabetes,
        "totChol": totChol,
        "sysBP": sysBP,
        "diaBP": diaBP,
        "BMI": BMI,
        "heartRate": heartRate,
        "glucose": glucose,
    }

    return pd.DataFrame([data])

# ==========================
# Prediction
# ==========================
input_df = user_input()

if st.button("🔍 Predict"):
    proba = model.predict_proba(input_df)[:, 1][0]
    pred = model.predict(input_df)[0]

    st.subheader("Prediction Result:")
    if pred == 1:
        st.error(f"⚠️ Likely Heart Disease (Confidence: {proba*100:.2f}%)")
    else:
        st.success(f"✅ Unlikely Heart Disease (Confidence: {(1-proba)*100:.2f}%)")

    st.markdown("### 📊 Entered Patient Data")
    st.dataframe(input_df, use_container_width=True)
