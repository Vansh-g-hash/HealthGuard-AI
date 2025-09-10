import streamlit as st
import pandas as pd
import joblib
import numpy as np
from io import BytesIO

st.set_page_config(page_title="AI for Healthcare", page_icon="🩺", layout="wide")

# --- Load Model ---
try:
    pipeline = joblib.load("diabetes_model.pkl")
    scaler = pipeline["scaler"]
    model = pipeline["model"]
except Exception as e:
    st.error("⚠️ Model not found. Please run model_training.py first to create diabetes_model.pkl")
    st.stop()

# --- App Title ---
st.title("🩺 AI for Healthcare - Diabetes Prediction")
st.write("This app predicts whether a patient is likely to have diabetes, based on medical parameters.")

# -----------------------------------------------------------
# SINGLE PATIENT FORM
# -----------------------------------------------------------
st.header("🔹 Single Patient Prediction")

with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, value=1)
        glucose = st.number_input("Glucose", min_value=0.0, value=120.0)
        bp = st.number_input("Blood Pressure", min_value=0.0, value=70.0)

    with col2:
        skin = st.number_input("Skin Thickness", min_value=0.0, value=20.0)
        insulin = st.number_input("Insulin", min_value=0.0, value=79.0)
        bmi = st.number_input("BMI", min_value=0.0, value=32.0)

    with col3:
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.471)
        age = st.number_input("Age", min_value=0, value=33)

    submit_btn = st.form_submit_button("Predict")

if submit_btn:
    input_df = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }])

    X_sc = scaler.transform(input_df)
    pred = model.predict(X_sc)[0]
    prob = model.predict_proba(X_sc)[0][1] if hasattr(model, "predict_proba") else None

    st.subheader("🧾 Result:")
    st.success(f"Prediction: {'Diabetes' if pred == 1 else 'No Diabetes'}")
    if prob is not None:
        st.info(f"Probability of diabetes: {prob:.2f}")

# -----------------------------------------------------------
# BATCH PREDICTION FROM CSV
# -----------------------------------------------------------
st.header("📂 Batch Prediction (Upload CSV)")

uploaded_file = st.file_uploader("Upload a CSV file with patient data (same format as dataset, no Outcome column)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded File Preview:")
    st.dataframe(df.head())

    try:
        X_sc = scaler.transform(df)
        preds = model.predict(X_sc)
        probs = model.predict_proba(X_sc)[:, 1] if hasattr(model, "predict_proba") else [None] * len(preds)

        results = df.copy()
        results["Prediction"] = ["Diabetes" if p == 1 else "No Diabetes" for p in preds]
        results["Probability"] = probs

        st.subheader("✅ Batch Predictions")
        st.dataframe(results)

        # --- Download Results ---
        csv_buffer = BytesIO()
        results.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv_buffer.getvalue(),
            file_name="diabetes_predictions.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"⚠️ Error processing CSV: {e}")

# -----------------------------------------------------------
# MODEL INFO & FEATURE IMPORTANCE
# -----------------------------------------------------------
st.header("📈 Model Information")

try:
    accuracy = model.score(scaler.transform(df) if uploaded_file else scaler.transform(input_df), 
                           preds if uploaded_file else [pred])
    st.write(f"**Model Accuracy (on provided data):** {accuracy:.2f}")
except:
    st.write("Model accuracy available in training logs.")

if hasattr(model, "feature_importances_"):
    st.subheader("📊 Feature Importance")
    feat_imp = pd.DataFrame({
        "Feature": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
        "Importance": model.feature_importances_,
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(feat_imp.set_index("Feature"))
