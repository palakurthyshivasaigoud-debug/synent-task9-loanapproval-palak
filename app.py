import os

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "loan_approval_model.pkl")

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("Loan Approval Prediction")
st.write("Enter applicant and loan details to predict whether the loan is likely to be approved.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please run `python train_model.py` first.")
    st.stop()

package = load_model()
model = package["model"]
feature_columns = package["feature_columns"]
model_name = package["model_name"]

st.caption(f"Model used: {model_name}")

with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2, step=1)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        income_annum = st.number_input("Annual Income", min_value=100000, max_value=20000000, value=5000000, step=100000)
        loan_amount = st.number_input("Loan Amount", min_value=100000, max_value=50000000, value=15000000, step=100000)
        loan_term = st.number_input("Loan Term", min_value=1, max_value=30, value=10, step=1)

    with col2:
        cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, value=700)
        residential_assets_value = st.number_input("Residential Assets Value", min_value=0, max_value=30000000, value=5000000, step=100000)
        commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, max_value=30000000, value=3000000, step=100000)
        luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, max_value=50000000, value=10000000, step=100000)
        bank_asset_value = st.number_input("Bank Asset Value", min_value=0, max_value=20000000, value=4000000, step=100000)

    submitted = st.form_submit_button("Predict Loan Status")

if submitted:
    total_assets_value = (
        residential_assets_value
        + commercial_assets_value
        + luxury_assets_value
        + bank_asset_value
    )
    input_data = {
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
        "loan_to_income_ratio": loan_amount / income_annum if income_annum else 0,
        "total_assets_value": total_assets_value,
        "loan_to_asset_ratio": loan_amount / total_assets_value if total_assets_value else 0,
    }

    row = pd.DataFrame([input_data])
    row = row.reindex(columns=feature_columns)

    prediction = model.predict(row)[0]
    probability = model.predict_proba(row)[0][1] if hasattr(model, "predict_proba") else None

    if prediction == 1:
        st.success("Prediction: Loan Approved")
    else:
        st.error("Prediction: Loan Rejected")

    if probability is not None:
        st.metric("Approval Probability", f"{probability * 100:.1f}%")

    st.subheader("Submitted Details")
    st.dataframe(row, use_container_width=True)
