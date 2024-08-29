import streamlit as st
import pandas as pd
from utils import load_data, train_models, predict_loan_status

def app():
    st.title("Loan Default Prediction")

    loan_data = load_data()
    best_model = train_models()

    st.write("### Enter Applicant Details:")
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
    income_annum = st.number_input("Annual Income (in INR)", min_value=0.0, step=100000.0)
    loan_amount = st.number_input("Loan Amount (in INR)", min_value=0.0, step=100000.0)
    loan_term = st.number_input("Loan Term (in months)", min_value=1, step=1)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, step=1)
    residential_assets_value = st.number_input("Residential Assets Value (in INR)", min_value=0.0, step=100000.0)
    commercial_assets_value = st.number_input("Commercial Assets Value (in INR)", min_value=0.0, step=100000.0)
    luxury_assets_value = st.number_input("Luxury Assets Value (in INR)", min_value=0.0, step=100000.0)
    bank_asset_value = st.number_input("Bank Asset Value (in INR)", min_value=0.0, step=100000.0)
    education = st.selectbox("Education Level", loan_data['education'].unique())
    self_employed = st.selectbox("Self Employed", loan_data['self_employed'].unique())

    input_data = {
        'no_of_dependents': no_of_dependents,
        'income_annum': income_annum,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'cibil_score': cibil_score,
        'residential_assets_value': residential_assets_value,
        'commercial_assets_value': commercial_assets_value,
        'luxury_assets_value': luxury_assets_value,
        'bank_asset_value': bank_asset_value,
        'education': education,
        'self_employed': self_employed
    }

    if st.button("Predict Loan Status"):
        result = predict_loan_status(best_model, input_data)
        if result == 1:
            st.success("Congratulations! The loan is likely to be approved.")
        else:
            st.error("Sorry! The loan is likely to be rejected.")