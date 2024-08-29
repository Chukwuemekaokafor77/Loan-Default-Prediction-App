import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import os
import warnings

@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), 'loan_approval_dataset.csv')
    loan_data = pd.read_csv(data_path)
    
    loan_data.columns = loan_data.columns.str.strip()
    loan_data['loan_status'] = loan_data['loan_status'].str.strip().str.lower()
    loan_data = loan_data.drop(columns=['loan_id'])
    
    return loan_data

@st.cache_resource
def train_models():
    loan_data = load_data()
    loan_data['loan_status_encoded'] = loan_data['loan_status'].map({'approved': 1, 'rejected': 0})
    loan_data_encoded = loan_data.drop(columns=['loan_status'])

    X = loan_data_encoded.drop(columns=['loan_status_encoded'])
    y = loan_data_encoded['loan_status_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ])

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(random_state=42))])

    model.fit(X_train, y_train)
    return model

def predict_loan_status(model, input_data):
    input_df = pd.DataFrame([input_data])
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prediction = model.predict(input_df)
    return prediction[0]