import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data

def app():
    st.title("Data Analysis")

    loan_data = load_data()

    st.write("### Dataset Preview")
    st.write(loan_data.head())

    st.write("### Missing Values")
    st.write(loan_data.isnull().sum())

    st.write("### Data Exploration")
    numerical_columns = loan_data.select_dtypes(include=['int64', 'float64']).columns
    st.write(loan_data[numerical_columns].describe())

    st.write("### Correlation Heatmap (Numerical Features)")
    plt.figure(figsize=(12, 10))
    sns.heatmap(loan_data[numerical_columns].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap (Numerical Features)')
    st.pyplot(plt)

    st.write("### Categorical Columns")
    categorical_columns = loan_data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        st.write(f"#### {col}")
        st.write(loan_data[col].value_counts())