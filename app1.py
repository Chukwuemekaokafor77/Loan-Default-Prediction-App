import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os
import warnings

# Set page configuration
st.set_page_config(page_title="Loan Default Prediction App", layout="wide")

# Load the dataset using a relative path
@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), 'loan_approval_dataset.csv')
    loan_data = pd.read_csv(data_path)
    
    # Trim whitespace from column names
    loan_data.columns = loan_data.columns.str.strip()
    
    # Ensure consistent formatting for the loan_status column
    loan_data['loan_status'] = loan_data['loan_status'].str.strip().str.lower()
    
    # Drop the 'loan_id' column as it is not a feature
    loan_data = loan_data.drop(columns=['loan_id'])
    
    return loan_data

loan_data = load_data()

# Display dataset information
st.write("### Dataset Preview")
st.write(loan_data.head())

# Check for missing values
st.write("### Missing Values")
st.write(loan_data.isnull().sum())

# Print out all column names for verification
st.write("### Column Names in Dataset")
st.write(loan_data.columns.tolist())

# Separate numerical and categorical columns
numerical_columns = loan_data.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = loan_data.select_dtypes(include=['object']).columns

# Data Exploration
st.write("### Data Exploration")
st.write(loan_data[numerical_columns].describe())

# Correlation Heatmap (only for numerical columns)
st.write("### Correlation Heatmap (Numerical Features)")
plt.figure(figsize=(12, 10))
sns.heatmap(loan_data[numerical_columns].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap (Numerical Features)')
st.pyplot(plt)

# Display information about categorical columns
st.write("### Categorical Columns")
for col in categorical_columns:
    st.write(f"#### {col}")
    st.write(loan_data[col].value_counts())

# Manually encode the 'loan_status' column with standardized values
loan_data['loan_status_encoded'] = loan_data['loan_status'].map({'approved': 1, 'rejected': 0})

# Drop the original loan_status column
loan_data_encoded = loan_data.drop(columns=['loan_status'])

# Check for NaN values in the target variable
st.write("### Checking for NaN values in 'loan_status_encoded'")
nan_count = loan_data_encoded['loan_status_encoded'].isna().sum()
st.write(f"NaN values in 'loan_status_encoded': {nan_count}")

# Drop rows where the target variable is NaN
loan_data_encoded.dropna(subset=['loan_status_encoded'], inplace=True)

# Separate features and target
X = loan_data_encoded.drop(columns=['loan_status_encoded'])
y = loan_data_encoded['loan_status_encoded']

# Split the data into training, validation, and test sets (70%, 15%, 15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ])

# Define the models
logistic_model = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', LogisticRegression(random_state=42))])

random_forest_model = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', RandomForestClassifier(random_state=42))])

# Train the models
@st.cache_resource
def train_models():
    logistic_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)
    return logistic_model, random_forest_model

logistic_model, random_forest_model = train_models()

# Cross-validation to select the best model
logistic_scores = cross_val_score(logistic_model, X_train, y_train, cv=5, scoring='accuracy')
random_forest_scores = cross_val_score(random_forest_model, X_train, y_train, cv=5, scoring='accuracy')

best_model = logistic_model if logistic_scores.mean() > random_forest_scores.mean() else random_forest_model

# Model evaluation on the validation set
val_predictions = best_model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)
st.write(f"### Best Model Accuracy on Validation Set: {val_accuracy:.4f}")

# Model evaluation on the test set
test_predictions = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
st.write(f"### Best Model Accuracy on Test Set: {test_accuracy:.4f}")

# Confusion matrix on the test set
conf_matrix = confusion_matrix(y_test, test_predictions)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', annot_kws={"fontsize":12})
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix - Best Model')
st.pyplot(plt)

# Display classification report
st.write("### Model Performance Metrics")
st.text(classification_report(y_test, test_predictions))

# Feature importance (only for Random Forest)
if isinstance(best_model[-1], RandomForestClassifier):
    importances = best_model[-1].feature_importances_
    feature_names = best_model[:-1].get_feature_names_out()
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    st.pyplot(plt)

# Streamlit app for loan status prediction
def predict_loan_status(input_data):
    # Convert input data to DataFrame to match the expected format
    input_df = pd.DataFrame([input_data])
    
    # Make prediction using the best model (which includes the preprocessor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prediction = best_model.predict(input_df)
    return prediction[0]

def main():
    st.title("Loan Default Prediction App")

    # User input form
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

    # Prepare the input data
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
    
    # Predict and display the result
    if st.button("Predict Loan Status"):
        result = predict_loan_status(input_data)
        if result == 1:
            st.success("Congratulations! The loan is likely to be approved.")
        else:
            st.error("Sorry! The loan is likely to be rejected.")

if __name__ == "__main__":
    main()