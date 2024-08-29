import streamlit as st
from pages import home, data_analysis, predictions

st.set_page_config(page_title="Loan Default Prediction App", layout="wide")

PAGES = {
    "Home": home,
    "Data Analysis": data_analysis,
    "Predictions": predictions
}

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]
    page.app()

if __name__ == "__main__":
    main()