# Loan Default Prediction App

## Overview

The Loan Default Prediction App is a machine learning application designed to predict the likelihood of loan approval based on various applicant characteristics. Built with Streamlit, this application provides an intuitive user interface for financial institutions to evaluate loan applications effectively.

## Features

- **Data Analysis**: Explore key statistics and visualizations of the loan dataset, including missing values and correlations.
- **Predictive Modeling**: Utilizes Logistic Regression and Random Forest algorithms to predict loan outcomes.
- **User-Friendly Interface**: Input applicant details and receive real-time predictions on loan approval.
- **Modular Structure**: Organized into separate modules for easy maintenance and future enhancements.
- **Containerization**: Built with Docker for consistent deployment across different environments.

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Docker

## Installation

To run this application locally, follow these steps:

1. Clone the repository:
   git clone https://github.com/Chukwuemekaokafor77/Loan-Default-Prediction-App.git
   cd Loan-Default-Prediction-App
Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the required packages:
pip install -r requirements.txt

Run the application:
streamlit run app.py

Docker
To run the application in a Docker container, follow these steps:
Build the Docker image:
docker build -t loan-default-prediction-app .

Run the Docker container:
docker run -p 8501:8501 loan-default-prediction-app

Access the application in your web browser at http://localhost:8501.
Contributing
Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments
Special thanks to Professor Ran Faldesh and Professor David Espinosa, both of Conestoga College Waterloo campus for your guidance, and the open-source community for your support and resources.



