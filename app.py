# import the used libraries

import streamlit as st
import pandas as pd
import joblib

# page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)
# side bar
with st.sidebar:
    st.header("About")
    st.write(
        """
        This application predicts whether a bank customer
        is likely to churn using a Random Forest Machine Learning model.
        """
    )

# Load the saved model
model = joblib.load('bank_churn_pipeline.pkl')
# Build the web interface
# interface title

st.title( " Customer Churn Prediction Dashboard")
st.write(" Enter the customer's information below and click Generate Model Prediction to get an instant prediction")
# Create the interface

# Arrange layout into 2 columns
col1, col2 = st.columns(2)
with col1:
    credit_score = st.number_input(
        "Credit Score:", min_value=300, max_value=850, value=650
    )
    geography = st.selectbox("Geography / Country:", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender:", ["Male", "Female"])
    age = st.number_input("Age (Years):", min_value=18, max_value=200, value=35)
    tenure = st.slider("Tenure (Years with Bank):", min_value=0, max_value=10, value=5)
with col2:
    balance = st.number_input("Account Balance ($):", min_value=0.0, value=50000.0)
    num_products = st.selectbox("Number of Products Used:", [1, 2, 3, 4], index=0)
    has_cr_card = st.checkbox("Customer Has Credit Card?", value=True)
    is_active = st.checkbox("Is Customer an Active Member?", value=True)
    salary = st.number_input("Estimated Annual Salary ($):", min_value=0.0, value=75000.0)

    # create the execution button
if st.button("Generate Model Prediction"):
    # Convert checkbox boolean values to binary 1s and 0s matching int64 formats
    has_card_int = 1 if has_cr_card else 0
    active_int = 1 if is_active else 0
    input_df = pd.DataFrame(
        [
            [
                credit_score,
                geography,
                gender,
                age,
                tenure,
                balance,
                num_products,
                has_card_int,
                active_int,
                salary,
            ]
        ],
        columns=[
            "CreditScore",
            "Geography",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
        ],
    )

# Process through pipeline automatically (handles OneHotEncoding and Prediction internally)
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

# Render results on screen
st.markdown("---")
if prediction == 1:
        st.error(
            f"**High Risk Alert**: Customer predicted to churn. (Probability: {probability:.2%})"
        )
else:
        st.success(
            f"**Low Risk**: Customer predicted to stay active. (Retention Probability: {1 - probability:.2%})"
        )

# footer
st.markdown("---")
st.caption("Developed by Mize Khalfan | Customer Churn Prediction using Machine Learning")