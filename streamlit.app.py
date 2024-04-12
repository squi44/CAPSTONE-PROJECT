import streamlit as st
import requests
from pydantic import BaseModel

# Define input data model
class TransactionData(BaseModel):
    time: float
    amount: float
    v1: float
    v2: float
    # Add more feature variables as needed

# Function to make API request for prediction
def get_prediction(data: TransactionData):
    url = "http://localhost:8000/predict"
    response = requests.post(url, json=data.dict())
    if response.status_code == 200:
        prediction = response.json()["result"]
        return prediction
    else:
        return None

# Streamlit app
def main():
    st.title("Credit Card Fraud Detection")

    # Prediction Section
    st.sidebar.subheader("Make Prediction")
    time = st.sidebar.number_input("Time", min_value=0.0, step=1.0)
    amount = st.sidebar.number_input("Amount", min_value=0.0, step=0.01)
    v1 = st.sidebar.number_input("V1", step=0.01)
    v2 = st.sidebar.number_input("V2", step=0.01)
    # Add more input fields as needed

    if st.sidebar.button("Predict"):
        transaction_data = TransactionData(time=time, amount=amount, v1=v1, v2=v2)
        prediction = get_prediction(transaction_data)
        if prediction is not None:
            st.success(f"Prediction: {prediction}")
        else:
            st.error("Failed to fetch prediction. Please try again later.")

if __name__ == "__main__":
    main()
