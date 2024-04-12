import streamlit as st
import subprocess
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Install required packages
def install_packages():
    subprocess.call("pip install -r requirements.txt", shell=True)

# Function to load the pre-trained XGBoost model
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("xgb_model.json")
    return model

# Function to predict fraud
def predict_fraud(model, transaction_data):
    # Make predictions using the pre-trained model
    predictions = model.predict(transaction_data)
    return predictions

def main():
    st.title("Fraud Detection App")
    st.write("Upload your transaction data to detect fraudulent transactions.")

    # File upload widget
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Install required packages
        install_packages()

        # Load pre-trained model
        model = load_model()

        # Read the uploaded file
        transaction_data = pd.read_csv(uploaded_file)

        # Display a preview of the uploaded data
        st.subheader("Uploaded Transaction Data:")
        st.write(transaction_data.head())

        # Button to start prediction
        if st.button("Detect Fraud"):
            st.write("Performing fraud detection...")

            # Perform prediction
            predictions = predict_fraud(model, transaction_data.drop(columns=['Class'], axis=1))

            # Display the results
            st.subheader("Prediction Results:")
            st.write(predictions)

            # Display detailed metrics
            st.subheader("Detailed Metrics:")
            st.write(classification_report(transaction_data['Class'], predictions))

            # Display confusion matrix
            st.subheader("Confusion Matrix:")
            cm = confusion_matrix(transaction_data['Class'], predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot()

            # Display visualization of prediction results
            st.subheader("Visualization of Prediction Results:")
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            sns.countplot(x='Class', data=transaction_data, ax=axes[0])
            axes[0].set_title("Actual Distribution")
            axes[0].set_xlabel("Class")
            axes[0].set_ylabel("Count")

            sns.countplot(x=predictions, ax=axes[1])
            axes[1].set_title("Predicted Distribution")
            axes[1].set_xlabel("Predicted Class")
            axes[1].set_ylabel("Count")
            st.pyplot()

if __name__ == "__main__":
    main()
