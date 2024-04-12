import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load the pre-trained XGBoost model
model = xgb.XGBClassifier()
model.load_model("xgb_model.json")

# Sample dataset (replace this with your actual dataset)
# For demonstration purposes, I'll create a simple DataFrame
data = {
    'Time': [0, 1, 2, 3, 4],
    'Amount': [10.0, 20.0, 30.0, 40.0, 50.0],
    'Location': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami'],
    'Class': [0, 1, 0, 0, 1]  # Dummy class labels (0: Non-fraud, 1: Fraud)
}
transaction_data = pd.DataFrame(data)

def preprocess_data(transaction_data):
    # Perform preprocessing such as scaling
    scaler = StandardScaler()
    scaled_amount = scaler.fit_transform(transaction_data[['Amount']])
    transaction_data['Scaled_Amount'] = scaled_amount
    return transaction_data

def predict_fraud(transaction_data):
    # Perform preprocessing
    preprocessed_data = preprocess_data(transaction_data)
    # Make predictions using the pre-trained model
    predictions = model.predict(preprocessed_data.drop(columns=['Class'], axis=1))
    return predictions

def main():
    st.title("Fraud Detection App")
    st.write("Welcome to the Fraud Detection App!")

    # Sidebar for user authentication
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username == "admin" and password == "password":
            st.sidebar.success("Logged in as Admin")
        else:
            st.sidebar.error("Invalid Username or Password")

    # File upload widget
    uploaded_file = st.file_uploader("Upload your transaction data (CSV file)", type="csv")

    if uploaded_file is not None:
        # Read the uploaded file
        transaction_data = pd.read_csv(uploaded_file)

        # Display a preview of the uploaded data
        st.subheader("Uploaded Transaction Data:")
        st.write(transaction_data.head())

        # Button to start prediction
        if st.button("Detect Fraud"):
            st.write("Performing fraud detection...")

            # Perform prediction
            predictions = predict_fraud(transaction_data)

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

    # Data Visualization Section
    st.sidebar.title("Data Visualization")
    st.sidebar.subheader("Visualize Transaction Data")
    # Add interactive visualization options here (e.g., trend analysis, geographical heatmaps)

    # Real-time Monitoring Section
    st.sidebar.title("Real-time Monitoring")
    st.sidebar.subheader("Enable Real-time Fraud Monitoring")
    # Add options to enable real-time monitoring and set up alerts for suspicious activities

    # Transaction Categorization Section
    st.sidebar.title("Transaction Categorization")
    st.sidebar.subheader("Categorize Transactions")
    # Add options to automatically categorize transactions for better expense tracking

    # Transaction Filtering and Search Section
    st.sidebar.title("Transaction Filtering and Search")
    st.sidebar.subheader("Filter and Search Transactions")
    # Add options for users to filter and search their transaction history based on various criteria

    # Customizable Alerts Section
    st.sidebar.title("Customizable Alerts")
    st.sidebar.subheader("Set Up Custom Alerts")
    # Allow users to customize fraud detection alerts based on their preferences

    # Educational Resources Section
    st.sidebar.title("Educational Resources")
    st.sidebar.subheader("Learn How to Protect Yourself from Fraud")
    # Provide links to educational resources or tips on fraud prevention

    # Exportable Reports Section
    st.sidebar.title("Exportable Reports")
    st.sidebar.subheader("Export Reports for Analysis")
    # Allow users to export transaction data and fraud detection reports for further analysis

    # Feedback Mechanism Section
    st.sidebar.title("Feedback Mechanism")
    st.sidebar.subheader("Provide Feedback")
    # Include a feedback mechanism to gather user input and improve the app

    # Integration with External APIs Section
    st.sidebar.title("Integration with External APIs")
    st.sidebar.subheader("Integrate External Services")
    # Add options to integrate with external APIs for additional features like credit score monitoring

   # Footer with statement by "sserunjogi aaron"
    st.markdown(
        """
        <div style="background-color:#f4f4f4;padding:10px;border-radius:10px">
        <p style="text-align:center;color:#333333;">"We must all work together to prevent fraud and protect our financial assets." - sserunjogi aaron</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
