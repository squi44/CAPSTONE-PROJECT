import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb
from scipy.stats import shapiro, kstest
import statsmodels.api as sm

# Function to preprocess data
def preprocess_data(df):
    scaler = StandardScaler()
    df.iloc[:, 1:-1] = scaler.fit_transform(df.iloc[:, 1:-1])
    return df

# Function to predict fraudulence
def predict_fraudulence(model, data):
    return model.predict(data)

# Function to perform statistical tests
def perform_statistical_tests(df):
    shapiro_results = {}
    ks_results = {}
    for col in df.columns:
        shapiro_stat, shapiro_p = shapiro(df[col])
        ks_stat, ks_p = kstest(df[col], 'norm')
        shapiro_results[col] = {'Statistic': shapiro_stat, 'p-value': shapiro_p}
        ks_results[col] = {'Statistic': ks_stat, 'p-value': ks_p}

    return shapiro_results, ks_results

# Function to plot Q-Q plots
def plot_qq_plots(df):
    for col in df.columns:
        sm.qqplot(df[col], line='s')
        plt.title(f"Q-Q Plot for {col}")
        st.pyplot()

# Function to train and evaluate models
def train_and_evaluate_model(df):
    X = df.drop(columns=['Class'], axis=1)
    y = df['Class']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Logistic Regression
    lr_model = LogisticRegression()
    lr_model.fit(x_train, y_train)
    lr_f1 = f1_score(y_test, lr_model.predict(x_test))
    
    # Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(x_train, y_train)
    rf_f1 = f1_score(y_test, rf_model.predict(x_test))
    
    # Decision Tree
    dt_model = DecisionTreeClassifier()
    dt_model.fit(x_train, y_train)
    dt_f1 = f1_score(y_test, dt_model.predict(x_test))
    
    # XGBoost
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(x_train, y_train)
    xgb_f1 = f1_score(y_test, xgb_model.predict(x_test))
    
    return lr_model, rf_model, dt_model, xgb_model, lr_f1, rf_f1, dt_f1, xgb_f1, x_test, y_test

# Function to plot distribution of features
def plot_distribution(df):
    num_columns = len(df.columns[1:-1])
    num_rows = (num_columns + 3) // 4  # Calculate the number of rows needed
    fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(20, num_rows * 5))
    axes = axes.flatten()
    for i, col in enumerate(df.columns[1:-1]):
        sns.distplot(df[col], ax=axes[i])
    plt.tight_layout()
    st.pyplot()

# Function to plot class distribution
def plot_class_distribution(df):
    fig = px.histogram(df, x='Class', title='Class Distribution', color='Class')
    st.plotly_chart(fig)

# Function to plot feature importance
def plot_feature_importance(model, df):
    if isinstance(model, xgb.XGBClassifier):
        fig = xgb.plot_importance(model)
        st.pyplot(fig)
    else:
        feature_importance = model.feature_importances_
        feature_names = df.columns[:-1]
        fig = go.Figure([go.Bar(x=feature_names, y=feature_importance)])
        fig.update_layout(title='Random Forest Feature Importance', xaxis_title='Features', yaxis_title='Importance')
        st.plotly_chart(fig)

# Function to plot precision-recall curve
def plot_precision_recall(models, names, x_test, y_test):
    fig = go.Figure()
    for model, name in zip(models, names):
        precisions, recalls, _ = precision_recall_curve(y_test, model.predict_proba(x_test)[:,1])
        fig.add_trace(go.Scatter(x=recalls, y=precisions, mode='lines', name=name))
    fig.update_layout(xaxis_title='Recall', yaxis_title='Precision', title='Precision-Recall Curve')
    st.plotly_chart(fig)

# Main function
def main():
    st.title("Fraud Detection Dashboard")

    # Load data
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = preprocess_data(df)
        st.success("Data successfully loaded and preprocessed.")
        st.subheader("Data Summary")
        st.write(df.head())

        # Perform statistical tests
        st.subheader("Statistical Tests")
        shapiro_results, ks_results = perform_statistical_tests(df)
        st.write("### Shapiro-Wilk Test Results:")
        st.write(pd.DataFrame.from_dict(shapiro_results, orient='index'))

        st.write("### Kolmogorov-Smirnov Test Results:")
        st.write(pd.DataFrame.from_dict(ks_results, orient='index'))

        # Q-Q Plot
        st.subheader("Q-Q Plots")
        plot_qq_plots(df)

        # Visualization
        st.subheader("Data Distribution")
        plot_distribution(df)
        st.subheader("Class Distribution")
        plot_class_distribution(df)

        # Train and evaluate models
        st.subheader("Model Training and Evaluation")
        lr_model, rf_model, dt_model, xgb_model, lr_f1, rf_f1, dt_f1, xgb_f1, x_test, y_test = train_and_evaluate_model(df)
        st.write("Logistic Regression F1 Score:", lr_f1)
        st.write("Random Forest F1 Score:", rf_f1)
        st.write("Decision Tree F1 Score:", dt_f1)
        st.write("XGBoost F1 Score:", xgb_f1)

        # Model comparison
        st.subheader("Model Comparison")
        models = ['Logistic Regression', 'Random Forest', 'Decision Tree', 'XGBoost']
        f1_scores = [lr_f1, rf_f1, dt_f1, xgb_f1]
        fig = go.Figure(data=[go.Bar(x=models, y=f1_scores)])
        fig.update_layout(title='Model Comparison', xaxis_title='Model', yaxis_title='F1 Score')
        st.plotly_chart(fig)

        # Feature Importance
        st.subheader("Feature Importance")
        plot_feature_importance(rf_model, df)

        # Precision-Recall Curve
        st.subheader("Precision-Recall Curve")
        plot_precision_recall([lr_model, rf_model, dt_model, xgb_model], models, x_test, y_test)

if __name__ == "__main__":
    main()