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

# Main function
def main():
    st.title("Fraud Detection Dashboard")

    # Load data
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.success("Data successfully loaded and preprocessed.")
        st.subheader("Data Summary")
        st.write(df.head())

        
    # Function to predict fraudulence
    def predict_fraudulence(data):
        # Preprocess input data
        processed_data = preprocess_input(data)
        # Predict fraudulence
        prediction = model.predict(processed_data)
        return prediction
    
    # Input form for transaction details
    st.write("### Enter Transaction Details:")
    time = st.number_input("Time Elapsed Since First Transaction (in seconds)", value=0.0)
    amount = st.number_input("Transaction Amount", value=0.0)
    
    # Predict fraudulence on button click
    if st.button("Predict"):
        # Create dataframe with input data
        input_data = pd.DataFrame({'Time': [time], 'Amount': [amount]})
        # Predict fraudulence
        prediction = predict_fraudulence(input_data)
        # Display prediction result
        if prediction[0] == 1:
            st.error("The transaction is **fraudulent**.")
        else:
            st.success("The transaction is **legitimate**.")



# Load dataset
@st.cache
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = preprocess_data(df)
    return df

# Model training and evaluation
def train_and_evaluate_model(df):
    X = df.drop(columns=['Class'], axis=1)
    y = df['Class']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Function to preprocess data
def preprocess_data(df):
    scaler = StandardScaler()
    # Fitting the scaler on the training data and transforming both training and testing data
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    df.iloc[:, 1:-1] = scaler.fit_transform(df.iloc[:, 1:-1])
    return df
    
    
    # Logistic Regression
    lr_model = LogisticRegression()
    lr_model.fit(x_train, y_train)
    lr_y_pred = lr_model.predict(x_test)
    lr_f1 = f1_score(y_test, lr_y_pred)
    
    # Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(x_train, y_train)
    rf_y_pred = rf_model.predict(x_test)
    rf_f1 = f1_score(y_test, rf_y_pred)
    
    # Decision Tree
    dt_model = DecisionTreeClassifier()
    dt_model.fit(x_train, y_train)
    dt_y_pred = dt_model.predict(x_test)
    dt_f1 = f1_score(y_test, dt_y_pred)
    
    # XGBoost
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(x_train, y_train)
    xgb_y_pred = xgb_model.predict(x_test)
    xgb_f1 = f1_score(y_test, xgb_y_pred)
    
    return lr_model, rf_model, dt_model, xgb_model, lr_f1, rf_f1, dt_f1, xgb_f1, x_test, y_test

# Visualizations
df = pd.read_csv(uploaded_file)
df_temp = df.drop(columns=['Time', 'Amount', 'Class'], axis=1)

# creating dist plots for each column
fig, ax = plt.subplots(ncols=4, nrows=7, figsize=(20, 50))
index = 0
ax = ax.flatten()

for col in df_temp.columns:
    sns.distplot(df_temp[col], ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=5)


# Statistical Tests to check for normal distribution 
for col in df_temp.columns:
    # Shapiro-Wilk Test
    stat, p = stats.shapiro(df_temp[col])
    print(f'Shapiro-Wilk Test for {col}: Statistic={stat}, p-value={p}')
    
    # Kolmogorov-Smirnov Test
    stat, p = stats.kstest(df_temp[col], 'norm')
    print(f'Kolmogorov-Smirnov Test for {col}: Statistic={stat}, p-value={p}')
    
# Q-Q Plot
for col in df_temp.columns:
    stats.probplot(df_temp[col], dist="norm", plot=plt)
    plt.title(f"Q-Q plot for {col}")
    plt.show()

# Descriptive Statistics
for col in df_temp.columns:
    mean = df_temp[col].mean()
    median = df_temp[col].median()
    std_dev = df_temp[col].std()
    print(f"Descriptive statistics for {col}: Mean={mean}, Median={median}, Std Dev={std_dev}")

# Descriptive statistics
desc_stats = df.describe()

# Streamlit app
st.title("Credit Card Fraud Detection Analysis")
st.write("### Shapiro-Wilk Test Results:")
st.write(pd.DataFrame.from_dict(shapiro_results, orient='index'))

st.write("### Kolmogorov-Smirnov Test Results:")
st.write(pd.DataFrame.from_dict(ks_results, orient='index'))

st.write("### Descriptive Statistics:")
st.write(desc_stats)

# Interpretation of results
st.write("## Interpretation")
st.write("### Implications of Shapiro-Wilk Test Results:")
st.write("The Shapiro-Wilk test assesses whether the distribution of each feature in the dataset follows a normal distribution. \
If the p-value of the test is less than a significance level (e.g., 0.05), we reject the null hypothesis that the data is normally distributed. \
In this case, the p-values are likely to be very low, indicating significant deviations from normality. This impacts subsequent analyses \
because many statistical methods assume normality, so alternative techniques or transformations may be necessary.")

st.write("### Implications of Kolmogorov-Smirnov Test Results:")
st.write("The Kolmogorov-Smirnov test also assesses normality, but it focuses on the distribution as a whole rather than specific parameters. \
Similar to the Shapiro-Wilk test, low p-values indicate significant deviations from normality, which can impact subsequent analyses.")

st.write("### Interpretation of Q-Q Plots:")
st.write("Q-Q plots visually compare the distribution of each feature in the dataset to a theoretical normal distribution. \
Significant deviations from normality are indicated by deviations from the diagonal reference line. Outliers or non-normal patterns \
in the data can be observed from these plots. If the majority of the points deviate significantly from the diagonal line, it suggests \
that the data does not follow a normal distribution.")

st.write("### Examination of Descriptive Statistics:")
st.write("Descriptive statistics provide summary information about the distribution of each feature. Mean, median, and standard deviation \
are indicators of central tendency and spread of the data. Skewed distributions or presence of outliers can be inferred from these statistics. \
If the mean and median are significantly different, it suggests skewness in the distribution. Outliers can be identified from observations \
that lie far from the mean. These statistics help in understanding the underlying distribution of the data and identifying potential \
issues such as class imbalance or data preprocessing requirements.")


def plot_class_distribution(df):
    fig = px.histogram(df, x='Class', title='Class Distribution', color='Class')
    st.plotly_chart(fig)

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

def plot_precision_recall(models, names, x_test, y_test):
    fig = go.Figure()
    for model, name in zip(models, names):
        precisions, recalls, _ = precision_recall_curve(y_test, model.predict_proba(x_test)[:,1])
        fig.add_trace(go.Scatter(x=recalls, y=precisions, mode='lines', name=name))
    fig.update_layout(xaxis_title='Recall', yaxis_title='Precision', title='Precision-Recall Curve')
    st.plotly_chart(fig)

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
