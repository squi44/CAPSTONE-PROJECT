# Capstone Project - Fraud Detection

## Project Overview

### Area of Concern
This project focuses on leveraging data science to detect fraud in financial transactions within the fields of accounting and finance. The challenge involves addressing the growing complexity of fraudulent activities leading to financial losses, aiming to develop models to identify patterns indicating fraud, thus enhancing the security of financial transactions.

### User Perspective
The project's findings aim to benefit businesses, banks, credit card companies, and online payment platforms by providing an efficient way to detect and prevent fraudulent activities. Implementation of the project results can minimize losses, protect reputations, and ensure the reliability of financial transactions.

### Main Concept
The project utilizes machine learning techniques for fraud detection by analyzing patterns and anomalies in transactions. It employs supervised learning for classification, unsupervised learning for anomaly detection, and ensemble methods. The analysis includes examining transaction frequency, location data, transaction amounts, and utilizing algorithms to identify patterns indicating fraudulent behavior.

### Impact
The project holds significant societal and business value, potentially saving billions of dollars, reducing investigation time and effort, and enhancing the overall security and reliability of financial transactions.

## Dataset Overview

The "Credit Card Fraud Detection" dataset contains transactions made by credit card holders in September 2013 by European cardholders. Key details include:
- 492 frauds out of 284,807 transactions
- Highly unbalanced dataset, with frauds accounting for 0.172% of all transactions
- Features V1 to V28 are PCA-transformed principal components
- 'Time' and 'Amount' are the only non-transformed features
- 'Time' represents seconds elapsed between each transaction and the first transaction
- 'Amount' is the transaction amount
- 'Class' is the response variable, taking value 1 in case of fraud and 0 otherwise

Unfortunately, due to confidentiality issues, original features and background information about the data cannot be provided.

## Preliminary Exploratory Data Analysis (EDA)

The EDA conducted includes:
- Identification of data quality issues
- Exploration of feature engineering opportunities
- Notable observations regarding data preprocessing
- Visualizations to describe relationships between variables and formulate hypotheses for further analysis

## Findings and Preprocessing

- The dataset is highly imbalanced, with the positive class (frauds) accounting for only 0.172% of all transactions.
- Features V1 to V28 are principal components obtained through PCA transformation.
- 'Time' and 'Amount' are the only non-transformed features.
- Initial analysis indicates that standard scaling is necessary due to differences in the range of column values.
- Correlation matrix analysis is insignificant due to the lack of meaningful information.
- Standard scaling is applied to the dataset to prepare for model training.

## Model Training and Evaluation

### Logistic Regression
- Achieved an accuracy score of 100% due to standard scaling but with a lower F1-Score of 72%.

### Random Forest Classifier
- Achieved high overall accuracy and F1-Scores, demonstrating strong performance for class 0(Non-Fraudulent transactions)

### Decision Tree Classifier
- Performance was similar to logistic regression with a slightly lower F1-Score.

### Extreme Gradient Boost (XGBoost)
- Showed improved performance with an F1-Score of 86%, indicating good results despite the unbalanced training data.

## Next Steps

1. Implement techniques to address class imbalance effectively, such as oversampling or ensemble methods.
2. Further fine-tuning and optimization of models to improve performance.
3. Explore advanced anomaly detection algorithms for fraud detection.
4. Deploy and integrate the best-performing model into a real-time fraud detection system.
5. Continuously monitor and update the model to adapt to changing fraud patterns.

## Conclusion

The initial analysis and model training show promising results in detecting credit card fraud. By addressing class imbalance and optimizing model performance, the project aims to provide an effective solution for financial institutions to detect and prevent fraudulent activities, ultimately enhancing the security and reliability of financial transactions.

