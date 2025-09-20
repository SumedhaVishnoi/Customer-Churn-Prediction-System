# Telecom Customer Churn Prediction Project

This project builds a machine learning pipeline to predict customer churn in the telecom industry. It involves data preprocessing, model training and evaluation, performance visualization, and deployment as an interactive Flask web application.

---

## Project Overview

Customer churn—the loss of customers—is critical for telecom companies. This project uses historical customer data to train classification models to predict which customers are likely to churn, enabling proactive retention strategies.

---

## Features

- **Data preparation:** Cleaning, removing irrelevant columns, encoding target variables, handling missing values.
- **Modeling:** Training multiple classifiers (Random Forest, Logistic Regression, GaussianNB, Decision Tree, XGBoost).
- **Evaluation:** Calculating accuracy and precision for each model; visualizing performance comparison.
- **Model selection:** Automatically selects and retrains the best performing model on the full dataset.
- **Prediction insights:** Output detailed classification report, churn probability, and churn vs stay rates.
- **Web application:** Flask app with interactive input form for churn prediction, visual display of model metrics, and prediction probabilities.

---

## Technologies Used

- Python 3.8+
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, flask, joblib

---

## Installation

1. Clone the repository or download the project files.
2. Ensure Python 3.8+ is installed.
3. Install dependencies:


---

## Dataset

This project uses the `telecom_customer_churn.csv` dataset containing customer demographics, usage, and churn status.

---

## Data Processing

- Unnecessary columns such as customer IDs, geolocation, and refund details are removed.
- Target variable "Customer Status" is encoded into numerical labels.
- Numeric features are selected for modeling.
- Data is split into training and testing sets, stratified by churn status.
- Missing values are imputed using mean strategy.

---

## Model Training & Evaluation

- Five models are trained and evaluated: Random Forest, Logistic Regression, GaussianNB, Decision Tree, and XGBoost.
- Accuracy and precision are calculated for performance comparison.
- Results are visualized using bar plots with matplotlib and seaborn.
- The best model based on accuracy is retrained on the full dataset and saved.

---

## Prediction and Insights

- Random Forest model is used for detailed evaluation.
- Outputs include classification report, churn probability distribution, and churn rates.
- Probability histograms help understand the confidence of predictions.

---

## Web Application (Flask)

- Provides an interactive input form to enter customer features and get churn predictions with probabilities.
- Displays performance comparison charts for all models.
- Dynamically serves prediction results as JSON for easy integration.
- Deployed locally for testing and demonstration.

---

## Running the Project

### Model Training

Run `customer_churn.py` to preprocess data, train models, evaluate, and save the best model (`rf_model.pkl`) and imputer (`imputer.pkl`).

### Run Web App

1. Ensure model artifacts (`rf_model.pkl`, `imputer.pkl`) and `model_results.csv` are in the app directory.
2. Run the Flask app:


3. Visit `http://127.0.0.1:5000/` to interact with the app.

---

## Future Enhancements

- Adding survival analysis to predict time to churn.
- Implementing customer segmentation to personalize predictions.
- Incorporating explainability methods like SHAP for interpretability.
- Deploying the app on cloud platforms for production use.

---

## Author

[Sumedha Vishnoi]



---
