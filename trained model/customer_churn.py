# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report

# Load data from dataset folder
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, 'dataset', 'telecom_customer_churn.csv')
df = pd.read_csv(csv_path)

# Drop unnecessary columns
drop_cols = ['Customer ID', 'Total Refunds', 'Zip Code', 'Latitude', 'Longitude', 'Churn Category', 'Churn Reason']
df1 = df.drop(columns=drop_cols, errors='ignore').copy()

# Show unique values for categorical columns
def unique_values_names(df):
    for column in df:
        if df[column].dtype == 'object':
            print(f'{column}: {df[column].unique()}')
unique_values_names(df1)

# Check missing values
print("Missing values per column:")
print(df1.isnull().sum() / df1.shape[0])

# Encode target variable
df1['Customer Status Encoded'] = df1['Customer Status'].map({'Stayed': 0, 'Churned': 1, 'Joined': 2})

# Select features and target
feature_cols = df1.select_dtypes(include=['number']).drop(columns=['Customer Status Encoded']).columns
X = df1[feature_cols]
y = df1['Customer Status Encoded']

# Split data (only for churn prediction: 0=Stayed, 1=Churned)
X = X[y.isin([0, 1])]
y = y[y.isin([0, 1])]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Model definitions
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'GaussianNB': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'XGBClassifier': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

results = []
for name, model in models.items():
    model.fit(X_train_imputed, y_train)
    y_pred = model.predict(X_test_imputed)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    results.append({'Model': name, 'Accuracy': acc, 'Precision': prec})

results_df = pd.DataFrame(results)

# Plot accuracy and precision
plt.figure(figsize=(10,6))
sns.barplot(x='Model', y='Accuracy', data=results_df, color='blue', label='Accuracy')
sns.barplot(x='Model', y='Precision', data=results_df, color='orange', alpha=0.6, label='Precision')
plt.ylabel('Score')
plt.title('Model Accuracy and Precision Comparison')
plt.legend()
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Best model by accuracy
best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
print(f"Best model by accuracy: {best_model_name}")
best_model = models[best_model_name]

# Retrain best model on full data
X_imputed_full = imputer.fit_transform(X)
best_model.fit(X_imputed_full, y)

print(f"{best_model_name} retrained on full dataset and ready for predictions.")

# Random Forest detailed evaluation
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_imputed, y_train)
y_pred = rf_model.predict(X_test_imputed)
acc = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {acc}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Churn probability distribution
y_proba = rf_model.predict_proba(X_test_imputed)
churn_count = np.sum(y_pred == 1)
stay_count = np.sum(y_pred == 0)
total = len(y_pred)
print(f"Churn Rate: {churn_count / total:.2%}")
print(f"Stay Rate: {stay_count / total:.2%}")

plt.hist(y_proba[:, 1], bins=20, alpha=0.7)
plt.xlabel("Predicted Probability of Churn")
plt.ylabel("Number of Customers")
plt.title("Distribution of Predicted Churn Probabilities")
plt.show()

# Save the best model (Random Forest), imputer, and results DataFrame
joblib.dump(rf_model, os.path.join(os.path.dirname(__file__), 'rf_model.pkl'))
joblib.dump(imputer, os.path.join(os.path.dirname(__file__), 'imputer.pkl'))
results_df.to_csv(os.path.join(os.path.dirname(__file__), 'model_results.csv'), index=False)

# Save feature columns for later use in web app
import joblib
joblib.dump(list(feature_cols), os.path.join(os.path.dirname(__file__), 'feature_names.pkl'))