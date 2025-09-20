from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os

app = Flask(__name__)

# Get the absolute path to the trained model folder
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
trained_model_dir = os.path.join(base_dir, 'trained model')

# Load the trained model and imputer saved from your code
model = joblib.load(os.path.join(trained_model_dir, 'rf_model.pkl'))
imputer = joblib.load(os.path.join(trained_model_dir, 'imputer.pkl'))
results_df = pd.read_csv(os.path.join(trained_model_dir, 'model_results.csv'))  # Or save results_df from your training code and load here
feature_names = joblib.load(os.path.join(trained_model_dir, 'feature_names.pkl'))

def plot_metrics():
    plt.figure(figsize=(10,6))
    sns.barplot(x='Model', y='Accuracy', data=results_df, color='blue', label='Accuracy')
    sns.barplot(x='Model', y='Precision', data=results_df, color='orange', alpha=0.6, label='Precision')
    plt.ylabel('Score')
    plt.title('Model Accuracy and Precision Comparison')
    plt.legend()
    plt.xticks(rotation=15)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.clf()
    return plot_url

@app.route('/')
def home():
    chart = plot_metrics()
    return render_template('index.html', chart=chart, feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {}
    for feature in feature_names:
        value = request.form.get(feature)
        input_data[feature] = float(value) if value else 0.0

    input_df = pd.DataFrame([input_data])
    input_imputed = imputer.transform(input_df)
    prediction = model.predict(input_imputed)[0]
    proba = model.predict_proba(input_imputed)[0][1]

    status = "Churned" if prediction == 1 else "Stayed"
    return jsonify({'prediction': status, 'probability': proba})

if __name__ == '__main__':
    app.run(debug=True)
