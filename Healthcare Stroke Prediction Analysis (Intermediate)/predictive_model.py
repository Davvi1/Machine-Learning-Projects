from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import json

app = Flask(__name__)
model = joblib.load("best_model.pkl")

with open("feature_columns.json") as f:
    feature_columns = json.load(f)

@app.route('/')
def home():
    return "Stroke prediction model is running."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    features_df = pd.DataFrame([data["features"]], columns=feature_columns)

    prediction = model.predict(features_df)[0]
    probability = model.predict_proba(features_df)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "probability": float(probability)
    })

if __name__ == '__main__':
    app.run(debug=True)