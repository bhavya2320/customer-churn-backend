from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained Logistic Regression model
model = joblib.load("churn_model.pkl")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    customer = pd.DataFrame({
        "tenure": [data["tenure"]],
        "monthly_charges": [data["monthly_charges"]],
        "support_calls": [data["support_calls"]],
        "contract": [data["contract"]]
    })

    prediction = model.predict(customer)[0]

    probability = model.predict_proba(customer)[0][1]

    if prediction == 1:
        result = "CHURN"
        risk = "HIGH"
    else:
        result = "NOT CHURN"
        risk = "LOW"

    return jsonify({
        "customer_id": data.get("customer_id", "Unknown"),
        "churn_probability": round(float(probability) * 100, 2),
        "prediction": result,
        "risk_level": risk
    })


@app.route("/", methods=["GET"])
def home():
    return "Customer Churn Prediction API is running!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)