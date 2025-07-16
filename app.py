from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("model/fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Map friendly field names to actual model feature names
        field_mapping = {
            "risk_score": "V4",
            "behavior_anomaly": "V10",
            "transaction_spike": "V11",
            "pattern_shift": "V12",
            "suspicious_level": "V14",
            "amount": "Amount"
        }

        # Reconstruct input list in correct order
        input_values = []
        for friendly, actual in field_mapping.items():
            val = float(request.form[friendly])
            input_values.append(val)

        # Predict
        values = np.array(input_values).reshape(1, -1)
        prediction = model.predict(values)[0]
        result = "⚠️ Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"
        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
