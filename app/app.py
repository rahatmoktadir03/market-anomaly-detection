from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)

# Load the anomaly detection model
model = joblib.load('models/')

@app.route('/predict', methods=['POST'])
def predict():
    # Get features from the POST request
    data = request.get_json()
    # Process the input data (you may need to adjust this based on your input format)
    input_data = np.array([data['data']]).reshape(1, -1)
    features = pd.DataFrame([data['features']], label_column_name = 'Fixed Income, MM & Interbank')  # Replace with actual columns
    prediction = model.predict(input_data)
    strategy = "sell" if prediction == 1 else "hold"  # Define strategy

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
