# app.py
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and preprocessing objects
model = joblib.load('crop_rf_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = [
            data['n'],
            data['p'],
            data['k'],
            data['temperature'],
            data['humidity'],
            data['ph'],
            data['rainfall']
        ]
        
        # Preprocess and predict
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)
        predicted_crop = le.inverse_transform(prediction)[0]
        
        return jsonify({'crop': predicted_crop})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
