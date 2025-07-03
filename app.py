from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the trained machine learning pipeline
try:
    model = joblib.load('task_duration_model.pkl')
except FileNotFoundError:
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not found. Please train the model first by running train.py'}), 500
    
    try:
        data = request.get_json(force=True)
        task_name = data.get('task_name')

        if not task_name:
            return jsonify({'error': 'task_name not provided'}), 400

        # The input to the pipeline's predict method should be an iterable.
        prediction = model.predict([task_name])

        # The prediction is a numpy array, so we get the first element.
        # We also cast it to a standard Python float for JSON serialization.
        duration = float(prediction[0])

        return jsonify({'predicted_duration': duration})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
