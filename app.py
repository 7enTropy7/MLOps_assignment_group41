"""
Flask app to host a trained MLPRegressor model and make predictions.

This app loads a pre-trained MLPRegressor model and scaler from files and exposes
a POST endpoint '/predict' to receive JSON data with features for prediction.
The app scales the input data using the loaded scaler, makes predictions using
the model, and returns the predictions as JSON.
"""
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

MODEL_FILENAME = 'mlp_regressor_model.pkl'
SCALER_FILENAME = 'scaler.pkl'

loaded_mlp = joblib.load(MODEL_FILENAME)
loaded_scaler = joblib.load(SCALER_FILENAME)

@app.route('/predict', methods=['POST'])
def predict():
    """Flask Endpoint to predict housing prices using a trained MLPRegressor model. 
    Accepts a POST request with JSON payload containing features for prediction.

    :return: JSON response with predicted housing prices
    :rtype: dict
    """
    data = request.get_json(force=True)
    new_data = np.array(data['features'])
    if len(new_data.shape) == 1:
        new_data = new_data.reshape(1, -1)
    new_data_scaled = loaded_scaler.transform(new_data)
    predictions = loaded_mlp.predict(new_data_scaled)
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=False)
