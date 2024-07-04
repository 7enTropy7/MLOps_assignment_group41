from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model_filename = 'mlp_regressor_model.pkl'
scaler_filename = 'scaler.pkl'
loaded_mlp = joblib.load(model_filename)
loaded_scaler = joblib.load(scaler_filename)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    new_data = np.array(data['features'])
    if len(new_data.shape) == 1:
        new_data = new_data.reshape(1, -1)
    new_data_scaled = loaded_scaler.transform(new_data)
    predictions = loaded_mlp.predict(new_data_scaled)
    
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
