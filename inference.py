"""
Perform Inference on the Trained Model.
"""
import joblib
import numpy as np

MODEL_FILENAME = 'mlp_regressor_model.pkl'
SCALER_FILENAME = 'scaler.pkl'

loaded_mlp = joblib.load(MODEL_FILENAME)
loaded_scaler = joblib.load(SCALER_FILENAME)

def run_inference(new_data):
    """
    Runs inference using the loaded MLP model on new data.
    """
    if new_data.ndim == 1:
        new_data = new_data.reshape(1, -1)
    new_data_scaled = loaded_scaler.transform(new_data)
    predictions = loaded_mlp.predict(new_data_scaled)
    return predictions

if __name__ == "__main__":
    example_data = np.array([0.00632, 18.0, 2.31, 0.0, 0.538, 6.575,
                             65.2, 4.0900, 1.0, 296.0, 15.3, 396.90, 4.98])

    prediction = run_inference(example_data)
    print(f'Prediction for example data: {prediction}')
