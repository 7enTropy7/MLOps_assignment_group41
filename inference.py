import joblib
import numpy as np

model_filename = 'mlp_regressor_model.pkl'
scaler_filename = 'scaler.pkl'
loaded_mlp = joblib.load(model_filename)
loaded_scaler = joblib.load(scaler_filename)

def run_inference(new_data):
    if len(new_data.shape) == 1:
        new_data = new_data.reshape(1, -1)
    new_data_scaled = loaded_scaler.transform(new_data)
    predictions = loaded_mlp.predict(new_data_scaled)    
    return predictions

if __name__ == "__main__":
    example_data = np.array([0.00632, 18.0, 2.31, 0.0, 0.538, 6.575, 65.2, 4.0900, 1.0, 296.0, 15.3, 396.90, 4.98])
    
    prediction = run_inference(example_data)
    print('Prediction for example data: {}'.format(prediction))
    
