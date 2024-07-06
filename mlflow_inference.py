"""Fetch correct version of trained model 
from MLFlow based on run_id and perform inference.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
import mlflow

mlflow.autolog()
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

RUN_ID = 'e39a4ea1e07b4560b30b2d44796e6c57'
MODEL_NAME = 'boston_housing_model'
MODEL_URI = f"runs:/{RUN_ID}/{MODEL_NAME}"
loaded_model = mlflow.pyfunc.load_model(MODEL_URI)

test_data = np.array([0.00632, 18.0, 2.31, 0.0, 0.538, 6.575,
                             65.2, 4.0900, 1.0, 296.0, 15.3, 396.90, 4.98])

scaler = StandardScaler()
if test_data.ndim == 1:
    test_data = test_data.reshape(1, -1)
test_data_scaled = scaler.fit_transform(test_data)

predictions = loaded_model.predict(test_data_scaled)

print(f'Predictions: {predictions}')
