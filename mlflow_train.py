"""
Version Control for MLP Regressor Model using MLFlow
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

import mlflow
from mlflow.models import infer_signature

data = pd.read_csv('boston_housing.csv')
X = data.drop('TARGET', axis=1)
y = data['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    "hidden_layer_sizes": (70,),
    "max_iter": 1200,
    "random_state": 69
}

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPRegressor(**params)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
train_score = mlp.score(X_train, y_train)
test_score = mlp.score(X_test, y_test)
print(f'Mean Squared Error: {mse}')
print(f'Training Set Score: {train_score}')
print(f'Test Set Score: {test_score}')

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("MLflow Quickstart")

with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("training r2 score", train_score)
    mlflow.log_metric("testing r2 score", test_score)
    mlflow.log_metric("mse", mse)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic MLP Regressor model for Boston Housing Dataset")

    # Infer the model signature
    signature = infer_signature(X_train, mlp.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=mlp,
        artifact_path="boston_housing_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )
