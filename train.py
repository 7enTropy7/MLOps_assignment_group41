"""
Train a Multilayer Perceptron (MLP) regressor on the Boston Housing dataset 
using scikit-learn.
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import joblib

data = pd.read_csv('boston_housing.csv')
X = data.drop('TARGET', axis=1)
y = data['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

mlp.fit(X_train, y_train)

MODEL_FILENAME = 'mlp_regressor_model.pkl'
joblib.dump(mlp, MODEL_FILENAME)

SCALER_FILENAME = 'scaler.pkl'
joblib.dump(scaler, SCALER_FILENAME)

y_pred = mlp.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Training Set Score: {mlp.score(X_train, y_train)}')
print(f'Test Set Score: {mlp.score(X_test, y_test)}')
