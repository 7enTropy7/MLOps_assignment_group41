from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import joblib

boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['TARGET'] = boston.target

X = data.drop('TARGET', axis=1)
y = data['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

mlp.fit(X_train, y_train)

model_filename = 'mlp_regressor_model.pkl'
joblib.dump(mlp, model_filename)

scaler_filename = 'scaler.pkl'
joblib.dump(scaler, scaler_filename)

y_pred = mlp.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print('Mean Squared Error: {}'.format(mse))
print('Training Set Score: {}'.format(mlp.score(X_train, y_train)))
print('Test Set Score: {}'.format(mlp.score(X_test, y_test)))
