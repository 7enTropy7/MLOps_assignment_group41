"""
This script performs hyperparameter optimization for an MLPRegressor 
using the Optuna library and visualizes the optimization 
process using Optuna Dashboard.
"""
import warnings
import optuna
from optuna_dashboard import run_server
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')

data = pd.read_csv('boston_housing.csv')
X = data.drop('TARGET', axis=1)
y = data['TARGET']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

def objective(trial):
    """
    Objective function for Optuna to perform hyperparameter-tuning.
    """
    params = {
        'learning_rate_init': trial.suggest_float('learning_rate_init ', 0.0001, 0.1, step=0.005),
        'first_layer_neurons': trial.suggest_int('first_layer_neurons', 10, 100, step=10),
        'second_layer_neurons': trial.suggest_int('second_layer_neurons', 10, 100, step=10),
        'activation': trial.suggest_categorical('activation', ['identity', 'tanh', 'relu']),
    }

    model = MLPRegressor(
        hidden_layer_sizes=(params['first_layer_neurons'], params['second_layer_neurons']),
        learning_rate_init=params['learning_rate_init'],
        activation=params['activation'],
        random_state=1,
        max_iter=100
    )

    model.fit(X_train, y_train)

    return mean_squared_error(y_valid, model.predict(X_valid), squared=False)

storage = optuna.storages.InMemoryStorage()
study = optuna.create_study(storage=storage, direction='minimize')
study.optimize(objective, n_trials=3)

run_server(storage)
