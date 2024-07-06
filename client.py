"""
Script to send a POST request with JSON data 
to a local server for prediction.
"""
import requests

data = {
    'features': [
        0.00632, 18.0, 2.31, 0.0,
        0.538, 6.575, 65.2, 4.0900,
        1.0, 296.0, 15.3, 396.90, 4.98
    ]
}

URL = 'http://127.0.0.1:8080/predict'
response = requests.post(URL, json=data, timeout=1.5)

print(response.json())
