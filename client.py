import requests

data = {'features': [0.00632, 18.0, 2.31, 0.0, 0.538, 6.575, 65.2, 4.0900, 1.0, 296.0, 15.3, 396.90, 4.98]}

response = requests.post('http://127.0.0.1:5000/predict', json=data)

print(response.json())