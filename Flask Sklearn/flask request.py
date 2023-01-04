
import requests

url = "http://localhost:5000/predict"

data = {'feature1': 1, 'feature2': 2, 'feature3': 3}

headers = {'Content-Type': 'application/json'}

response = requests.post(url, json=data, headers=headers)

prediction = response.json()

print(prediction)