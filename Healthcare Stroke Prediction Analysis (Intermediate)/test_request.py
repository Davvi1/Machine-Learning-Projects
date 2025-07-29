import requests

url = "http://127.0.0.1:5000/predict"

sample_input = {
    "features": [67, 0, 1, 0, 1, 0, 0, 1, 0.8, 0]
}

response = requests.post(url, json=sample_input)

print("Status Code:", response.status_code)
print("Response:", response.json())
