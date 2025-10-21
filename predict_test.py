import requests

data = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

response = requests.post("http://127.0.0.1:5000/predict", data=data)
print(response.text)
