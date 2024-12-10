"""
Script to post to FastAPI instance for model inference
author: Jefferson
Date: Dec. 10th 2024
"""

import requests
import json

url = "https://ml-devops-udacity.onrender.com/inference/"
# url = "http://localhost:8000/inference/"

# explicit the sample to perform inference on
sample = {
    "age": 50,
    "workclass": "Private",
    "fnlgt": 234721,
    "education": "Doctorate",
    "education_num": 16,
    "marital_status": "Separated",
    "occupation": "Exec-managerial",
    "relationship": "Not-in-family",
    "race": "Black",
    "sex": "Female",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 50,
    "native_country": "United-States",
}

data = json.dumps(sample)

# post to API and collect response
response = requests.post(url, data=data)

# display output - response will show sample details + model prediction added
print("response status code", response.status_code)
print("response content:")
print(response.json())
