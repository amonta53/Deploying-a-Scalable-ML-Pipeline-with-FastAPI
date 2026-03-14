# ---------------------------------------------------------------------
# Module: local_api
# Local script for interacting with the FastAPI application.
#
# The goal of this module is to verify that the API is working by
# sending one GET request and one POST request to the locally running
# application.
#
# Responsibilities include:
# - sending a GET request to the root endpoint
# - sending a POST request to the prediction endpoint
# - printing response status codes and returned results
# ---------------------------------------------------------------------

import requests

# ---------------------------------------------------------------------
# API Configuration
# Base URL and sample payload used for local testing.
# ---------------------------------------------------------------------
BASE_URL = "http://127.0.0.1:8000"
GET_URL = f"{BASE_URL}/"
POST_URL = f"{BASE_URL}/data/"
TEST_PAYLOAD = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 284582,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 50,
    "native-country": "United-States",
}

# ---------------------------------------------------------------------
# Local API Calls
# Send one GET request and one POST request, then print status codes
# and results so the API behavior can be verified.
# ---------------------------------------------------------------------
get_response = requests.get(GET_URL)
print(f"Status Code: {get_response.status_code}")
print(f"Result: {get_response.json()['message']}")

post_response = requests.post(POST_URL, json=TEST_PAYLOAD)
print(f"Status Code: {post_response.status_code}")
print(f"Result: {post_response.json()['result']}")
