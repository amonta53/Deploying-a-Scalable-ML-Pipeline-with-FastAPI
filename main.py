# ---------------------------------------------------------------------
# Module: main
# FastAPI application for serving machine learning predictions.
#
# The goal of this module is to expose the trained model through a
# simple RESTful API. The API provides a root endpoint for a welcome
# message and a prediction endpoint for model inference.
#
# Responsibilities include:
# - creating the FastAPI application
# - loading model artifacts
# - defining API request and response behavior
# - generating predictions from incoming request data
# ---------------------------------------------------------------------
import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model

# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

# ---------------------------------------------------------------------
# Model Artifacts
# Load the saved encoder and model so they are available to the API
# during inference requests.
# ---------------------------------------------------------------------

ENCODER_PATH = os.path.join("model", "encoder.pkl")
MODEL_PATH = os.path.join("model", "model.pkl")

encoder = load_model(ENCODER_PATH)
model = load_model(MODEL_PATH)

# ---------------------------------------------------------------------
# FastAPI Application
# Create the RESTful API application instance.
# ---------------------------------------------------------------------

app = FastAPI()

# ---------------------------------------------------------------------
# Root Endpoint
# Return a simple welcome message to confirm the API is running.
# ---------------------------------------------------------------------

@app.get("/")
async def get_root():
    """Return a welcome message from the API."""
    return {"message": "Hello from the API!"}


# ---------------------------------------------------------------------
# Inference Endpoint
# Accept a single record, preprocess the data, run model inference,
# and return the predicted salary label.
# ---------------------------------------------------------------------

@app.post("/data/")
async def post_inference(data: Data):
    """
    Run model inference for a single input record and return the
    predicted salary classification.
    """
    # DO NOT MODIFY: turn the Pydantic model into a dict.
    data_dict = data.dict()

    # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
    # The data has names with hyphens and Python does not allow those as variable names.
    # Here it uses the functionality of FastAPI/Pydantic/etc to deal with this.
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    data_processed, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
    )

    _inference = inference(model, data_processed)
    return {"result": apply_label(_inference)}
