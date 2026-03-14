# ---------------------------------------------------------------------
# Module: test_ml
# Unit tests for the machine learning pipeline.
#
# The goal of this module is to verify that the core model and data
# functions behave as expected. These tests help confirm that data
# preparation, training, prediction, metrics, and model persistence
# are working correctly.
#
# Responsibilities include:
# - testing data preprocessing behavior
# - testing model training behavior
# - testing prediction output
# - validating evaluation metric calculations
# - confirming model save and load behavior
# - validating slice-based performance output
#
# These tests are executed with pytest and provide a quick way to
# confirm that the ML pipeline is functioning correctly.
# ---------------------------------------------------------------------

import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data, apply_label
from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    save_model,
    load_model,
    performance_on_categorical_slice,
)


# ---------------------------------------------------------------------
# Test Data and Expected Results
# Centralized test inputs reduce repetition and make updates easier.
# This keeps the tests easier to read and maintain.
# ---------------------------------------------------------------------

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

LABEL_COLUMN = "salary"

TEST_DATA = pd.DataFrame(
    {
        "age": [39, 50, 38, 53],
        "workclass": ["State-gov", "Self-emp-not-inc", "Private", "Private"],
        "fnlwgt": [77516, 83311, 215646, 234721],
        "education": ["Bachelors", "Bachelors", "HS-grad", "11th"],
        "education-num": [13, 13, 9, 7],
        "marital-status": [
            "Never-married",
            "Married-civ-spouse",
            "Divorced",
            "Married-civ-spouse",
        ],
        "occupation": [
            "Adm-clerical",
            "Exec-managerial",
            "Handlers-cleaners",
            "Handlers-cleaners",
        ],
        "relationship": ["Not-in-family", "Husband", "Not-in-family", "Husband"],
        "race": ["White", "White", "White", "Black"],
        "sex": ["Male", "Male", "Male", "Male"],
        "capital-gain": [2174, 0, 0, 0],
        "capital-loss": [0, 0, 0, 0],
        "hours-per-week": [40, 13, 40, 40],
        "native-country": [
            "United-States",
            "United-States",
            "United-States",
            "United-States",
        ],
        "salary": ["<=50K", ">50K", "<=50K", ">50K"],
    }
)

TEST_ACTUALS = [1, 0, 1, 1]
TEST_PREDICTIONS = [1, 0, 0, 1]

EXPECTED_PRECISION = 1.000
EXPECTED_RECALL = 0.667
EXPECTED_F1 = 0.800

EXPECTED_ROW_COUNT = 4
EXPECTED_PREDICTION_COUNT = 4
EXPECTED_BINARY_CLASSES = {0, 1}


# ---------------------------------------------------------------------
# Data Processing Tests
# These tests verify that preprocessing returns the expected structures
# and that labels are converted correctly.
# ---------------------------------------------------------------------

def test_process_data_training_returns_expected_outputs():
    """
    Verify that process_data returns processed features, labels,
    encoder, and label binarizer in training mode.
    """
    X, y, encoder, lb = process_data(
        TEST_DATA,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL_COLUMN,
        training=True,
    )

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(y) == EXPECTED_ROW_COUNT
    assert encoder is not None
    assert lb is not None


def test_process_data_inference_returns_expected_row_count():
    """
    Verify that process_data returns the correct number of rows
    in inference mode when using a fitted encoder and label binarizer.
    """
    _, _, encoder, lb = process_data(
        TEST_DATA,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL_COLUMN,
        training=True,
    )

    inference_data = TEST_DATA.drop(columns=[LABEL_COLUMN])

    X, y, returned_encoder, returned_lb = process_data(
        inference_data,
        categorical_features=CATEGORICAL_FEATURES,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    assert isinstance(X, np.ndarray)
    assert X.shape[0] == EXPECTED_ROW_COUNT
    assert isinstance(y, np.ndarray)
    assert len(y) == 0
    assert returned_encoder is encoder
    assert returned_lb is lb


def test_apply_label_converts_binary_output_to_string():
    """
    Verify that apply_label converts binary model output
    into the expected salary label string.
    """
    assert apply_label([1]) == ">50K"
    assert apply_label([0]) == "<=50K"


# ---------------------------------------------------------------------
# Model Training and Inference Tests
# These tests verify that the model trains successfully and produces
# predictions in the expected format.
# ---------------------------------------------------------------------

def test_train_model_returns_random_forest():
    """
    Verify that train_model returns a RandomForestClassifier.
    """
    X_train, y_train, _, _ = process_data(
        TEST_DATA,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL_COLUMN,
        training=True,
    )

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)


def test_inference_returns_expected_number_of_predictions():
    """
    Verify that inference returns one prediction per input row.
    """
    X_train, y_train, _, _ = process_data(
        TEST_DATA,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL_COLUMN,
        training=True,
    )

    model = train_model(X_train, y_train)
    preds = inference(model, X_train)

    assert len(preds) == EXPECTED_PREDICTION_COUNT


def test_inference_predictions_are_valid_binary_classes():
    """
    Verify that inference only returns expected binary class values.
    """
    X_train, y_train, _, _ = process_data(
        TEST_DATA,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL_COLUMN,
        training=True,
    )

    model = train_model(X_train, y_train)
    preds = inference(model, X_train)

    assert set(preds).issubset(EXPECTED_BINARY_CLASSES)


# ---------------------------------------------------------------------
# Metric Tests
# These tests verify that evaluation metrics are calculated correctly
# for a known set of actual and predicted values.
# ---------------------------------------------------------------------

def test_compute_model_metrics_expected_values():
    """
    Verify that precision, recall, and F1-score match expected values.
    """
    precision, recall, fbeta = compute_model_metrics(TEST_ACTUALS, TEST_PREDICTIONS)

    assert round(precision, 3) == EXPECTED_PRECISION
    assert round(recall, 3) == EXPECTED_RECALL
    assert round(fbeta, 3) == EXPECTED_F1


# ---------------------------------------------------------------------
# Persistence Tests
# These tests verify that model artifacts can be saved to disk and
# loaded back successfully.
# ---------------------------------------------------------------------

def test_save_and_load_model_round_trip():
    """
    Verify that a trained model can be saved and loaded successfully.
    """
    X_train, y_train, _, _ = process_data(
        TEST_DATA,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL_COLUMN,
        training=True,
    )

    model = train_model(X_train, y_train)

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "test_model.pkl")

        save_model(model, model_path)
        loaded_model = load_model(model_path)

        assert isinstance(loaded_model, RandomForestClassifier)


# ---------------------------------------------------------------------
# Slice Performance Tests
# These tests verify that metrics can be calculated correctly for
# a categorical slice of the data.
# ---------------------------------------------------------------------

def test_performance_on_categorical_slice_returns_expected_keys():
    """
    Verify that performance_on_categorical_slice returns the expected
    metric fields for a selected slice value.
    """
    X_train, y_train, encoder, lb = process_data(
        TEST_DATA,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL_COLUMN,
        training=True,
    )

    model = train_model(X_train, y_train)

    slice_metrics = performance_on_categorical_slice(
        data=TEST_DATA,
        column_name="workclass",
        slice_value="Private",
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL_COLUMN,
        encoder=encoder,
        lb=lb,
        model=model,
    )

    expected_keys = {
        "column_name",
        "slice_value",
        "count",
        "precision",
        "recall",
        "fbeta",
    }

    assert isinstance(slice_metrics, dict)
    assert set(slice_metrics.keys()) == expected_keys
    assert slice_metrics["column_name"] == "workclass"
    assert slice_metrics["slice_value"] == "Private"