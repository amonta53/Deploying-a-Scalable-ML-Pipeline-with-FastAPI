# ---------------------------------------------------------------------
# Module: test_ml
# Unit tests for the machine learning pipeline.
#
# The goal of this module is to verify that the core model functions
# behave as expected. These tests help confirm that training, metrics,
# and data handling are working correctly.
#
# Responsibilities include:
# - testing model training behavior
# - validating evaluation metric calculations
# - confirming expected data structures and outputs
#
# These tests are executed with pytest and provide a quick way to
# confirm that the ML pipeline is functioning correctly.
# ---------------------------------------------------------------------

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.model import train_model, compute_model_metrics


# ---------------------------------------------------------------------
# Test Data and Expected Results
# Centralized test inputs help reduce repetition and make updates easier
# if the sample data or expected outcomes need to change later.
# ---------------------------------------------------------------------

TEST_FEATURE_DATA = {
    "age": [25, 45, 30, 50],
    "education_num": [13, 9, 10, 14],
    "hours_per_week": [40, 50, 35, 60],
}

TEST_TARGET_DATA = [0, 1, 0, 1]

TEST_X_TRAIN = pd.DataFrame(TEST_FEATURE_DATA)
TEST_Y_TRAIN = pd.Series(TEST_TARGET_DATA)

TEST_ACTUALS = [1, 0, 1, 1]
TEST_PREDICTIONS = [1, 0, 0, 1]

EXPECTED_PRECISION = 1.000
EXPECTED_RECALL = 0.667
EXPECTED_F1 = 0.800

EXPECTED_ROW_COUNT = 4
EXPECTED_COLUMN_COUNT = 3


# ---------------------------------------------------------------------
# Unit Tests
# Each test focuses on one specific part of the ML pipeline so failures
# are easier to understand and troubleshoot.
# ---------------------------------------------------------------------

def test_train_model_returns_random_forest():
    """
    Verify that the training function returns a RandomForestClassifier.

    This test confirms the pipeline is using the expected algorithm
    and that the model training function produces a valid model object.
    """
    model = train_model(TEST_X_TRAIN, TEST_Y_TRAIN)

    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics_expected_values():
    """
    Verify that precision, recall, and F1-score are calculated correctly.

    This test uses a small fixed set of actual and predicted values so the
    expected metric results are known in advance and easy to validate.
    """
    precision, recall, fbeta = compute_model_metrics(TEST_ACTUALS, TEST_PREDICTIONS)

    assert round(precision, 3) == EXPECTED_PRECISION
    assert round(recall, 3) == EXPECTED_RECALL
    assert round(fbeta, 3) == EXPECTED_F1


def test_training_data_structure():
    """
    Verify that the sample training data has the expected structure.

    This test confirms that the feature data is stored in a DataFrame,
    the target data is stored in a Series, and both have the expected size.
    """
    assert isinstance(TEST_X_TRAIN, pd.DataFrame)
    assert isinstance(TEST_Y_TRAIN, pd.Series)
    assert TEST_X_TRAIN.shape == (EXPECTED_ROW_COUNT, EXPECTED_COLUMN_COUNT)
    assert len(TEST_Y_TRAIN) == EXPECTED_ROW_COUNT