# ---------------------------------------------------------------------
# Module: ml.model
# Central location for model-related operations.
#
# The goal of this module is to keep all model behavior in one place so
# the rest of the pipeline (training scripts, API endpoints, tests)
# can interact with the model through a clean interface.
# 
# Responsibilities include:
# - training the model
# - generating predictions
# - calculating evaluation metrics
# - persisting model artifacts to disk
# --------------------------------------------------------------------- 
#import pickle -- IGNORE 
# --- Chose to use joblib instead of pickle for model serialization, as it is more efficient for large numpy arrays.
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data

# Optional: Hyperparameter tuning could be implemented here using
# GridSearchCV or RandomizedSearchCV, but for this project a baseline
# RandomForestClassifier provides sufficient performance.  
# Maybe as a Future Enhancement, we could implement hyperparameter tuning and model selection here, and then update the training script to call this function instead of directly instantiating a RandomForestClassifier.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns a fitted model.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Training Labels.

    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
        Precision score.
    recall : float
        Recall score.
    fbeta : float
        F-beta score.
    """
    # zero_division=1 prevents metric calculation from failing when a slice
    # has no predicted positives. Not perfect, but it keeps the pipeline moving.
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences using a trained model and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Prediction labels
    """
    preds = model.predict(X)
    return preds

def save_model(model, path):
    """Serialize a model or encoder to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path):
    """Load a serialized model or encoder from disk.""" 
    return joblib.load(path)


def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """ Computes the model metrics on a slice of the data specified by a column name and

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features: list
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    model : RandomForestClassifier 
        Trained machine learning model.

    Returns
    -------
    slice_metrics : list[dict]
        Metrics for each category value of the selected feature.
    """
    # Filter the dataset to only the rows matching the requested slice.
    slice_df = data[data[column_name] == slice_value]

    # Reuse the fitted encoder and label binarizer so the slice is transformed
    # exactly the same way as the training and test data.
    X_slice, y_slice, _, _ = process_data(
        slice_df,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb
    )

    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)

    slice_metrics = {
        "column_name": column_name,
        "slice_value": slice_value,
        "count": len(slice_df),
        "precision": precision,
        "recall": recall,
        "fbeta": fbeta,
    }

    return slice_metrics