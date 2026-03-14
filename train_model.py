# ---------------------------------------------------------------------
# Script: train_model.py
# Training entrypoint for the machine learning pipeline.
#
# This script orchestrates the full training workflow. It loads the
# census dataset, preprocesses features using shared utilities,
# trains the model, evaluates performance, and saves the trained
# artifacts for later use by the API service.
#
# Responsibilities include:
# - loading the dataset
# - preprocessing features
# - training the machine learning model
# - evaluating model performance
# - saving model artifacts for deployment
# ---------------------------------------------------------------------
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)


# Resolve the project root dynamically instead of hardcoding a path.
project_path = os.path.dirname(os.path.abspath(__file__))

# Load the census dataset.
data_path = os.path.join(project_path, "data", "census.csv")
print(data_path)
data = pd.read_csv(data_path, skipinitialspace=True)

# ---------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------
label = "salary"
test_size = 0.20
random_state = 42

# Split the dataset into training and test sets.
# Optional enhancement, use K-fold cross validation instead of a train-test
#  split.
train, test = train_test_split(
    data, test_size=test_size, random_state=random_state)

# DO NOT MODIFY
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

# Process the training data.
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label=label,
    training=True,
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label=label,
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train the model on the training dataset.
model = train_model(X_train, y_train)

# Save the model, encoder, and label binarizer artifacts.
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)

encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)

lb_path = os.path.join(project_path, "model", "lb.pkl")
save_model(lb, lb_path)

# Reload the model from disk to confirm the saved artifact can be reused.
model = load_model(model_path)

# Run inference on the test dataset.
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Compute model performance across categorical slices.
for col in cat_features:
    # iterate through the unique values in one categorical feature
    for slicevalue in sorted(test[col].unique()):
        metrics = performance_on_categorical_slice(
            test,
            col,
            slicevalue,
            cat_features,
            label,
            encoder,
            lb,
            model,
        )

        p = metrics["precision"]
        r = metrics["recall"]
        fb = metrics["fbeta"]
        count = metrics["count"]

        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(
                f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
