import json
from pathlib import Path

import joblib

from config import FEATURES, METADATA_FILE, MODEL_FILE, TARGET_CAP

def load_model(model_dir):
    """
    Load the saved model and its metadata. Called once when the API starts.
    Args:
        model_dir (Path): Folder holding the model and metadata files.
    Returns:
        model, metadata (tuple): The fitted estimator and the training metadata dict.
    Raises:
        FileNotFoundError: If either file is missing (run train.py first).
    """
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / MODEL_FILE)
    metadata = json.loads((model_dir / METADATA_FILE).read_text())
    return model, metadata

def predict_data(model, X):
    """
    Predict median house values for the input data.
    Args:
        model: The fitted estimator.
        X (list or numpy.ndarray): Rows of features in FEATURES order.
    Returns:
        y_pred (numpy.ndarray): Predicted values in units of $100,000.
    """
    return model.predict(X)

def input_warnings(row, prediction, ranges):
    """
    Flag inputs the model has seen little training data for, and predictions at the
    dataset's price cap.
    Args:
        row (list): One row of features in FEATURES order.
        prediction (float): The model's prediction for the row, in $100,000 units.
        ranges (dict): Training percentiles from the metadata, {feature: {"p1", "p99"}}.
    Returns:
        warnings (list[str]): Human-readable warnings, empty if none apply.
    """
    warnings = []
    for feature, value in zip(FEATURES, row):
        lo, hi = ranges[feature]["p1"], ranges[feature]["p99"]
        if not lo <= value <= hi:
            warnings.append(f"{feature}={value} is outside the 1st-99th percentile of "
                            f"training data ({lo} to {hi}); the prediction may be unreliable.")
    if prediction >= TARGET_CAP:
        warnings.append("Training values are capped at $500,001, so the true value may be higher.")
    return warnings
