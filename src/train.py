import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from config import (FEATURES, METADATA_FILE, MODEL_DIR, MODEL_FILE, MODEL_VERSION,
                    RANDOM_STATE, TARGET_UNIT_USD)
from data import load_data, split_data

# Random forest was left out: it scored worse than gradient boosting and saves to ~145 MB
CANDIDATES = {
    "linear_regression": LinearRegression(),
    "decision_tree": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
    "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
}

def compare_models(X_train, y_train, cv=5):
    """
    Score every candidate model with k-fold cross-validation on the training set.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
        cv (int): Number of folds.
    Returns:
        scores (dict): Mean and standard deviation of the RMSE for each candidate.
    """
    scores = {}
    for name, model in CANDIDATES.items():
        rmse = -cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
        scores[name] = {"rmse_mean": round(float(rmse.mean()), 4), "rmse_std": round(float(rmse.std()), 4)}
    return scores

def fit_model(name, X_train, y_train):
    """
    Fit a fresh copy of the named candidate on the full training set.
    Args:
        name (str): Key in CANDIDATES.
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    Returns:
        model: The fitted estimator.
    """
    model = clone(CANDIDATES[name])
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the held-out test set.
    Args:
        model: A fitted estimator.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test target values.
    Returns:
        metrics (dict): RMSE, MAE (both in $100,000 units and in USD) and R².
    """
    y_pred = model.predict(X_test)
    rmse = float(root_mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(float(r2_score(y_test, y_pred)), 4),
        "rmse_usd": round(rmse * TARGET_UNIT_USD),
        "mae_usd": round(mae * TARGET_UNIT_USD),
    }

def feature_ranges(X_train):
    """
    Record the 1st and 99th percentile of each training feature. The API warns when a
    request falls outside this range, since the model has seen little data there.
    Args:
        X_train (numpy.ndarray): Training features.
    Returns:
        ranges (dict): {feature: {"p1": float, "p99": float}}
    """
    p1 = np.percentile(X_train, 1, axis=0)
    p99 = np.percentile(X_train, 99, axis=0)
    return {f: {"p1": round(float(lo), 4), "p99": round(float(hi), 4)}
            for f, lo, hi in zip(FEATURES, p1, p99)}

def save_model(model, metadata, model_dir=MODEL_DIR):
    """
    Save the model with joblib and its metadata as JSON.
    Args:
        model: A fitted estimator.
        metadata (dict): Training details served by GET /model/info.
        model_dir (Path): Folder to write both files to.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / MODEL_FILE)
    (model_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2))

def train(model_dir=MODEL_DIR):
    """
    Compare the candidates, fit the best one, evaluate it on the test set and save it.
    Args:
        model_dir (Path): Folder to write the model and metadata to.
    Returns:
        metadata (dict): What was saved alongside the model.
    """
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    cv_scores = compare_models(X_train, y_train)
    best = min(cv_scores, key=lambda name: cv_scores[name]["rmse_mean"])
    model = fit_model(best, X_train, y_train)

    metadata = {
        "model_version": MODEL_VERSION,
        "model_name": best,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": FEATURES,
        "target_unit_usd": TARGET_UNIT_USD,
        "cv_rmse": cv_scores,
        "test_metrics": evaluate_model(model, X_test, y_test),
        "feature_ranges": feature_ranges(X_train),
    }
    save_model(model, metadata, model_dir)
    return metadata

if __name__ == "__main__":
    metadata = train()
    for name, score in metadata["cv_rmse"].items():
        print(f"{name:<24} CV RMSE {score['rmse_mean']:.3f} ± {score['rmse_std']:.3f}")
    print(f"Selected {metadata['model_name']}: test {metadata['test_metrics']}")
