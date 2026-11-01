import json

import pytest

from config import FEATURES, METADATA_FILE, MODEL_FILE
from predict import input_warnings, load_model, predict_data
from train import train


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Run the full training pipeline once into a temporary folder."""
    model_dir = tmp_path_factory.mktemp("model")
    metadata = train(model_dir)
    return model_dir, metadata


def test_train_writes_model_and_metadata(trained):
    model_dir, metadata = trained
    assert (model_dir / MODEL_FILE).exists()
    assert json.loads((model_dir / METADATA_FILE).read_text()) == metadata


def test_selects_lowest_cv_rmse(trained):
    _, metadata = trained
    scores = metadata["cv_rmse"]
    assert metadata["model_name"] == min(scores, key=lambda n: scores[n]["rmse_mean"])


def test_model_beats_quality_bar(trained):
    _, metadata = trained
    assert metadata["test_metrics"]["r2"] > 0.8
    assert metadata["test_metrics"]["rmse"] < 0.5


def test_saved_model_loads_and_predicts(trained):
    model_dir, metadata = trained
    model, loaded = load_model(model_dir)
    assert loaded == metadata
    row = [8.3252, 41.0, 6.98, 1.02, 322.0, 2.56, 37.88, -122.23]
    assert 3.0 < predict_data(model, [row])[0] < 5.5


def test_feature_ranges_cover_all_features(trained):
    _, metadata = trained
    ranges = metadata["feature_ranges"]
    assert list(ranges) == FEATURES
    assert all(r["p1"] < r["p99"] for r in ranges.values())


def test_input_warnings():
    ranges = {f: {"p1": 0.0, "p99": 10.0} for f in FEATURES}
    assert input_warnings([5.0] * 8, 2.0, ranges) == []
    assert len(input_warnings([11.0] + [5.0] * 7, 2.0, ranges)) == 1
    assert input_warnings([5.0] * 8, 5.0, ranges)[0].startswith("Training values are capped")
