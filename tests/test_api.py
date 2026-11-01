import pytest
from fastapi.testclient import TestClient

import main

# First row of the dataset; its true median house value is $452,600
SAMPLE = {
    "med_inc": 8.3252, "house_age": 41.0, "ave_rooms": 6.98, "ave_bedrms": 1.02,
    "population": 322.0, "ave_occup": 2.56, "latitude": 37.88, "longitude": -122.23,
}


@pytest.fixture(scope="module")
def client():
    # The context manager runs the lifespan, which loads the model
    with TestClient(main.app) as c:
        yield c


def test_health_reports_model_loaded(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_predict_returns_price_in_usd(client):
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "1.0.0"
    assert body["warnings"] == []
    # Within the model's test RMSE (~$46k) of the true value
    assert abs(body["median_house_value"] - 452_600) < 50_000


def test_predict_warns_on_rare_inputs(client):
    response = client.post("/predict", json={**SAMPLE, "med_inc": 14.5})
    assert response.status_code == 200
    warnings = response.json()["warnings"]
    assert len(warnings) == 1
    assert warnings[0].startswith("MedInc=14.5 is outside")


@pytest.mark.parametrize("change", [
    {"med_inc": -1},                     # below gt=0
    {"latitude": 50},                    # outside California
    {"ave_bedrms": 9, "ave_rooms": 5},   # more bedrooms than rooms
    {"house_age": "old"},                # not a number
])
def test_predict_rejects_invalid_input(client, change):
    response = client.post("/predict", json={**SAMPLE, **change})
    assert response.status_code == 422


def test_predict_rejects_missing_field(client):
    incomplete = {k: v for k, v in SAMPLE.items() if k != "latitude"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_batch_matches_single_predictions_in_order(client):
    other = {**SAMPLE, "med_inc": 2.0, "latitude": 34.05, "longitude": -118.25}
    response = client.post("/predict/batch", json={"samples": [SAMPLE, other]})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    for sample, prediction in zip([SAMPLE, other], predictions):
        single = client.post("/predict", json=sample).json()
        assert prediction["median_house_value"] == single["median_house_value"]


@pytest.mark.parametrize("size", [0, 101])
def test_batch_rejects_empty_or_oversized(client, size):
    response = client.post("/predict/batch", json={"samples": [SAMPLE] * size})
    assert response.status_code == 422


def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    info = response.json()
    assert info["model_name"] == "hist_gradient_boosting"
    assert info["test_metrics"]["r2"] > 0.8
    assert set(info["cv_rmse"]) == {"linear_regression", "decision_tree", "hist_gradient_boosting"}


def test_missing_model_returns_503(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "MODEL_DIR", tmp_path)
    with TestClient(main.app) as c:
        assert c.get("/").json() == {"status": "healthy", "model_loaded": False}
        assert c.post("/predict", json=SAMPLE).status_code == 503
        assert c.get("/model/info").status_code == 503
