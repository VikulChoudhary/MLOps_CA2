
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert "model_loaded" in response.json()


def test_features_endpoint():
    response = client.get("/features")

    if response.status_code == 200:
        data = response.json()

        assert "features" in data
        assert "target" in data
        assert data["target"] == "Fertility Rates"
    else:
        assert response.status_code == 503


def test_metrics_endpoint():
    response = client.get("/metrics")

    if response.status_code == 200:
        data = response.json()

        assert "metrics" in data
        assert "dataset_rows" in data
    else:
        assert response.status_code == 503


def test_prediction_with_valid_features():
    features_response = client.get("/features")

    if features_response.status_code != 200:
        return

    feature_info = features_response.json()
    features = {}

    for column in feature_info["numeric_features"]:
        features[column] = 2020 if column == "Year" else 100.0

    for column in feature_info["categorical_features"]:
        features[column] = "Unknown"

    response = client.post(
        "/predict",
        json={"features": features},
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))
    assert data["target"] == "Fertility Rates"


def test_prediction_missing_features():
    response = client.post(
        "/predict",
        json={"features": {}},
    )

    assert response.status_code in (422, 503)


def test_unknown_endpoint():
    response = client.get("/this-endpoint-does-not-exist")

    assert response.status_code == 404
