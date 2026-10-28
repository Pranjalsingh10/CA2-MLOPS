from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)

SAMPLE_WINE = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
}


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_quality_and_probabilities() -> None:
    response = client.post("/predict", json=SAMPLE_WINE)

    assert response.status_code == 200
    result = response.json()
    assert result["predicted_quality"] in {3, 4, 5, 6, 7, 8}
    assert set(result["probabilities"]) == {"3", "4", "5", "6", "7", "8"}
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6


def test_predict_rejects_missing_or_extra_features() -> None:
    missing_feature = dict(SAMPLE_WINE)
    missing_feature.pop("alcohol")
    extra_feature = {**SAMPLE_WINE, "color": "red"}

    assert client.post("/predict", json=missing_feature).status_code == 422
    assert client.post("/predict", json=extra_feature).status_code == 422
