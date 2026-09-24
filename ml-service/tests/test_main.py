"""
Smoke tests for the FastAPI app in main.py: confirms predictor errors come
back as clean {"detail": "..."} JSON with the right HTTP status -- never a
raw Python traceback. Uses the same tiny synthetic models as test_predict.py.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from main import app
from src import predict as predictor

client = TestClient(app)


def test_predict_returns_503_when_untrained(monkeypatch, tmp_path):
    monkeypatch.setattr(predictor, "MODEL_DIR", str(tmp_path))
    predictor._cache.clear()

    response = client.post(
        "/predict", json={"text": "this is a normal length sentence", "model": "logistic_regression"}
    )

    assert response.status_code == 503
    body = response.json()
    assert "detail" in body
    assert "Traceback" not in body["detail"]


def test_predict_returns_422_for_empty_text(use_trained_model_dir):
    response = client.post("/predict", json={"text": "", "model": "logistic_regression"})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_predict_returns_422_for_unknown_model(use_trained_model_dir):
    response = client.post(
        "/predict", json={"text": "this is a normal length sentence", "model": "not_a_real_model"}
    )
    assert response.status_code == 422


def test_predict_returns_valid_shape_when_trained(use_trained_model_dir):
    response = client.post(
        "/predict",
        json={"text": "senate committee approves new budget plan", "model": "logistic_regression"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in ("REAL", "FAKE")
    assert body["model"] == "logistic_regression"


def test_health_reports_trained_state(use_trained_model_dir):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["trained"] is True
    assert set(body["models_available"]) == {"logistic_regression", "multinomial_nb"}


def test_health_reports_untrained_state(monkeypatch, tmp_path):
    monkeypatch.setattr(predictor, "MODEL_DIR", str(tmp_path))
    predictor._cache.clear()

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["trained"] is False
    assert body["models_available"] == []
