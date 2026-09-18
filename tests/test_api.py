from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_predict_valid_ticker():
    resp = client.get("/predict?ticker=NVDA")
    assert resp.status_code == 200
    assert resp.json()["prediction"] in ("up", "down")

def test_predict_unknown_ticker():
    resp = client.get("/predict?ticker=NONSENSE")
    assert resp.status_code == 404

def test_predict_unknown_model():
    resp = client.get("/predict?ticker=NVDA&model_name=NONSENSE")
    assert resp.status_code == 400
