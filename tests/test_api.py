import json

import fakeredis
import fakeredis.aioredis

from fraud_detection.api import app, get_models_dir, get_redis_client
from fraud_detection.config import SCORED_CHANNEL

from fastapi.testclient import TestClient


class _StubScaler:
    def transform(self, df):
        return df[["Amount", "Time"]].to_numpy()


class _StubModel:
    def __init__(self, score_value, flagged_value):
        self.score_value = score_value
        self.flagged_value = flagged_value

    def score(self, X):
        import numpy as np
        return np.full(X.shape[0], self.score_value)

    def flag(self, X):
        import numpy as np
        return np.full(X.shape[0], self.flagged_value)


def _sample_transaction():
    row = {f"V{i}": 0.0 for i in range(1, 29)}
    row["Amount"] = 10.0
    row["Time"] = 100.0
    return row


def test_score_endpoint_returns_combined_result():
    with TestClient(app) as client:
        app.state.scaler = _StubScaler()
        app.state.if_model = _StubModel(0.9, True)
        app.state.ae_model = _StubModel(0.2, False)

        response = client.post("/score", json=_sample_transaction())

        assert response.status_code == 200
        body = response.json()
        assert body["ensemble_flagged"] is True
        assert body["isolation_forest"]["score"] == 0.9
        assert body["autoencoder"]["score"] == 0.2
        assert body["amount"] == 10.0
        assert body["time"] == 100.0


def test_score_endpoint_422_on_missing_field():
    with TestClient(app) as client:
        app.state.scaler = _StubScaler()
        app.state.if_model = _StubModel(0.9, True)
        app.state.ae_model = _StubModel(0.2, False)

        incomplete = _sample_transaction()
        del incomplete["V1"]
        response = client.post("/score", json=incomplete)

        assert response.status_code == 422


def test_models_endpoint_returns_metrics(tmp_path):
    metrics = {"isolation_forest": {"auc": 0.9}, "autoencoder": {"auc": 0.85}}
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    app.dependency_overrides[get_models_dir] = lambda: tmp_path

    try:
        with TestClient(app) as client:
            response = client.get("/models")
            assert response.status_code == 200
            assert response.json() == metrics
    finally:
        app.dependency_overrides.clear()


def test_models_endpoint_404_when_missing(tmp_path):
    app.dependency_overrides[get_models_dir] = lambda: tmp_path

    try:
        with TestClient(app) as client:
            response = client.get("/models")
            assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_stream_forwards_scored_message():
    server = fakeredis.FakeServer()
    publisher = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
    app.dependency_overrides[get_redis_client] = lambda: fakeredis.aioredis.FakeRedis(
        server=server, decode_responses=True
    )

    try:
        with TestClient(app) as client:
            with client.websocket_connect("/stream") as websocket:
                publisher.publish(SCORED_CHANNEL, json.dumps({"transaction_id": "abc", "ensemble_flagged": True}))
                received = websocket.receive_text()
                assert json.loads(received)["transaction_id"] == "abc"
    finally:
        app.dependency_overrides.clear()
