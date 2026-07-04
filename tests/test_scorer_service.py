import json

import fakeredis
import numpy as np

from fraud_detection.scorer_service import ensure_group, read_and_handle_batch, score_transaction


class _StubScaler:
    def transform(self, df):
        return df[["Amount", "Time"]].to_numpy()


class _StubModel:
    def __init__(self, score_value, flagged_value):
        self.score_value = score_value
        self.flagged_value = flagged_value

    def score(self, X):
        return np.full(X.shape[0], self.score_value)

    def flag(self, X):
        return np.full(X.shape[0], self.flagged_value)


def _sample_raw_transaction():
    row = {f"V{i}": 0.0 for i in range(1, 29)}
    row["Amount"] = 10.0
    row["Time"] = 100.0
    return row


def test_score_transaction_returns_expected_fields():
    result = score_transaction(
        _sample_raw_transaction(), _StubScaler(), _StubModel(0.9, True), _StubModel(0.2, False)
    )
    assert result == {"if_score": 0.9, "if_flagged": True, "ae_score": 0.2, "ae_flagged": False}


def test_read_and_handle_batch_publishes_combined_result():
    server = fakeredis.FakeServer()
    client = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
    subscriber = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
    pubsub = subscriber.pubsub()
    pubsub.subscribe("scored_transactions")
    pubsub.get_message(timeout=1)

    ensure_group(client, "transactions", "scorer-group")
    payload = {"transaction_id": "tx-1", "data": json.dumps(_sample_raw_transaction())}
    client.xadd("transactions", payload)

    processed = read_and_handle_batch(
        client, _StubScaler(), _StubModel(0.9, True), _StubModel(0.2, False),
        stream_name="transactions", group_name="scorer-group",
        consumer_name="scorer-1", channel_name="scored_transactions",
        block_ms=100, count=10,
    )

    assert processed == 1
    message = pubsub.get_message(timeout=1)
    scored = json.loads(message["data"])
    assert scored["transaction_id"] == "tx-1"
    assert scored["ensemble_flagged"] is True
    assert scored["amount"] == 10.0
    assert scored["time"] == 100.0
