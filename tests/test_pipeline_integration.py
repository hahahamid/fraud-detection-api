import json

import fakeredis

from fraud_detection.replay_producer import load_transactions, publish_transactions
from fraud_detection.scorer_service import ensure_group, load_models, read_and_handle_batch
from fraud_detection.train import train


def test_producer_to_scorer_round_trip(sample_csv_path, tmp_path):
    models_dir = tmp_path / "models"
    train(data_path=sample_csv_path, models_dir=models_dir)
    scaler, if_model, ae_model = load_models(models_dir)

    df = load_transactions(sample_csv_path).head(3)

    server = fakeredis.FakeServer()
    client = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
    subscriber = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
    pubsub = subscriber.pubsub()
    pubsub.subscribe("scored_transactions")
    pubsub.get_message(timeout=1)

    publish_transactions(df, client, stream_name="transactions", delay_seconds=0)
    ensure_group(client, "transactions", "scorer-group")
    processed = read_and_handle_batch(
        client, scaler, if_model, ae_model,
        stream_name="transactions", group_name="scorer-group",
        consumer_name="scorer-1", channel_name="scored_transactions",
        block_ms=100, count=10,
    )

    assert processed == 3
    for _ in range(3):
        message = pubsub.get_message(timeout=1)
        scored = json.loads(message["data"])
        assert set(scored.keys()) == {
            "transaction_id", "isolation_forest", "autoencoder", "ensemble_flagged",
        }
        assert 0.0 <= scored["isolation_forest"]["score"] <= 1.0
        assert 0.0 <= scored["autoencoder"]["score"] <= 1.0
