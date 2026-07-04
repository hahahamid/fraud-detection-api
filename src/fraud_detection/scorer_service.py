import json
import logging

import pandas as pd
import redis

from fraud_detection.config import (
    MODELS_DIR,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT,
    SCORED_CHANNEL,
    TRANSACTIONS_CONSUMER_GROUP,
    TRANSACTIONS_STREAM,
)
from fraud_detection.ensemble import combine
from fraud_detection.models.autoencoder import AutoencoderModel
from fraud_detection.models.isolation_forest import IsolationForestModel
from fraud_detection.preprocessing import load_scaler, transform
from fraud_detection.redis_utils import wait_for_redis

logger = logging.getLogger(__name__)


def load_models(models_dir=MODELS_DIR):
    scaler = load_scaler(models_dir / "scaler.joblib")
    if_model = IsolationForestModel.load(models_dir / "isolation_forest.joblib")
    ae_model = AutoencoderModel.load(models_dir / "autoencoder.joblib")
    return scaler, if_model, ae_model


def score_transaction(raw_data: dict, scaler, if_model, ae_model) -> dict:
    df = pd.DataFrame([raw_data])
    X = transform(df, scaler)
    return {
        "if_score": float(if_model.score(X)[0]),
        "if_flagged": bool(if_model.flag(X)[0]),
        "ae_score": float(ae_model.score(X)[0]),
        "ae_flagged": bool(ae_model.flag(X)[0]),
    }


def ensure_group(redis_client, stream_name: str, group_name: str) -> None:
    try:
        redis_client.xgroup_create(stream_name, group_name, id="0", mkstream=True)
    except redis.exceptions.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


def handle_message(message_id, fields, redis_client, scaler, if_model, ae_model,
                    stream_name, group_name, channel_name):
    try:
        transaction_id = fields["transaction_id"]
        raw_data = json.loads(fields["data"])
        result = score_transaction(raw_data, scaler, if_model, ae_model)
        scored = combine(
            transaction_id,
            result["if_score"], result["if_flagged"],
            result["ae_score"], result["ae_flagged"],
        )
        redis_client.publish(channel_name, json.dumps(scored))
    except Exception:
        logger.exception("Failed to process message %s, skipping", message_id)
    finally:
        redis_client.xack(stream_name, group_name, message_id)


def read_and_handle_batch(redis_client, scaler, if_model, ae_model, stream_name, group_name,
                           consumer_name, channel_name, block_ms=5000, count=10) -> int:
    response = redis_client.xreadgroup(
        group_name, consumer_name, {stream_name: ">"}, count=count, block=block_ms
    )
    if not response:
        return 0
    processed = 0
    for _stream, messages in response:
        for message_id, fields in messages:
            handle_message(message_id, fields, redis_client, scaler, if_model, ae_model,
                            stream_name, group_name, channel_name)
            processed += 1
    return processed


def run_forever(redis_client, scaler, if_model, ae_model, stream_name=TRANSACTIONS_STREAM,
                 group_name=TRANSACTIONS_CONSUMER_GROUP, consumer_name="scorer-1",
                 channel_name=SCORED_CHANNEL, block_ms=5000, count=10):
    ensure_group(redis_client, stream_name, group_name)
    while True:
        read_and_handle_batch(redis_client, scaler, if_model, ae_model, stream_name, group_name,
                               consumer_name, channel_name, block_ms=block_ms, count=count)


if __name__ == "__main__":
    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    wait_for_redis(client)
    scaler, if_model, ae_model = load_models()
    run_forever(client, scaler, if_model, ae_model)
