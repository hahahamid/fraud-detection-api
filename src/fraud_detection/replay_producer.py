import argparse
import json
import time

import pandas as pd
import redis

from fraud_detection.config import REDIS_DB, REDIS_HOST, REDIS_PORT, TRANSACTIONS_STREAM
from fraud_detection.redis_utils import wait_for_redis


def load_transactions(csv_path) -> pd.DataFrame:
    return pd.read_csv(csv_path).sort_values("Time").reset_index(drop=True)


def publish_transactions(
    df: pd.DataFrame,
    redis_client,
    stream_name: str = TRANSACTIONS_STREAM,
    delay_seconds: float = 0.5,
) -> None:
    for idx, row in df.iterrows():
        payload = {"transaction_id": str(idx), "data": json.dumps(row.to_dict())}
        redis_client.xadd(stream_name, payload)
        time.sleep(delay_seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    args = parser.parse_args()

    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    wait_for_redis(client)
    df = load_transactions(args.csv_path)
    publish_transactions(df, client, delay_seconds=args.delay_seconds)


if __name__ == "__main__":
    main()
