import json

import fakeredis

from fraud_detection.replay_producer import load_transactions, publish_transactions


def test_publish_transactions_writes_all_rows_in_order(sample_csv_path):
    df = load_transactions(sample_csv_path)
    small_df = df.head(3)
    client = fakeredis.FakeStrictRedis(decode_responses=True)

    publish_transactions(small_df, client, stream_name="transactions", delay_seconds=0)

    entries = client.xrange("transactions", "-", "+")
    assert len(entries) == 3
    for (_, fields), (idx, row) in zip(entries, small_df.iterrows()):
        assert fields["transaction_id"] == str(idx)
        assert json.loads(fields["data"])["Time"] == row["Time"]
