import json

import streamlit as st
import websocket

from fraud_detection.config import API_WS_URL


def format_row(scored: dict) -> dict:
    return {
        "transaction_id": scored["transaction_id"],
        "amount": round(scored["amount"], 2),
        "time": scored["time"],
        "if_score": round(scored["isolation_forest"]["score"], 3),
        "if_flagged": scored["isolation_forest"]["flagged"],
        "ae_score": round(scored["autoencoder"]["score"], 3),
        "ae_flagged": scored["autoencoder"]["flagged"],
        "ensemble_flagged": scored["ensemble_flagged"],
    }


def main():
    st.title("Live Fraud Detection Feed")
    table_placeholder = st.empty()
    stats_placeholder = st.empty()

    rows = []
    total = 0
    flagged_count = 0

    ws = websocket.create_connection(API_WS_URL)
    try:
        while True:
            message = ws.recv()
            scored = json.loads(message)
            row = format_row(scored)
            rows.insert(0, row)
            rows[:] = rows[:200]
            total += 1
            if row["ensemble_flagged"]:
                flagged_count += 1

            table_placeholder.dataframe(rows)
            stats_placeholder.write(f"Total: {total} | Flagged: {flagged_count}")
    finally:
        ws.close()


if __name__ == "__main__":
    main()
