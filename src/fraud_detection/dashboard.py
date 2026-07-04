import json

import pandas as pd
import requests
import streamlit as st
import websocket

from fraud_detection.config import API_BASE_URL, API_WS_URL


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


def highlight_flagged_row(row: pd.Series) -> list:
    if row["ensemble_flagged"]:
        return ["background-color: #ffcccc"] * len(row)
    return [""] * len(row)


def build_comparison_panel(metrics: dict) -> dict:
    panel = {}
    for model_name in ("isolation_forest", "autoencoder"):
        model_metrics = metrics[model_name]
        panel[model_name] = {
            "precision": round(model_metrics["precision"], 3),
            "recall": round(model_metrics["recall"], 3),
            "auc": round(model_metrics["auc"], 3),
        }
    return panel


def main():
    st.title("Live Fraud Detection Feed")

    try:
        response = requests.get(f"{API_BASE_URL}/models", timeout=5)
        response.raise_for_status()
        panel = build_comparison_panel(response.json())
        st.subheader("Model Comparison")
        columns = st.columns(2)
        for column, model_name in zip(columns, ("isolation_forest", "autoencoder")):
            with column:
                st.write(model_name.replace("_", " ").title())
                st.metric("Precision", panel[model_name]["precision"])
                st.metric("Recall", panel[model_name]["recall"])
                st.metric("AUC", panel[model_name]["auc"])
    except requests.exceptions.RequestException:
        st.warning("Model metrics unavailable — is the API running?")

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

            table_placeholder.dataframe(pd.DataFrame(rows).style.apply(highlight_flagged_row, axis=1))
            stats_placeholder.write(f"Total: {total} | Flagged: {flagged_count}")
    finally:
        ws.close()


if __name__ == "__main__":
    main()
