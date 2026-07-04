import pandas as pd

from fraud_detection.dashboard import format_row, highlight_flagged_row


def test_format_row_flattens_scored_transaction():
    scored = {
        "transaction_id": "tx-1",
        "amount": 149.619,
        "time": 100.0,
        "isolation_forest": {"score": 0.8765, "flagged": True},
        "autoencoder": {"score": 0.1234, "flagged": False},
        "ensemble_flagged": True,
    }
    row = format_row(scored)
    assert row == {
        "transaction_id": "tx-1",
        "amount": 149.62,
        "time": 100.0,
        "if_score": 0.876,
        "if_flagged": True,
        "ae_score": 0.123,
        "ae_flagged": False,
        "ensemble_flagged": True,
    }


def test_highlight_flagged_row_returns_red_background_when_flagged():
    row = pd.Series({"transaction_id": "tx-1", "amount": 10.0, "ensemble_flagged": True})
    styles = highlight_flagged_row(row)
    assert styles == ["background-color: #ffcccc"] * len(row)


def test_highlight_flagged_row_returns_empty_when_not_flagged():
    row = pd.Series({"transaction_id": "tx-1", "amount": 10.0, "ensemble_flagged": False})
    styles = highlight_flagged_row(row)
    assert styles == [""] * len(row)
