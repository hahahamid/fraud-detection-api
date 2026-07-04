from fraud_detection.ensemble import combine


def test_ensemble_flagged_when_either_model_flags():
    result = combine("tx-1", 0.9, True, 0.1, False, 149.62, 100.0)
    assert result["ensemble_flagged"] is True
    assert result["isolation_forest"] == {"score": 0.9, "flagged": True}
    assert result["autoencoder"] == {"score": 0.1, "flagged": False}
    assert result["amount"] == 149.62
    assert result["time"] == 100.0


def test_ensemble_not_flagged_when_neither_flags():
    result = combine("tx-2", 0.1, False, 0.2, False, 10.0, 200.0)
    assert result["ensemble_flagged"] is False


def test_ensemble_flagged_when_both_flag():
    result = combine("tx-3", 0.9, True, 0.95, True, 5.0, 300.0)
    assert result["ensemble_flagged"] is True
    assert result["transaction_id"] == "tx-3"
