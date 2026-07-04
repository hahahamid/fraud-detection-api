def combine(
    transaction_id: str,
    if_score: float,
    if_flagged: bool,
    ae_score: float,
    ae_flagged: bool,
    amount: float,
    time: float,
) -> dict:
    return {
        "transaction_id": transaction_id,
        "amount": amount,
        "time": time,
        "isolation_forest": {"score": if_score, "flagged": if_flagged},
        "autoencoder": {"score": ae_score, "flagged": ae_flagged},
        "ensemble_flagged": if_flagged or ae_flagged,
    }
