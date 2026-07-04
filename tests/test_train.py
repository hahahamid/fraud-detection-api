import json

from fraud_detection.train import train


def test_train_produces_models_and_metrics(sample_csv_path, tmp_path):
    models_dir = tmp_path / "models"
    metrics = train(data_path=sample_csv_path, models_dir=models_dir)

    assert (models_dir / "scaler.joblib").exists()
    assert (models_dir / "isolation_forest.joblib").exists()
    assert (models_dir / "autoencoder.joblib").exists()
    assert (models_dir / "test_holdout.csv").exists()
    assert (models_dir / "metrics.json").exists()

    with open(models_dir / "metrics.json") as f:
        saved_metrics = json.load(f)
    assert saved_metrics == metrics

    for model_name in ("isolation_forest", "autoencoder"):
        assert 0.0 <= metrics[model_name]["precision"] <= 1.0
        assert 0.0 <= metrics[model_name]["recall"] <= 1.0
        assert metrics[model_name]["auc"] > 0.5
        assert metrics[model_name]["avg_latency_ms"] >= 0.0
