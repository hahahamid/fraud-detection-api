import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score

from fraud_detection.config import MODELS_DIR, RAW_DATA_PATH
from fraud_detection.models.autoencoder import AutoencoderModel
from fraud_detection.models.isolation_forest import IsolationForestModel
from fraud_detection.preprocessing import fit_scaler, save_scaler, transform


def chronological_split(df: pd.DataFrame, train_fraction: float = 0.7):
    df_sorted = df.sort_values("Time").reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_fraction)
    return df_sorted.iloc[:split_idx].copy(), df_sorted.iloc[split_idx:].copy()


def pick_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if len(thresholds) == 0:
        return 0.5
    denom = precision[:-1] + recall[:-1]
    f1 = np.where(denom > 0, 2 * precision[:-1] * recall[:-1] / np.where(denom > 0, denom, 1), 0)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx])


def evaluate(scores: np.ndarray, flags: np.ndarray, labels: np.ndarray, latencies_ms: list) -> dict:
    true_positives = int((flags & (labels == 1)).sum())
    return {
        "precision": true_positives / max(int(flags.sum()), 1),
        "recall": true_positives / max(int((labels == 1).sum()), 1),
        "auc": float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else 0.0,
        "avg_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
    }


def _measure_latencies_ms(model, X: np.ndarray) -> list:
    latencies = []
    for row in X:
        start = time.perf_counter()
        model.score(row.reshape(1, -1))
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def train(data_path=RAW_DATA_PATH, models_dir=MODELS_DIR) -> dict:
    df = pd.read_csv(data_path)
    train_df, test_df = chronological_split(df)

    scaler = fit_scaler(train_df)
    train_legit = train_df[train_df["Class"] == 0]
    X_train = transform(train_legit, scaler)
    X_test = transform(test_df, scaler)
    y_test = test_df["Class"].to_numpy()

    models_dir.mkdir(parents=True, exist_ok=True)
    save_scaler(scaler, models_dir / "scaler.joblib")
    test_df.to_csv(models_dir / "test_holdout.csv", index=False)

    if_model = IsolationForestModel()
    if_model.fit(X_train)
    if_scores = if_model.score(X_test)
    if_model.set_threshold(pick_threshold(if_scores, y_test))
    if_model.save(models_dir / "isolation_forest.joblib")

    ae_model = AutoencoderModel(input_dim=X_train.shape[1])
    ae_model.fit(X_train)
    ae_scores = ae_model.score(X_test)
    ae_model.set_threshold(pick_threshold(ae_scores, y_test))
    ae_model.save(models_dir / "autoencoder.joblib")

    metrics = {
        "isolation_forest": evaluate(
            if_scores, if_scores >= if_model.threshold, y_test, _measure_latencies_ms(if_model, X_test)
        ),
        "autoencoder": evaluate(
            ae_scores, ae_scores >= ae_model.threshold, y_test, _measure_latencies_ms(ae_model, X_test)
        ),
    }
    with open(models_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    result = train()
    print(json.dumps(result, indent=2))
