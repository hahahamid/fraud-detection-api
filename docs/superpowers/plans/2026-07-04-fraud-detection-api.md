# Fraud Detection API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real-time fraud detection demo: an offline-trained Isolation Forest + Autoencoder scored against a replayed transaction stream (via Redis Streams, standing in for Kafka), served through a FastAPI layer, and visualized live in a Streamlit dashboard.

**Architecture:** An offline trainer produces two anomaly-detection models and their thresholds from the Kaggle credit card fraud dataset. A producer script replays held-out test transactions into a Redis Stream; a scorer service consumes them, scores with both models, and publishes combined results to a Redis pub/sub channel. FastAPI exposes synchronous scoring plus a websocket that forwards the pub/sub feed; Streamlit renders it live.

**Tech Stack:** Python 3.11+, pandas/numpy, scikit-learn (Isolation Forest), PyTorch (autoencoder), redis-py (Streams + pub/sub, sync and asyncio), FastAPI + uvicorn, Streamlit, joblib, pytest + fakeredis.

## Global Constraints

- Python 3.11+, single local virtual environment — no Docker.
- Redis Streams stands in for Kafka — no real Kafka broker.
- No persistence of streamed/scored transactions — dashboard state is in-memory only (rolling window).
- Ensemble rule: a transaction is `ensemble_flagged` if *either* model flags it (logical OR).
- `data/` (raw Kaggle CSV) and `models/` (trained artifacts) are gitignored — not committed.
- All source lives under `src/fraud_detection/`; tests under `tests/`.
- Model wrapper interface is fixed across both models: `fit(X)`, `score(X) -> np.ndarray` (normalized 0-1, higher = more anomalous), `flag(X) -> np.ndarray[bool]`, `set_threshold(t)`, `save(path)`, classmethod `load(path)`.

---

### Task 1: Project scaffolding, config, and preprocessing

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `src/fraud_detection/__init__.py`
- Create: `src/fraud_detection/config.py`
- Create: `src/fraud_detection/preprocessing.py`
- Create: `src/fraud_detection/redis_utils.py`
- Test: `tests/conftest.py`
- Test: `tests/test_preprocessing.py`
- Test: `tests/test_redis_utils.py`

**Interfaces:**
- Produces: `FEATURE_COLUMNS: list[str]` (30 columns: `V1..V28` + `scaled_amount`, `scaled_time`), `PCA_FEATURE_COLUMNS: list[str]` (`V1..V28`), `DATA_DIR`, `MODELS_DIR`, `RAW_DATA_PATH: Path`, `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `TRANSACTIONS_STREAM: str`, `TRANSACTIONS_CONSUMER_GROUP: str`, `SCORED_CHANNEL: str`, `API_WS_URL: str` from `fraud_detection.config`.
- Produces: `fit_scaler(train_df: pd.DataFrame) -> StandardScaler`, `transform(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray` (shape `(n, 30)`), `save_scaler(scaler, path) -> None`, `load_scaler(path) -> StandardScaler` from `fraud_detection.preprocessing`.
- Produces: `wait_for_redis(redis_client, max_retries: int = 10, initial_delay: float = 1.0) -> None` from `fraud_detection.redis_utils` — pings `redis_client`, retrying with exponential backoff on `redis.exceptions.ConnectionError`, logging each attempt; raises `redis.exceptions.ConnectionError` if still unreachable after `max_retries`.
- Produces (test fixtures, `tests/conftest.py`): `sample_dataframe` — a pytest fixture returning a deterministic 200-row `pd.DataFrame` with columns `Time` (0..199), `V1..V28`, `Amount`, `Class` (1 at every 10th row, 0 otherwise; `V1..V5` shifted +6.0 on fraud rows so the classes are separable). `sample_csv_path` — a pytest fixture (`tmp_path`, `sample_dataframe`) that writes `sample_dataframe` to a CSV in `tmp_path` and returns the `Path`.

- [ ] **Step 1: Create directory structure and manifest files**

Run:
```bash
mkdir -p src/fraud_detection/models tests
touch src/fraud_detection/__init__.py src/fraud_detection/models/__init__.py
```

Create `requirements.txt`:
```
pandas
numpy
scikit-learn
torch
redis
fastapi
uvicorn[standard]
httpx
streamlit
websocket-client
joblib
pytest
fakeredis
```

Create `.gitignore`:
```
.venv/
__pycache__/
*.pyc
/data/
/models/
tmp/
.DS_Store
```

Note the leading slashes on `/data/` and `/models/` — they anchor the pattern to the repo root. An unanchored `models/` would also match `src/fraud_detection/models/`, silently hiding that source package from git.

Create `pyproject.toml`:
```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
```

- [ ] **Step 2: Create config.py**

Create `src/fraud_detection/config.py`:
```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RAW_DATA_PATH = DATA_DIR / "creditcard.csv"

PCA_FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = PCA_FEATURE_COLUMNS + ["scaled_amount", "scaled_time"]

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

TRANSACTIONS_STREAM = "transactions"
TRANSACTIONS_CONSUMER_GROUP = "scorer-group"
SCORED_CHANNEL = "scored_transactions"

API_WS_URL = "ws://localhost:8000/stream"
```

- [ ] **Step 3: Write the failing tests for preprocessing**

Create `tests/conftest.py`:
```python
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_rows = 200
    data = {f"V{i}": rng.normal(0, 1, n_rows) for i in range(1, 29)}
    class_labels = np.array([1 if i % 10 == 0 else 0 for i in range(n_rows)])
    for i in range(1, 6):
        data[f"V{i}"] = data[f"V{i}"] + class_labels * 6.0
    data["Time"] = np.arange(n_rows, dtype=float)
    data["Amount"] = rng.uniform(1, 500, n_rows)
    data["Class"] = class_labels
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_path(tmp_path, sample_dataframe):
    path = tmp_path / "sample_transactions.csv"
    sample_dataframe.to_csv(path, index=False)
    return path
```

Create `tests/test_preprocessing.py`:
```python
import numpy as np

from fraud_detection.preprocessing import (
    FEATURE_COLUMNS,
    fit_scaler,
    load_scaler,
    save_scaler,
    transform,
)


def test_transform_produces_expected_shape(sample_dataframe):
    scaler = fit_scaler(sample_dataframe)
    result = transform(sample_dataframe, scaler)
    assert result.shape == (len(sample_dataframe), len(FEATURE_COLUMNS))


def test_transform_is_deterministic(sample_dataframe):
    scaler = fit_scaler(sample_dataframe)
    first = transform(sample_dataframe, scaler)
    second = transform(sample_dataframe, scaler)
    np.testing.assert_array_equal(first, second)


def test_save_and_load_scaler_round_trip(sample_dataframe, tmp_path):
    scaler = fit_scaler(sample_dataframe)
    path = tmp_path / "scaler.joblib"
    save_scaler(scaler, path)
    loaded = load_scaler(path)
    original = transform(sample_dataframe, scaler)
    reloaded = transform(sample_dataframe, loaded)
    np.testing.assert_array_equal(original, reloaded)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pytest tests/test_preprocessing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fraud_detection.preprocessing'`

- [ ] **Step 5: Implement preprocessing.py**

Create `src/fraud_detection/preprocessing.py`:
```python
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from fraud_detection.config import FEATURE_COLUMNS, PCA_FEATURE_COLUMNS

__all__ = ["FEATURE_COLUMNS", "fit_scaler", "transform", "save_scaler", "load_scaler"]


def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[["Amount", "Time"]])
    return scaler


def transform(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    scaled = scaler.transform(df[["Amount", "Time"]])
    pca_part = df[PCA_FEATURE_COLUMNS].to_numpy()
    return np.hstack([pca_part, scaled])


def save_scaler(scaler: StandardScaler, path) -> None:
    joblib.dump(scaler, path)


def load_scaler(path) -> StandardScaler:
    return joblib.load(path)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_preprocessing.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Write the failing test for redis_utils**

Create `tests/test_redis_utils.py`:
```python
import pytest
import redis

from fraud_detection.redis_utils import wait_for_redis


class _AlwaysFailsPing:
    def ping(self):
        raise redis.exceptions.ConnectionError("unreachable")


class _SucceedsPing:
    def __init__(self):
        self.calls = 0

    def ping(self):
        self.calls += 1
        return True


def test_wait_for_redis_returns_immediately_when_reachable():
    client = _SucceedsPing()
    wait_for_redis(client, max_retries=3, initial_delay=0.01)
    assert client.calls == 1


def test_wait_for_redis_raises_after_max_retries():
    client = _AlwaysFailsPing()
    with pytest.raises(redis.exceptions.ConnectionError):
        wait_for_redis(client, max_retries=3, initial_delay=0.01)
```

- [ ] **Step 8: Run test to verify it fails**

Run: `pytest tests/test_redis_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fraud_detection.redis_utils'`

- [ ] **Step 9: Implement redis_utils.py**

Create `src/fraud_detection/redis_utils.py`:
```python
import logging
import time

import redis

logger = logging.getLogger(__name__)


def wait_for_redis(redis_client, max_retries: int = 10, initial_delay: float = 1.0) -> None:
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            redis_client.ping()
            return
        except redis.exceptions.ConnectionError:
            logger.error(
                "Redis unreachable (attempt %d/%d), retrying in %.1fs", attempt, max_retries, delay
            )
            time.sleep(delay)
            delay *= 2
    raise redis.exceptions.ConnectionError(f"Could not connect to Redis after {max_retries} attempts")
```

- [ ] **Step 10: Run test to verify it passes**

Run: `pytest tests/test_redis_utils.py -v`
Expected: PASS (2 tests)

- [ ] **Step 11: Commit**

```bash
git add requirements.txt .gitignore pyproject.toml src/fraud_detection/__init__.py \
  src/fraud_detection/models/__init__.py src/fraud_detection/config.py \
  src/fraud_detection/preprocessing.py src/fraud_detection/redis_utils.py \
  tests/conftest.py tests/test_preprocessing.py tests/test_redis_utils.py
git commit -m "Add project scaffolding, config, preprocessing, and Redis connection retry helper"
```

---

### Task 2: Isolation Forest model wrapper

**Files:**
- Create: `src/fraud_detection/models/isolation_forest.py`
- Test: `tests/test_isolation_forest.py`

**Interfaces:**
- Produces: `class IsolationForestModel` with `__init__(self, **kwargs)`, `fit(self, X: np.ndarray) -> None`, `score(self, X: np.ndarray) -> np.ndarray` (values in `[0, 1]`, higher = more anomalous), `flag(self, X: np.ndarray) -> np.ndarray[bool]` (uses `self.threshold`), `set_threshold(self, threshold: float) -> None`, `save(self, path) -> None`, classmethod `load(cls, path) -> IsolationForestModel`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_isolation_forest.py`:
```python
import numpy as np

from fraud_detection.models.isolation_forest import IsolationForestModel


def _make_data():
    rng = np.random.default_rng(0)
    inliers = rng.normal(0, 1, size=(100, 5))
    outliers = rng.normal(8, 1, size=(10, 5))
    return inliers, outliers


def test_outliers_score_higher_than_inliers():
    inliers, outliers = _make_data()
    model = IsolationForestModel()
    model.fit(inliers)
    assert model.score(outliers).mean() > model.score(inliers).mean()


def test_save_and_load_round_trip(tmp_path):
    inliers, outliers = _make_data()
    model = IsolationForestModel()
    model.fit(inliers)
    model.set_threshold(0.7)
    path = tmp_path / "isolation_forest.joblib"
    model.save(path)
    loaded = IsolationForestModel.load(path)
    np.testing.assert_array_equal(model.score(outliers), loaded.score(outliers))
    assert loaded.threshold == 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_isolation_forest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fraud_detection.models.isolation_forest'`

- [ ] **Step 3: Implement isolation_forest.py**

Create `src/fraud_detection/models/isolation_forest.py`:
```python
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestModel:
    def __init__(self, **kwargs):
        self._model = IsolationForest(random_state=42, **kwargs)
        self._score_min = 0.0
        self._score_max = 1.0
        self.threshold = 0.5

    def fit(self, X: np.ndarray) -> None:
        self._model.fit(X)
        raw = -self._model.decision_function(X)
        self._score_min = float(raw.min())
        self._score_max = float(raw.max())

    def score(self, X: np.ndarray) -> np.ndarray:
        raw = -self._model.decision_function(X)
        span = self._score_max - self._score_min
        if span == 0:
            return np.zeros_like(raw)
        normalized = (raw - self._score_min) / span
        return np.clip(normalized, 0.0, 1.0)

    def flag(self, X: np.ndarray) -> np.ndarray:
        return self.score(X) >= self.threshold

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold

    def save(self, path) -> None:
        joblib.dump(
            {
                "model": self._model,
                "score_min": self._score_min,
                "score_max": self._score_max,
                "threshold": self.threshold,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "IsolationForestModel":
        payload = joblib.load(path)
        instance = cls()
        instance._model = payload["model"]
        instance._score_min = payload["score_min"]
        instance._score_max = payload["score_max"]
        instance.threshold = payload["threshold"]
        return instance
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_isolation_forest.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fraud_detection/models/isolation_forest.py tests/test_isolation_forest.py
git commit -m "Add Isolation Forest model wrapper"
```

---

### Task 3: Autoencoder model wrapper

**Files:**
- Create: `src/fraud_detection/models/autoencoder.py`
- Test: `tests/test_autoencoder.py`

**Interfaces:**
- Produces: `class AutoencoderModel` with `__init__(self, input_dim: int, epochs: int = 20, lr: float = 1e-3)`, `fit(self, X: np.ndarray) -> None`, `score(self, X: np.ndarray) -> np.ndarray` (values in `[0, 1]`, higher = more anomalous), `flag(self, X: np.ndarray) -> np.ndarray[bool]`, `set_threshold(self, threshold: float) -> None`, `save(self, path) -> None`, classmethod `load(cls, path) -> AutoencoderModel`. Same shape/threshold semantics as `IsolationForestModel` from Task 2.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_autoencoder.py`:
```python
import numpy as np

from fraud_detection.models.autoencoder import AutoencoderModel


def _make_data():
    rng = np.random.default_rng(0)
    inliers = rng.normal(0, 1, size=(100, 5)).astype(np.float32)
    outliers = rng.normal(8, 1, size=(10, 5)).astype(np.float32)
    return inliers, outliers


def test_outliers_score_higher_than_inliers():
    inliers, outliers = _make_data()
    model = AutoencoderModel(input_dim=5, epochs=50)
    model.fit(inliers)
    assert model.score(outliers).mean() > model.score(inliers).mean()


def test_save_and_load_round_trip(tmp_path):
    inliers, outliers = _make_data()
    model = AutoencoderModel(input_dim=5, epochs=50)
    model.fit(inliers)
    model.set_threshold(0.6)
    path = tmp_path / "autoencoder.joblib"
    model.save(path)
    loaded = AutoencoderModel.load(path)
    np.testing.assert_allclose(model.score(outliers), loaded.score(outliers), rtol=1e-5)
    assert loaded.threshold == 0.6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_autoencoder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fraud_detection.models.autoencoder'`

- [ ] **Step 3: Implement autoencoder.py**

Create `src/fraud_detection/models/autoencoder.py`:
```python
import joblib
import numpy as np
import torch
from torch import nn


class _Net(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderModel:
    def __init__(self, input_dim: int, epochs: int = 20, lr: float = 1e-3):
        self.input_dim = input_dim
        self.epochs = epochs
        self.lr = lr
        self._net = _Net(input_dim)
        self._score_min = 0.0
        self._score_max = 1.0
        self.threshold = 0.5

    def fit(self, X: np.ndarray) -> None:
        torch.manual_seed(42)
        tensor = torch.tensor(X, dtype=torch.float32)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self._net.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            reconstructed = self._net(tensor)
            loss = loss_fn(reconstructed, tensor)
            loss.backward()
            optimizer.step()
        raw = self._reconstruction_error(X)
        self._score_min = float(raw.min())
        self._score_max = float(raw.max())

    def _reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        self._net.eval()
        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32)
            reconstructed = self._net(tensor)
            error = torch.mean((reconstructed - tensor) ** 2, dim=1)
        return error.numpy()

    def score(self, X: np.ndarray) -> np.ndarray:
        raw = self._reconstruction_error(X)
        span = self._score_max - self._score_min
        if span == 0:
            return np.zeros_like(raw)
        normalized = (raw - self._score_min) / span
        return np.clip(normalized, 0.0, 1.0)

    def flag(self, X: np.ndarray) -> np.ndarray:
        return self.score(X) >= self.threshold

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold

    def save(self, path) -> None:
        joblib.dump(
            {
                "state_dict": self._net.state_dict(),
                "input_dim": self.input_dim,
                "score_min": self._score_min,
                "score_max": self._score_max,
                "threshold": self.threshold,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "AutoencoderModel":
        payload = joblib.load(path)
        instance = cls(input_dim=payload["input_dim"])
        instance._net.load_state_dict(payload["state_dict"])
        instance._score_min = payload["score_min"]
        instance._score_max = payload["score_max"]
        instance.threshold = payload["threshold"]
        return instance
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_autoencoder.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fraud_detection/models/autoencoder.py tests/test_autoencoder.py
git commit -m "Add Autoencoder model wrapper"
```

---

### Task 4: Ensemble combination logic

**Files:**
- Create: `src/fraud_detection/ensemble.py`
- Test: `tests/test_ensemble.py`

**Interfaces:**
- Produces: `combine(transaction_id: str, if_score: float, if_flagged: bool, ae_score: float, ae_flagged: bool) -> dict` returning `{"transaction_id": str, "isolation_forest": {"score": float, "flagged": bool}, "autoencoder": {"score": float, "flagged": bool}, "ensemble_flagged": bool}` where `ensemble_flagged = if_flagged or ae_flagged`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ensemble.py`:
```python
from fraud_detection.ensemble import combine


def test_ensemble_flagged_when_either_model_flags():
    result = combine("tx-1", 0.9, True, 0.1, False)
    assert result["ensemble_flagged"] is True
    assert result["isolation_forest"] == {"score": 0.9, "flagged": True}
    assert result["autoencoder"] == {"score": 0.1, "flagged": False}


def test_ensemble_not_flagged_when_neither_flags():
    result = combine("tx-2", 0.1, False, 0.2, False)
    assert result["ensemble_flagged"] is False


def test_ensemble_flagged_when_both_flag():
    result = combine("tx-3", 0.9, True, 0.95, True)
    assert result["ensemble_flagged"] is True
    assert result["transaction_id"] == "tx-3"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ensemble.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fraud_detection.ensemble'`

- [ ] **Step 3: Implement ensemble.py**

Create `src/fraud_detection/ensemble.py`:
```python
def combine(
    transaction_id: str,
    if_score: float,
    if_flagged: bool,
    ae_score: float,
    ae_flagged: bool,
) -> dict:
    return {
        "transaction_id": transaction_id,
        "isolation_forest": {"score": if_score, "flagged": if_flagged},
        "autoencoder": {"score": ae_score, "flagged": ae_flagged},
        "ensemble_flagged": if_flagged or ae_flagged,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ensemble.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fraud_detection/ensemble.py tests/test_ensemble.py
git commit -m "Add ensemble combination logic"
```

---

### Task 5: Offline trainer

**Files:**
- Create: `src/fraud_detection/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `fit_scaler(train_df) -> StandardScaler`, `transform(df, scaler) -> np.ndarray`, `save_scaler(scaler, path)` from `fraud_detection.preprocessing` (Task 1). `IsolationForestModel` (`fit`, `score`, `set_threshold`, `save`) from `fraud_detection.models.isolation_forest` (Task 2). `AutoencoderModel(input_dim: int)` (`fit`, `score`, `set_threshold`, `save`) from `fraud_detection.models.autoencoder` (Task 3). `MODELS_DIR`, `RAW_DATA_PATH` from `fraud_detection.config` (Task 1).
- Produces: `train(data_path=RAW_DATA_PATH, models_dir=MODELS_DIR) -> dict` — trains both models, writes `scaler.joblib`, `isolation_forest.joblib`, `autoencoder.joblib`, `test_holdout.csv` (the held-out test-split rows, for later replay), and `metrics.json` into `models_dir`, and returns the metrics dict of the form `{"isolation_forest": {"precision": float, "recall": float, "auc": float, "avg_latency_ms": float}, "autoencoder": {...}}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_train.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fraud_detection.train'`

- [ ] **Step 3: Implement train.py**

Create `src/fraud_detection/train.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/fraud_detection/train.py tests/test_train.py
git commit -m "Add offline trainer producing both models and metrics.json"
```

---

### Task 6: Redis Streams producer

**Files:**
- Create: `src/fraud_detection/replay_producer.py`
- Test: `tests/test_replay_producer.py`

**Interfaces:**
- Consumes: `TRANSACTIONS_STREAM`, `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB` from `fraud_detection.config` (Task 1). `wait_for_redis(redis_client, max_retries=10, initial_delay=1.0) -> None` from `fraud_detection.redis_utils` (Task 1).
- Produces: `load_transactions(csv_path) -> pd.DataFrame` (reads CSV, sorts by `Time`, resets index). `publish_transactions(df: pd.DataFrame, redis_client, stream_name: str = TRANSACTIONS_STREAM, delay_seconds: float = 0.5) -> None` — for each row (in `df` order), publishes to the Redis Stream via `xadd` a message with fields `{"transaction_id": str(row_index), "data": json.dumps(row.to_dict())}`, sleeping `delay_seconds` between rows.

- [ ] **Step 1: Write the failing test**

Create `tests/test_replay_producer.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_replay_producer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fraud_detection.replay_producer'`

- [ ] **Step 3: Implement replay_producer.py**

Create `src/fraud_detection/replay_producer.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_replay_producer.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/fraud_detection/replay_producer.py tests/test_replay_producer.py
git commit -m "Add Redis Streams producer for transaction replay"
```

---

### Task 7: Scorer service

**Files:**
- Create: `src/fraud_detection/scorer_service.py`
- Test: `tests/test_scorer_service.py`

**Interfaces:**
- Consumes: `load_scaler(path) -> StandardScaler`, `transform(df, scaler) -> np.ndarray` from `fraud_detection.preprocessing` (Task 1). `IsolationForestModel.load(path)` and `AutoencoderModel.load(path)`, each exposing `.score(X: np.ndarray) -> np.ndarray` and `.flag(X: np.ndarray) -> np.ndarray[bool]`, from Tasks 2/3. `combine(transaction_id, if_score, if_flagged, ae_score, ae_flagged) -> dict` from `fraud_detection.ensemble` (Task 4). `MODELS_DIR`, `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `TRANSACTIONS_STREAM`, `TRANSACTIONS_CONSUMER_GROUP`, `SCORED_CHANNEL` from `fraud_detection.config` (Task 1). `wait_for_redis(redis_client, max_retries=10, initial_delay=1.0) -> None` from `fraud_detection.redis_utils` (Task 1).
- Produces: `load_models(models_dir=MODELS_DIR) -> tuple[StandardScaler, IsolationForestModel, AutoencoderModel]`. `score_transaction(raw_data: dict, scaler, if_model, ae_model) -> dict` returning `{"if_score": float, "if_flagged": bool, "ae_score": float, "ae_flagged": bool}`. `ensure_group(redis_client, stream_name: str, group_name: str) -> None`. `read_and_handle_batch(redis_client, scaler, if_model, ae_model, stream_name, group_name, consumer_name, channel_name, block_ms=5000, count=10) -> int` (returns number of messages processed). `run_forever(redis_client, scaler, if_model, ae_model, stream_name=TRANSACTIONS_STREAM, group_name=TRANSACTIONS_CONSUMER_GROUP, consumer_name="scorer-1", channel_name=SCORED_CHANNEL, block_ms=5000, count=10) -> None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scorer_service.py`:
```python
import json

import fakeredis
import numpy as np

from fraud_detection.scorer_service import ensure_group, read_and_handle_batch, score_transaction


class _StubScaler:
    def transform(self, df):
        return df[["Amount", "Time"]].to_numpy()


class _StubModel:
    def __init__(self, score_value, flagged_value):
        self.score_value = score_value
        self.flagged_value = flagged_value

    def score(self, X):
        return np.full(X.shape[0], self.score_value)

    def flag(self, X):
        return np.full(X.shape[0], self.flagged_value)


def _sample_raw_transaction():
    row = {f"V{i}": 0.0 for i in range(1, 29)}
    row["Amount"] = 10.0
    row["Time"] = 100.0
    return row


def test_score_transaction_returns_expected_fields():
    result = score_transaction(
        _sample_raw_transaction(), _StubScaler(), _StubModel(0.9, True), _StubModel(0.2, False)
    )
    assert result == {"if_score": 0.9, "if_flagged": True, "ae_score": 0.2, "ae_flagged": False}


def test_read_and_handle_batch_publishes_combined_result():
    server = fakeredis.FakeServer()
    client = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
    subscriber = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
    pubsub = subscriber.pubsub()
    pubsub.subscribe("scored_transactions")
    pubsub.get_message(timeout=1)

    ensure_group(client, "transactions", "scorer-group")
    payload = {"transaction_id": "tx-1", "data": json.dumps(_sample_raw_transaction())}
    client.xadd("transactions", payload)

    processed = read_and_handle_batch(
        client, _StubScaler(), _StubModel(0.9, True), _StubModel(0.2, False),
        stream_name="transactions", group_name="scorer-group",
        consumer_name="scorer-1", channel_name="scored_transactions",
        block_ms=100, count=10,
    )

    assert processed == 1
    message = pubsub.get_message(timeout=1)
    scored = json.loads(message["data"])
    assert scored["transaction_id"] == "tx-1"
    assert scored["ensemble_flagged"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scorer_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fraud_detection.scorer_service'`

- [ ] **Step 3: Implement scorer_service.py**

Create `src/fraud_detection/scorer_service.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scorer_service.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fraud_detection/scorer_service.py tests/test_scorer_service.py
git commit -m "Add scorer service consuming the transaction stream"
```

---

### Task 8: Producer-to-scorer integration test

**Files:**
- Test: `tests/test_pipeline_integration.py`

**Interfaces:**
- Consumes: `train(data_path, models_dir) -> dict` from `fraud_detection.train` (Task 5). `load_transactions(csv_path) -> pd.DataFrame`, `publish_transactions(df, redis_client, stream_name, delay_seconds)` from `fraud_detection.replay_producer` (Task 6). `load_models(models_dir) -> tuple`, `ensure_group(redis_client, stream_name, group_name)`, `read_and_handle_batch(redis_client, scaler, if_model, ae_model, stream_name, group_name, consumer_name, channel_name, block_ms, count) -> int` from `fraud_detection.scorer_service` (Task 7).

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline_integration.py`:
```python
import json

import fakeredis

from fraud_detection.replay_producer import load_transactions, publish_transactions
from fraud_detection.scorer_service import ensure_group, load_models, read_and_handle_batch
from fraud_detection.train import train


def test_producer_to_scorer_round_trip(sample_csv_path, tmp_path):
    models_dir = tmp_path / "models"
    train(data_path=sample_csv_path, models_dir=models_dir)
    scaler, if_model, ae_model = load_models(models_dir)

    df = load_transactions(sample_csv_path).head(3)

    server = fakeredis.FakeServer()
    client = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
    subscriber = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
    pubsub = subscriber.pubsub()
    pubsub.subscribe("scored_transactions")
    pubsub.get_message(timeout=1)

    publish_transactions(df, client, stream_name="transactions", delay_seconds=0)
    ensure_group(client, "transactions", "scorer-group")
    processed = read_and_handle_batch(
        client, scaler, if_model, ae_model,
        stream_name="transactions", group_name="scorer-group",
        consumer_name="scorer-1", channel_name="scored_transactions",
        block_ms=100, count=10,
    )

    assert processed == 3
    for _ in range(3):
        message = pubsub.get_message(timeout=1)
        scored = json.loads(message["data"])
        assert set(scored.keys()) == {
            "transaction_id", "isolation_forest", "autoencoder", "ensemble_flagged",
        }
        assert 0.0 <= scored["isolation_forest"]["score"] <= 1.0
        assert 0.0 <= scored["autoencoder"]["score"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails or passes for the wrong reason**

Run: `pytest tests/test_pipeline_integration.py -v`
Expected: Should already PASS if Tasks 5-7 are correctly implemented, since this test only composes existing functions. If it fails, the failure points at an integration mismatch between `train`'s output and `scorer_service`'s expectations — fix that mismatch before proceeding.

- [ ] **Step 3: Run full test suite to confirm no regressions**

Run: `pytest -v`
Expected: All tests PASS (preprocessing, isolation_forest, autoencoder, ensemble, train, replay_producer, scorer_service, pipeline_integration)

- [ ] **Step 4: Commit**

```bash
git add tests/test_pipeline_integration.py
git commit -m "Add producer-to-scorer round trip integration test"
```

---

### Task 9: FastAPI serving layer

**Files:**
- Create: `src/fraud_detection/api.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `load_models(models_dir) -> tuple[StandardScaler, IsolationForestModel, AutoencoderModel]`, `score_transaction(raw_data, scaler, if_model, ae_model) -> dict` (`{"if_score": float, "if_flagged": bool, "ae_score": float, "ae_flagged": bool}`) from `fraud_detection.scorer_service` (Task 7). `combine(transaction_id, if_score, if_flagged, ae_score, ae_flagged) -> dict` from `fraud_detection.ensemble` (Task 4). `MODELS_DIR`, `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `SCORED_CHANNEL`, `PCA_FEATURE_COLUMNS` from `fraud_detection.config` (Task 1).
- Produces: FastAPI `app` object with `POST /score` (body validated against `TransactionIn`, a pydantic model requiring `Time: float`, `Amount: float`, and every column in `PCA_FEATURE_COLUMNS` as `float`, plus optional `transaction_id: str | None`; returns 422 automatically on missing/invalid fields; on success returns the combined scored dict), `GET /models` (returns contents of `models/metrics.json`, 404 if absent), `WS /stream` (forwards messages published to `SCORED_CHANNEL` to the connected client as JSON text frames). Also produces `get_models_dir() -> Path` and `get_redis_client() -> redis.asyncio.Redis`, both overridable via `app.dependency_overrides` for testing, and `read_metrics(models_dir: Path) -> dict` (raises `FileNotFoundError` if `metrics.json` is missing).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_api.py`:
```python
import json

import fakeredis
import fakeredis.aioredis

from fraud_detection.api import app, get_models_dir, get_redis_client
from fraud_detection.config import SCORED_CHANNEL

from fastapi.testclient import TestClient


class _StubScaler:
    def transform(self, df):
        return df[["Amount", "Time"]].to_numpy()


class _StubModel:
    def __init__(self, score_value, flagged_value):
        self.score_value = score_value
        self.flagged_value = flagged_value

    def score(self, X):
        import numpy as np
        return np.full(X.shape[0], self.score_value)

    def flag(self, X):
        import numpy as np
        return np.full(X.shape[0], self.flagged_value)


def _sample_transaction():
    row = {f"V{i}": 0.0 for i in range(1, 29)}
    row["Amount"] = 10.0
    row["Time"] = 100.0
    return row


def test_score_endpoint_returns_combined_result():
    with TestClient(app) as client:
        app.state.scaler = _StubScaler()
        app.state.if_model = _StubModel(0.9, True)
        app.state.ae_model = _StubModel(0.2, False)

        response = client.post("/score", json=_sample_transaction())

        assert response.status_code == 200
        body = response.json()
        assert body["ensemble_flagged"] is True
        assert body["isolation_forest"]["score"] == 0.9
        assert body["autoencoder"]["score"] == 0.2


def test_score_endpoint_422_on_missing_field():
    with TestClient(app) as client:
        app.state.scaler = _StubScaler()
        app.state.if_model = _StubModel(0.9, True)
        app.state.ae_model = _StubModel(0.2, False)

        incomplete = _sample_transaction()
        del incomplete["V1"]
        response = client.post("/score", json=incomplete)

        assert response.status_code == 422


def test_models_endpoint_returns_metrics(tmp_path):
    metrics = {"isolation_forest": {"auc": 0.9}, "autoencoder": {"auc": 0.85}}
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    app.dependency_overrides[get_models_dir] = lambda: tmp_path

    try:
        with TestClient(app) as client:
            response = client.get("/models")
            assert response.status_code == 200
            assert response.json() == metrics
    finally:
        app.dependency_overrides.clear()


def test_models_endpoint_404_when_missing(tmp_path):
    app.dependency_overrides[get_models_dir] = lambda: tmp_path

    try:
        with TestClient(app) as client:
            response = client.get("/models")
            assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_stream_forwards_scored_message():
    server = fakeredis.FakeServer()
    publisher = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
    app.dependency_overrides[get_redis_client] = lambda: fakeredis.aioredis.FakeRedis(
        server=server, decode_responses=True
    )

    try:
        with TestClient(app) as client:
            with client.websocket_connect("/stream") as websocket:
                publisher.publish(SCORED_CHANNEL, json.dumps({"transaction_id": "abc", "ensemble_flagged": True}))
                received = websocket.receive_text()
                assert json.loads(received)["transaction_id"] == "abc"
    finally:
        app.dependency_overrides.clear()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fraud_detection.api'`

- [ ] **Step 3: Implement api.py**

Create `src/fraud_detection/api.py`:
```python
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import redis.asyncio as aioredis
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import create_model

from fraud_detection.config import (
    MODELS_DIR,
    PCA_FEATURE_COLUMNS,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT,
    SCORED_CHANNEL,
)
from fraud_detection.ensemble import combine
from fraud_detection.scorer_service import load_models, score_transaction

TransactionIn = create_model(
    "TransactionIn",
    transaction_id=(Optional[str], None),
    Time=(float, ...),
    Amount=(float, ...),
    **{column: (float, ...) for column in PCA_FEATURE_COLUMNS},
)


def get_models_dir() -> Path:
    return MODELS_DIR


def get_redis_client() -> aioredis.Redis:
    return aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    models_dir = get_models_dir()
    if (models_dir / "scaler.joblib").exists():
        app.state.scaler, app.state.if_model, app.state.ae_model = load_models(models_dir)
    else:
        app.state.scaler, app.state.if_model, app.state.ae_model = None, None, None
    yield


app = FastAPI(lifespan=lifespan)


def read_metrics(models_dir: Path) -> dict:
    metrics_path = models_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics file at {metrics_path}")
    with open(metrics_path) as f:
        return json.load(f)


@app.post("/score")
def score_endpoint(transaction: TransactionIn):
    if app.state.scaler is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    raw_data = transaction.model_dump(exclude={"transaction_id"})
    result = score_transaction(raw_data, app.state.scaler, app.state.if_model, app.state.ae_model)
    return combine(
        transaction.transaction_id or "manual",
        result["if_score"], result["if_flagged"],
        result["ae_score"], result["ae_flagged"],
    )


@app.get("/models")
def models_endpoint(models_dir: Path = Depends(get_models_dir)):
    try:
        return read_metrics(models_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.websocket("/stream")
async def stream_endpoint(websocket: WebSocket, redis_client: aioredis.Redis = Depends(get_redis_client)):
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(SCORED_CHANNEL)
    await websocket.accept()
    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            await websocket.send_text(message["data"])
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(SCORED_CHANNEL)
        await pubsub.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fraud_detection/api.py tests/test_api.py
git commit -m "Add FastAPI serving layer with score, models, and stream endpoints"
```

---

### Task 10: Streamlit dashboard

**Files:**
- Create: `src/fraud_detection/dashboard.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `API_WS_URL` from `fraud_detection.config` (Task 1). Connects to the `WS /stream` endpoint from `fraud_detection.api` (Task 9); expects JSON text frames matching `combine()`'s output shape from `fraud_detection.ensemble` (Task 4): `{"transaction_id": str, "isolation_forest": {"score": float, "flagged": bool}, "autoencoder": {"score": float, "flagged": bool}, "ensemble_flagged": bool}`.
- Produces: `format_row(scored: dict) -> dict` (pure function, flattens a scored-transaction dict into display columns). `main() -> None` (Streamlit entry point — not unit tested, verified manually).

- [ ] **Step 1: Write the failing test**

Create `tests/test_dashboard.py`:
```python
from fraud_detection.dashboard import format_row


def test_format_row_flattens_scored_transaction():
    scored = {
        "transaction_id": "tx-1",
        "isolation_forest": {"score": 0.8765, "flagged": True},
        "autoencoder": {"score": 0.1234, "flagged": False},
        "ensemble_flagged": True,
    }
    row = format_row(scored)
    assert row == {
        "transaction_id": "tx-1",
        "if_score": 0.876,
        "if_flagged": True,
        "ae_score": 0.123,
        "ae_flagged": False,
        "ensemble_flagged": True,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dashboard.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fraud_detection.dashboard'`

- [ ] **Step 3: Implement dashboard.py**

Create `src/fraud_detection/dashboard.py`:
```python
import json

import streamlit as st
import websocket

from fraud_detection.config import API_WS_URL


def format_row(scored: dict) -> dict:
    return {
        "transaction_id": scored["transaction_id"],
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dashboard.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Manual verification**

With redis-server, `scorer_service`, `uvicorn fraud_detection.api:app`, and `replay_producer` all running (see Task 11 for exact commands), run `streamlit run src/fraud_detection/dashboard.py`, open the printed local URL, and confirm rows appear live with scores and flags updating as transactions stream in.

- [ ] **Step 6: Commit**

```bash
git add src/fraud_detection/dashboard.py tests/test_dashboard.py
git commit -m "Add Streamlit dashboard for live transaction feed"
```

---

### Task 11: README and end-to-end run instructions

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: All CLI entry points from prior tasks — `python -m fraud_detection.train`, `python -m fraud_detection.scorer_service`, `python -m fraud_detection.replay_producer --csv-path <path> --delay-seconds <n>`, `uvicorn fraud_detection.api:app`, `streamlit run src/fraud_detection/dashboard.py`.

- [ ] **Step 1: Write README.md**

Create `README.md`:
```markdown
# Fraud Detection API

A real-time fraud detection demo inspired by tools like Stripe Radar. Trains
an Isolation Forest and an Autoencoder on the Kaggle credit card fraud
dataset, then replays held-out transactions through a Redis Streams
producer/consumer pipeline, scores them live, and displays results in a
Streamlit dashboard.

## Setup

1. Create and activate a virtual environment, then install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Install and start Redis locally (macOS via Homebrew):
   ```bash
   brew install redis
   redis-server
   ```
3. Download the Kaggle "Credit Card Fraud Detection" dataset and place it at
   `data/creditcard.csv` (the `data/` directory is gitignored).

## Running the demo

1. Train both models (one-time, or whenever retraining):
   ```bash
   python -m fraud_detection.train
   ```
   This writes `models/scaler.joblib`, `models/isolation_forest.joblib`,
   `models/autoencoder.joblib`, `models/test_holdout.csv` (the held-out
   transactions used for the live replay), and `models/metrics.json`.

2. In separate terminals, start each piece of the pipeline, in this order:
   ```bash
   # Terminal 1 (if not already running as a background service)
   redis-server

   # Terminal 2
   python -m fraud_detection.scorer_service

   # Terminal 3
   uvicorn fraud_detection.api:app --port 8000

   # Terminal 4
   streamlit run src/fraud_detection/dashboard.py

   # Terminal 5
   python -m fraud_detection.replay_producer \
     --csv-path models/test_holdout.csv --delay-seconds 0.5
   ```

3. Open the Streamlit URL printed in Terminal 4's output to watch transactions
   flow in with live fraud scores.

## Testing

```bash
pytest -v
```

All tests use `fakeredis` and synthetic fixtures — no running Redis or
downloaded dataset is required to run the test suite.

## Architecture

See `docs/superpowers/specs/2026-07-04-fraud-detection-api-design.md` for the
full design rationale.
```

- [ ] **Step 2: Manual verification**

Follow the README's Setup and Running steps end-to-end with the real Kaggle
dataset, confirming: `train.py` completes and prints metrics with AUC above
0.5 for both models; all five processes start without error; the dashboard
shows transactions streaming in with scores and color-appropriate flags.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Add README with setup and end-to-end run instructions"
```
