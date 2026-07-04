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
