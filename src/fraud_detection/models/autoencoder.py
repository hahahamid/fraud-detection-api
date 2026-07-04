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
