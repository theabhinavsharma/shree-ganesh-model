from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LogisticRegressionGD:
    learning_rate: float = 0.05
    epochs: int = 20
    l2: float = 1e-3
    batch_size: int = 8192
    seed: int = 42
    positive_class_weight: str = "balanced"
    coef_: np.ndarray | None = None
    intercept_: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        if x.ndim != 2:
            raise ValueError("x must be 2D")
        if y.ndim != 1:
            raise ValueError("y must be 1D")
        rows, cols = x.shape
        rng = np.random.default_rng(self.seed)
        self.coef_ = np.zeros(cols, dtype=np.float32)
        self.intercept_ = 0.0

        pos_weight = 1.0
        neg_weight = 1.0
        if self.positive_class_weight == "balanced":
            pos = float(y.sum())
            neg = float(len(y) - pos)
            if pos > 0 and neg > 0:
                pos_weight = neg / pos

        batch_size = max(256, min(self.batch_size, rows))
        indices = np.arange(rows)
        for _ in range(self.epochs):
            rng.shuffle(indices)
            for start in range(0, rows, batch_size):
                batch_idx = indices[start : start + batch_size]
                xb = x[batch_idx]
                yb = y[batch_idx].astype(np.float32)
                logits = xb @ self.coef_ + self.intercept_
                probs = _sigmoid(logits)
                sample_weight = np.where(yb > 0.5, pos_weight, neg_weight).astype(np.float32)
                error = (probs - yb) * sample_weight
                grad_w = (xb.T @ error) / len(batch_idx) + self.l2 * self.coef_
                grad_b = float(error.mean())
                self.coef_ -= self.learning_rate * grad_w
                self.intercept_ -= self.learning_rate * grad_b
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("model is not fit")
        logits = x @ self.coef_ + self.intercept_
        probs = _sigmoid(logits)
        return np.vstack([1.0 - probs, probs]).T

    def to_dict(self) -> dict[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "l2": self.l2,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "positive_class_weight": self.positive_class_weight,
            "coef": self.coef_.tolist() if self.coef_ is not None else [],
            "intercept": self.intercept_,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "LogisticRegressionGD":
        model = cls(
            learning_rate=float(payload["learning_rate"]),
            epochs=int(payload["epochs"]),
            l2=float(payload["l2"]),
            batch_size=int(payload["batch_size"]),
            seed=int(payload["seed"]),
            positive_class_weight=str(payload.get("positive_class_weight", "balanced")),
        )
        coef = np.asarray(payload.get("coef", []), dtype=np.float32)
        model.coef_ = coef if coef.size else None
        model.intercept_ = float(payload.get("intercept", 0.0))
        return model


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))

