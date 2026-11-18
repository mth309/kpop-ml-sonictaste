"""Classical scikit-learn baselines (DummyMean and regularized RidgeCV)."""

from typing import Optional

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

from dataset import pool_layers


class DummyMeanModel:
    """Predicts the training set mean rating for every track (sanity-check baseline)."""

    def __init__(self) -> None:
        self.mean_rating: float = 3.0

    def fit(self, embeddings: np.ndarray, ratings: np.ndarray) -> "DummyMeanModel":
        self.mean_rating = float(np.mean(ratings))
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return np.full((len(embeddings),), self.mean_rating, dtype=np.float32)


class RidgeBaselineModel:
    """Regularized linear regression over pooled MERT embeddings (zscore or centered L2)."""

    def __init__(
        self,
        layer_mode: str = "mean",
        normalize_mode: str = "zscore",
        layer_weights: Optional[np.ndarray] = None,
        alphas: Optional[np.ndarray] = None,
    ) -> None:
        self.layer_mode = layer_mode
        self.normalize_mode = normalize_mode
        self.layer_weights = np.asarray(layer_weights, dtype=np.float32) if layer_weights is not None else None
        if alphas is not None:
            self.alphas = alphas
        elif normalize_mode == "l2":
            self.alphas = np.logspace(-3, 4, 35)
        else:
            self.alphas = np.logspace(0, 5, 25)

        if normalize_mode == "l2":
            steps = [
                ("center", StandardScaler(with_mean=True, with_std=False)),
                ("l2", Normalizer(norm="l2")),
                ("ridge", RidgeCV(alphas=self.alphas)),
            ]
        else:
            steps = [
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=self.alphas)),
            ]
        self.pipeline = Pipeline(steps)
        self.chosen_alpha_: Optional[float] = None

    def _pool(self, embeddings: np.ndarray) -> np.ndarray:
        if self.layer_weights is not None and embeddings.ndim == 3:
            w = self.layer_weights.reshape(1, -1, 1)
            return (embeddings * w).sum(axis=1).astype(np.float32, copy=False)
        return pool_layers(embeddings, mode=self.layer_mode).astype(np.float32, copy=False)

    def fit(self, embeddings: np.ndarray, ratings: np.ndarray) -> "RidgeBaselineModel":
        X = self._pool(embeddings)
        self.pipeline.fit(X, ratings)
        self.chosen_alpha_ = float(self.pipeline.named_steps["ridge"].alpha_)
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        X = self._pool(embeddings)
        preds = self.pipeline.predict(X)
        return np.clip(preds, 1.0, 5.0).astype(np.float32)
