"""Zero-training Cosine Similarity k-NN taste predictor & Sonic Twins explainer."""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from dataset import pool_layers


@dataclass
class SonicNeighbor:
    """A single nearest-neighbor track from the reference library."""

    artist_name: str
    track_title: str
    rating: float
    similarity: float


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


class CosineKNNModel:
    """Predicts ratings via cosine-similarity-weighted average of top-K nearest tracks."""

    def __init__(self, k: int = 12, layer_mode: str = "mean", power: float = 2.0) -> None:
        self.k = k
        self.layer_mode = layer_mode
        self.power = power
        self.ref_vectors: Optional[np.ndarray] = None
        self.ref_ratings: Optional[np.ndarray] = None
        self.ref_records: Optional[List[dict]] = None
        self.fallback_mean: float = 3.0

    def fit(
        self,
        embeddings: np.ndarray,
        ratings: np.ndarray,
        records: Optional[Sequence[dict]] = None,
    ) -> "CosineKNNModel":
        pooled = pool_layers(embeddings, mode=self.layer_mode)
        self.ref_vectors = _l2_normalize(pooled.astype(np.float32, copy=False))
        self.ref_ratings = np.asarray(ratings, dtype=np.float32)
        self.ref_records = list(records) if records is not None else None
        self.fallback_mean = float(np.mean(self.ref_ratings)) if len(self.ref_ratings) else 3.0
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if self.ref_vectors is None or self.ref_ratings is None:
            raise RuntimeError("CosineKNNModel must be fit() before calling predict()")

        q_vecs = _l2_normalize(pool_layers(embeddings, mode=self.layer_mode).astype(np.float32, copy=False))
        sims = q_vecs @ self.ref_vectors.T  # (M, N_ref)
        k_eff = min(self.k, sims.shape[1])

        top_idx = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
        row_idx = np.arange(sims.shape[0])[:, None]
        top_sims = sims[row_idx, top_idx]
        top_ratings = self.ref_ratings[top_idx]

        weights = np.power(np.clip(top_sims, 0.0, None), self.power)
        weight_sums = weights.sum(axis=1)

        preds = np.full(sims.shape[0], self.fallback_mean, dtype=np.float32)
        valid = weight_sums > 1e-8
        preds[valid] = (weights[valid] * top_ratings[valid]).sum(axis=1) / weight_sums[valid]
        return preds

    def explain(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        exclude_artist: Optional[str] = None,
    ) -> List[SonicNeighbor]:
        """Returns the top-K most sonically similar reference tracks for a single song."""
        if self.ref_vectors is None or self.ref_ratings is None or self.ref_records is None:
            raise RuntimeError("CosineKNNModel must be fit(..., records=...) before explain()")

        q_arr = query_embedding[None, ...] if query_embedding.ndim <= 2 else query_embedding[:1]
        q_vec = _l2_normalize(pool_layers(q_arr, mode=self.layer_mode).astype(np.float32, copy=False))[0]
        sims = self.ref_vectors @ q_vec

        excl = exclude_artist.strip().casefold() if exclude_artist else None
        best_idx = np.argsort(-sims)
        neighbors: List[SonicNeighbor] = []
        for idx in best_idx:
            rec = self.ref_records[int(idx)]
            if excl and (rec["artist_name"] or "").strip().casefold() == excl:
                continue
            neighbors.append(
                SonicNeighbor(
                    artist_name=rec["artist_name"],
                    track_title=rec["track_title"],
                    rating=float(self.ref_ratings[int(idx)]),
                    similarity=float(sims[int(idx)]),
                )
            )
            if len(neighbors) >= top_k:
                break
        return neighbors
