"""Dataset alignment, fresh rating sync, and artist-grouped CV / holdout splitting."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold

from catalog import load_canonical_rated_tracks
from config import KPOP_DB_PATH
from embedder import DEFAULT_MANIFEST_PATH, DEFAULT_NPZ_PATH


@dataclass
class AlignedDataset:
    """In-memory dataset aligning MERT embeddings (N, 13, 768) with ratings and group keys."""

    track_ids: np.ndarray        # (N,) int64
    embeddings: np.ndarray       # (N, 13, 768) float32
    ratings: np.ndarray          # (N,) float32
    vocal_groups: np.ndarray     # (N,) str (for leakage-free GroupKFold)
    artists: np.ndarray          # (N,) str
    titles: np.ndarray           # (N,) str
    records: List[dict]          # Full per-track metadata dicts


def pool_layers(embeddings: np.ndarray, mode: str = "mean") -> np.ndarray:
    """Reduces (N, 13, 768) layer embeddings to (N, 768) for 2D classical/k-NN models.

    Modes:
        - 'mean': Uniform average across all 13 MERT layers.
        - 'upper': Average across upper layers 6..12 (higher-level musical semantics).
        - 'mid': Average across middle layers 3..8 (timbre, production, instrumentation).
        - 'last': Final transformer layer (layer 12).
    """
    if embeddings.ndim == 2:
        return embeddings
    if mode == "mean":
        return embeddings.mean(axis=1)
    if mode == "upper":
        return embeddings[:, 6:, :].mean(axis=1)
    if mode == "mid":
        return embeddings[:, 3:9, :].mean(axis=1)
    if mode == "last":
        return embeddings[:, -1, :]
    raise ValueError(f"Unknown layer pooling mode: {mode!r}")


def load_aligned_dataset(
    npz_path: Path = DEFAULT_NPZ_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    sync_from_db: bool = True,
    db_path: Path = KPOP_DB_PATH,
) -> AlignedDataset:
    """Loads cached embeddings and optionally syncs fresh ratings from kpop_ratings.db (read-only)."""
    if not npz_path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {npz_path}")

    with np.load(npz_path) as data:
        cached_ids = data["track_ids"]
        cached_embs = data["embeddings"]

    id_to_row: Dict[int, int] = {int(tid): idx for idx, tid in enumerate(cached_ids)}
    manifest: Dict[str, dict] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    aligned_indices: List[int] = []
    records: List[dict] = []

    if sync_from_db and db_path.exists():
        live_tracks = load_canonical_rated_tracks(db_path=db_path)
        for t in live_tracks:
            row_idx = id_to_row.get(t.track_id)
            if row_idx is None:
                continue
            aligned_indices.append(row_idx)
            records.append(
                {
                    "track_id": t.track_id,
                    "artist_id": t.artist_id,
                    "artist_name": t.artist_name,
                    "artist_type": t.artist_type,
                    "vocal_group_key": t.vocal_group_key,
                    "album_id": t.album_id,
                    "album_title": t.album_title,
                    "album_type": t.album_type,
                    "release_date": t.release_date,
                    "track_title": t.track_title,
                    "track_type": t.track_type,
                    "rating": float(t.rating),
                    "is_title_track": t.is_title_track,
                }
            )
    else:
        for tid in cached_ids:
            meta = manifest.get(str(int(tid)))
            if not meta or meta.get("rating") is None:
                continue
            aligned_indices.append(id_to_row[int(tid)])
            records.append(meta)

    idx_arr = np.array(aligned_indices, dtype=np.int64)
    return AlignedDataset(
        track_ids=np.array([r["track_id"] for r in records], dtype=np.int64),
        embeddings=cached_embs[idx_arr].astype(np.float32, copy=False),
        ratings=np.array([float(r["rating"]) for r in records], dtype=np.float32),
        vocal_groups=np.array([r["vocal_group_key"] for r in records]),
        artists=np.array([r["artist_name"] for r in records]),
        titles=np.array([r["track_title"] for r in records]),
        records=records,
    )


def iter_group_kfold_splits(
    dataset: AlignedDataset,
    n_splits: int = 5,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yields (train_idx, val_idx) grouped strictly by vocal_group_key."""
    unique_groups = len(np.unique(dataset.vocal_groups))
    actual_splits = min(n_splits, unique_groups)
    if actual_splits < 2:
        raise ValueError(f"Need at least 2 distinct vocal groups for CV, found {unique_groups}")
    gkf = GroupKFold(n_splits=actual_splits)
    dummy_x = np.zeros((len(dataset.track_ids), 1), dtype=np.float32)
    yield from gkf.split(dummy_x, dataset.ratings, groups=dataset.vocal_groups)


def make_holdout_split(
    dataset: AlignedDataset,
    holdout_artists: Sequence[str],
    isolate_vocal_groups: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Splits dataset into train/val by holding out named artists.

    When isolate_vocal_groups=True, any linked soloist/parent group sharing the same
    vocal_group_key as a held-out artist is also placed in validation to prevent leakage.
    """
    wanted = {a.strip().casefold() for a in holdout_artists if a.strip()}
    if not wanted:
        raise ValueError("holdout_artists must contain at least one artist name")

    artist_mask = np.array([a.strip().casefold() in wanted for a in dataset.artists], dtype=bool)
    if not artist_mask.any():
        available = sorted(set(dataset.artists.tolist()))
        raise ValueError(f"None of {sorted(wanted)} found in dataset. Sample artists: {available[:10]}")

    if isolate_vocal_groups:
        held_keys = set(dataset.vocal_groups[artist_mask].tolist())
        val_mask = np.isin(dataset.vocal_groups, list(held_keys))
    else:
        val_mask = artist_mask

    extra_artists = sorted(set(dataset.artists[val_mask & ~artist_mask].tolist()))
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    return train_idx, val_idx, extra_artists


def build_contrast_features(
    base_ds: AlignedDataset,
    win_ds: AlignedDataset,
) -> Tuple[AlignedDataset, np.ndarray, np.ndarray]:
    """Aligns 13-layer and 5s-window datasets by track_id and builds (mid + 5s std) features.

    Returns:
        (aligned_base_ds, X_win_aligned, X_contrast)
    """
    win_id_to_idx = {int(tid): i for i, tid in enumerate(win_ds.track_ids)}
    common_base_idx = [i for i, tid in enumerate(base_ds.track_ids) if int(tid) in win_id_to_idx]
    common_win_idx = [win_id_to_idx[int(base_ds.track_ids[i])] for i in common_base_idx]

    aligned_base = AlignedDataset(
        track_ids=base_ds.track_ids[common_base_idx],
        embeddings=base_ds.embeddings[common_base_idx],
        ratings=base_ds.ratings[common_base_idx],
        vocal_groups=base_ds.vocal_groups[common_base_idx],
        artists=base_ds.artists[common_base_idx],
        titles=base_ds.titles[common_base_idx],
        records=[base_ds.records[i] for i in common_base_idx],
    )
    X_win_aligned = win_ds.embeddings[common_win_idx]
    X_mid = pool_layers(aligned_base.embeddings, mode="mid")
    X_win_std = X_win_aligned.std(axis=1)
    X_contrast = np.concatenate([X_mid, X_win_std], axis=1).astype(np.float32, copy=False)
    return aligned_base, X_win_aligned, X_contrast
