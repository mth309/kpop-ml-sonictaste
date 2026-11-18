"""Live inference CLI: score a new song by preview query or local audio file."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
from typing import List, Optional, Tuple

import numpy as np

from audio import (
    _REJECT_PREVIEW_RE,
    _artist_matches,
    _get_http_session,
    _normalize_text,
    _search_deezer,
    _search_itunes,
    _strip_parens,
    decode_audio_to_tensor,
)
from catalog import kpop_db
from config import CHECKPOINT_DIR, MERT_SAMPLE_RATE, ensure_data_dirs
from dataset import AlignedDataset, build_contrast_features, load_aligned_dataset, pool_layers
from embedder import (
    DEFAULT_WINDOWS_NPZ_PATH,
    extract_layer_embeddings,
    extract_window_embeddings,
    load_mert_model,
)
from models.knn import CosineKNNModel
from models.sklearn_heads import RidgeBaselineModel
from models.torch_head import TorchTasteModel


@dataclass
class EnsemblePrediction:
    """Predictions from all 3 taste heads plus the fitted k-NN Sonic Twins explainer."""

    knn_preds: np.ndarray
    ridge_preds: np.ndarray
    torch_preds: np.ndarray
    hit_probs: np.ndarray
    blends: np.ndarray
    knn_model: CosineKNNModel
    used_contrast: bool = False


def _compute_catalog_fingerprint(dataset: AlignedDataset, has_windows: bool) -> str:
    """Computes a deterministic hash over track IDs, ratings, and feature mode."""
    h = hashlib.sha256()
    h.update(b"win=1;" if has_windows else b"win=0;")
    h.update(dataset.track_ids.tobytes())
    h.update(dataset.ratings.tobytes())
    return h.hexdigest()[:16]


def _load_or_fit_ensemble(
    dataset: AlignedDataset,
    win_ds: Optional[AlignedDataset] = None,
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> Tuple[CosineKNNModel, RidgeBaselineModel, TorchTasteModel, bool]:
    """Loads cached taste models if catalog fingerprint matches; otherwise fits and saves them."""
    has_windows = win_ds is not None and len(win_ds.track_ids) > 0
    if has_windows:
        aligned_ds, _, X_contrast = build_contrast_features(dataset, win_ds)
    else:
        aligned_ds = dataset
        X_contrast = None

    fingerprint = _compute_catalog_fingerprint(aligned_ds, has_windows=has_windows)
    meta_path = checkpoint_dir / "meta.json"
    knn_path = checkpoint_dir / "knn_mid.pkl"
    ridge_path = checkpoint_dir / "ridge_contrast.pkl"
    torch_path = checkpoint_dir / "torch_linear.pt"

    if meta_path.exists() and knn_path.exists() and ridge_path.exists() and torch_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("fingerprint") == fingerprint:
                with knn_path.open("rb") as f:
                    knn = pickle.load(f)
                with ridge_path.open("rb") as f:
                    ridge = pickle.load(f)
                torch_model = TorchTasteModel.load(torch_path)
                print(f"[Checkpoint] Loaded cached ensemble ({len(aligned_ds.track_ids)} tracks).")
                return knn, ridge, torch_model, bool(meta.get("has_windows", has_windows))
        except Exception:
            pass

    knn = CosineKNNModel(k=12, layer_mode="mid").fit(
        aligned_ds.embeddings, aligned_ds.ratings, records=aligned_ds.records
    )
    if has_windows and X_contrast is not None:
        ridge = RidgeBaselineModel(layer_mode="mean").fit(X_contrast, aligned_ds.ratings)
    else:
        ridge = RidgeBaselineModel(layer_mode="mid").fit(aligned_ds.embeddings, aligned_ds.ratings)
    torch_model = TorchTasteModel(arch="linear").fit(aligned_ds.embeddings, aligned_ds.ratings)

    ensure_data_dirs()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with knn_path.open("wb") as f:
        pickle.dump(knn, f)
    with ridge_path.open("wb") as f:
        pickle.dump(ridge, f)
    torch_model.save(torch_path)
    meta_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "n_tracks": int(len(aligned_ds.track_ids)),
                "has_windows": has_windows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Checkpoint] Fitted & saved ensemble ({len(aligned_ds.track_ids)} tracks).")
    return knn, ridge, torch_model, has_windows


def _predict_with_ensemble(
    dataset: AlignedDataset,
    query_embs: np.ndarray,
    query_win_embs: Optional[np.ndarray] = None,
    win_ds: Optional[AlignedDataset] = None,
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> EnsemblePrediction:
    """Loads (or fits & caches) the 3 taste heads and scores query embeddings."""
    if win_ds is None and query_win_embs is not None and DEFAULT_WINDOWS_NPZ_PATH.exists():
        win_ds = load_aligned_dataset(npz_path=DEFAULT_WINDOWS_NPZ_PATH, sync_from_db=True)

    use_win = win_ds is not None and query_win_embs is not None
    knn, ridge, torch_model, used_contrast = _load_or_fit_ensemble(
        dataset,
        win_ds=win_ds if use_win else None,
        checkpoint_dir=checkpoint_dir,
    )

    knn_preds = knn.predict(query_embs)
    if used_contrast and query_win_embs is not None:
        q_mid = pool_layers(query_embs, mode="mid")
        q_std = query_win_embs.std(axis=1)
        q_ridge_in = np.concatenate([q_mid, q_std], axis=1).astype(np.float32, copy=False)
        ridge_preds = ridge.predict(q_ridge_in)
    else:
        ridge_preds = ridge.predict(query_embs)

    torch_preds, hit_probs = torch_model.predict_with_hit_prob(query_embs)
    blends = 0.60 * ridge_preds + 0.40 * torch_preds
    return EnsemblePrediction(
        knn_preds=knn_preds,
        ridge_preds=ridge_preds,
        torch_preds=torch_preds,
        hit_probs=hit_probs,
        blends=blends,
        knn_model=knn,
        used_contrast=used_contrast,
    )


def _resolve_query_to_temp_file(query: str, tmp_dir: Path) -> Path:
    if " - " in query:
        artist, title = query.split(" - ", 1)
    else:
        raise ValueError('Query must be in the format "Artist - Song Title"')

    hit = _search_deezer(artist.strip(), title.strip())
    ext = ".mp3"
    source = "Deezer"
    if not hit:
        hit = _search_itunes(artist.strip(), title.strip())
        ext = ".m4a"
        source = "iTunes"
    if not hit:
        raise RuntimeError(f"Could not find a 30s preview on Deezer or iTunes for: {query}")

    preview_url, matched_artist, matched_title = hit
    print(f"Matched [{source}]: {matched_artist} - {matched_title}")
    resp = _get_http_session().get(preview_url, timeout=15)
    resp.raise_for_status()
    out_path = tmp_dir / f"query{ext}"
    out_path.write_bytes(resp.content)
    return out_path


def score_audio(file_path: Optional[str] = None, query: Optional[str] = None) -> None:
    dataset = load_aligned_dataset(sync_from_db=True)
    print(f"Loaded {len(dataset.track_ids)} rated library tracks for comparison & training.")

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = Path(file_path) if file_path else _resolve_query_to_temp_file(query or "", Path(tmp))
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform = decode_audio_to_tensor(audio_path, target_sr=MERT_SAMPLE_RATE).numpy()

    processor, mert_model, device = load_mert_model()
    print(f"Extracting MERT-v1-95M embedding on {device}...")
    query_emb = extract_layer_embeddings([waveform], processor, mert_model, device)  # (1, 13, 768)
    query_win = extract_window_embeddings([waveform], processor, mert_model, device)  # (1, 6, 768)

    ens = _predict_with_ensemble(dataset, query_emb, query_win_embs=query_win)
    knn_pred = float(ens.knn_preds[0])
    ridge_pred = float(ens.ridge_preds[0])
    torch_pred = float(ens.torch_preds[0])
    hit_prob = float(ens.hit_probs[0])
    blend_pred = float(ens.blends[0])
    ridge_label = "RidgeCV (mid+5sStd)" if ens.used_contrast else "RidgeCV (mid)"

    print("\n=== Taste Prediction ===")
    print(f"  Cosine k-NN (k=12, mid) : {knn_pred:.2f}*")
    print(f"  {ridge_label:<24}: {ridge_pred:.2f}*")
    print(f"  PyTorch LayerAttnLin    : {torch_pred:.2f}*  (Hit prob >=4.0*: {hit_prob*100:.0f}%)")
    print(f"  Ensemble (60R + 40T)    : {blend_pred:.2f}*  ({'Likely Hit >=4.0*' if blend_pred >= 4.0 else 'Below Hit Threshold'})")

    print("\n--- Top 5 Sonic Twins in Your Library ---")
    for nb in ens.knn_model.explain(query_emb[0], top_k=5):
        print(f"  {nb.similarity:.3f} sim | {nb.rating:.1f}* | {nb.artist_name} - {nb.track_title}")


def _fetch_artist_top_tracks_deezer(
    artist_query: str,
    top_n: int = 5,
    timeout: float = 10.0,
) -> Tuple[str, List[Tuple[str, str]]]:
    """Fetches an artist's top studio tracks and 30s preview URLs from Deezer."""
    session = _get_http_session()
    resp = session.get(
        "https://api.deezer.com/search/artist",
        params={"q": artist_query, "limit": 5},
        timeout=timeout,
    )
    resp.raise_for_status()
    candidates = resp.json().get("data", [])
    if not candidates:
        raise RuntimeError(f"Artist not found on Deezer: {artist_query}")

    # Prefer exact normalized name match, then higher Deezer fan count
    exact_norm = _normalize_text(artist_query)
    ordered = sorted(
        candidates,
        key=lambda c: (
            _normalize_text(c.get("name", "")) == exact_norm,
            _artist_matches(artist_query, c.get("name", "")),
            int(c.get("nb_fan") or 0),
        ),
        reverse=True,
    )

    for chosen in ordered:
        artist_id = chosen["id"]
        artist_name = chosen.get("name", artist_query)

        top_resp = session.get(
            f"https://api.deezer.com/artist/{artist_id}/top",
            params={"limit": max(top_n * 3, 15)},
            timeout=timeout,
        )
        if top_resp.status_code != 200:
            continue
        items = top_resp.json().get("data", [])

        # Prioritize original-language tracks ahead of alternate-language editions
        valid_items = [
            it for it in items
            if it.get("title") and it.get("preview") and not _REJECT_PREVIEW_RE.search(it["title"])
        ]
        ordered_items = sorted(valid_items, key=lambda it: bool(kpop_db.is_alt_lang_text(it["title"])))

        seen_titles = set()
        tracks: List[Tuple[str, str]] = []
        for item in ordered_items:
            title = item["title"]
            preview = item["preview"]
            norm_base = _normalize_text(_strip_parens(title))
            if not norm_base or norm_base in seen_titles:
                continue
            seen_titles.add(norm_base)
            tracks.append((title, preview))
            if len(tracks) >= top_n:
                break

        if tracks:
            return artist_name, tracks

    raise RuntimeError(f"No playable 30s previews found on Deezer for artist: {artist_query}")


def predict_artist_top_tracks_data(
    artist_query: str,
    top_n: int = 5,
    mert_bundle: Optional[Tuple] = None,
    dataset: Optional[AlignedDataset] = None,
) -> dict:
    """Fetches an artist's top-N Deezer tracks and returns structured taste predictions."""
    ds = dataset if dataset is not None else load_aligned_dataset(sync_from_db=True)
    matched_artist, top_tracks = _fetch_artist_top_tracks_deezer(artist_query, top_n=top_n)

    waveforms = []
    valid_titles: List[str] = []
    session = _get_http_session()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for idx, (title, url) in enumerate(top_tracks):
            try:
                dl = session.get(url, timeout=15)
                if dl.status_code != 200 or len(dl.content) < 2048:
                    continue
                out_path = tmp_dir / f"track_{idx}.mp3"
                out_path.write_bytes(dl.content)
                waveforms.append(decode_audio_to_tensor(out_path, target_sr=MERT_SAMPLE_RATE).numpy())
                valid_titles.append(title)
            except Exception:
                continue

    if not waveforms:
        raise RuntimeError(f"Failed to download/decode previews for {matched_artist}")

    processor, mert_model, device = mert_bundle if mert_bundle is not None else load_mert_model()
    query_embs = extract_layer_embeddings(waveforms, processor, mert_model, device)
    query_wins = extract_window_embeddings(waveforms, processor, mert_model, device)

    ens = _predict_with_ensemble(ds, query_embs, query_win_embs=query_wins)
    order = sorted(range(len(valid_titles)), key=lambda i: float(ens.blends[i]), reverse=True)

    tracks_out = []
    for i in order:
        twins = (
            ens.knn_model.explain(query_embs[i], top_k=1, exclude_artist=matched_artist)
            or ens.knn_model.explain(query_embs[i], top_k=1)
        )
        twin = twins[0]
        tracks_out.append(
            {
                "title": valid_titles[i],
                "knn": round(float(ens.knn_preds[i]), 2),
                "ridge": round(float(ens.ridge_preds[i]), 2),
                "torch": round(float(ens.torch_preds[i]), 2),
                "hit_prob": round(float(ens.hit_probs[i]), 3),
                "blend": round(float(ens.blends[i]), 2),
                "twin": {
                    "artist": twin.artist_name,
                    "title": twin.track_title,
                    "rating": round(float(twin.rating), 1),
                    "similarity": round(float(twin.similarity), 3),
                },
            }
        )

    return {
        "matched_artist": matched_artist,
        "avg_score": round(float(ens.blends.mean()), 2),
        "hit_count": int((ens.blends >= 4.0).sum()),
        "total_tracks": len(valid_titles),
        "n_library_tracks": int(len(ds.track_ids)),
        "tracks": tracks_out,
    }


def score_artist_top_tracks(artist_query: str, top_n: int = 5) -> None:
    """Scores an artist's top-N popular tracks on Deezer to preview watchlist compatibility."""
    result = predict_artist_top_tracks_data(artist_query, top_n=top_n)
    matched_artist = result["matched_artist"]
    tracks = result["tracks"]
    print(f"Loaded {result['n_library_tracks']} rated library tracks.")
    print(f"Matched Artist [Deezer]: {matched_artist} ({len(tracks)} top tracks)")

    print(f"\n=== Watchlist Preview: {matched_artist} (Sorted by Predicted Score) ===")
    print(f"{'Track':<28} {'k-NN':>6} {'Ridge':>6} {'Torch':>6} {'Hit%':>5} {'Blend':>6}  {'Closest Sonic Twin'}")
    print("-" * 102)
    for t in tracks:
        twin = t["twin"]
        title_disp = (t["title"][:25] + "...") if len(t["title"]) > 28 else t["title"]
        print(
            f"{title_disp:<28} {t['knn']:>5.2f}* {t['ridge']:>5.2f}* "
            f"{t['torch']:>5.2f}* {t['hit_prob']*100:>4.0f}% {t['blend']:>5.2f}*  "
            f"{twin['similarity']:.3f} | {twin['rating']:.1f}* {twin['artist']} - {twin['title']}"
        )

    print("-" * 102)
    print(
        f"Estimated Top-{result['total_tracks']} Compatibility Avg: {result['avg_score']:.2f}* "
        f"({result['hit_count']}/{result['total_tracks']} predicted >=4.0* hits)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a new song or an artist's top tracks against your personal taste.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", type=str, help='Song query in format "Artist - Song Title"')
    group.add_argument("--file", type=str, help="Path to a local .mp3/.m4a/.wav/.flac file")
    group.add_argument("--artist", type=str, help="Artist name to fetch and score their top N Deezer tracks")
    parser.add_argument("--top", type=int, default=5, help="Number of top tracks to score with --artist (default: 5)")
    args = parser.parse_args()

    if args.artist:
        score_artist_top_tracks(args.artist, top_n=args.top)
    else:
        score_audio(file_path=args.file, query=args.query)


if __name__ == "__main__":
    main()
