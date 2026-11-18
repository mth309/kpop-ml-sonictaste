"""GPU-accelerated audio embedding extractor using frozen MERT-v1-95M."""

import argparse
import csv
import json
from pathlib import Path
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from audio import _find_existing_local, decode_audio_to_tensor, resolve_track_audio
from catalog import TrackRecord, load_canonical_rated_tracks
from config import (
    AUDIO_OVERRIDES_DIR,
    EMBEDDING_DIR,
    MERT_MODEL_ID,
    MERT_SAMPLE_RATE,
    ensure_data_dirs,
    get_device,
)

DEFAULT_NPZ_PATH = EMBEDDING_DIR / "mert_v1_95m.npz"
DEFAULT_WINDOWS_NPZ_PATH = EMBEDDING_DIR / "mert_v1_95m_windows.npz"
DEFAULT_MANIFEST_PATH = EMBEDDING_DIR / "mert_v1_95m_manifest.json"
DEFAULT_UNMATCHED_PATH = EMBEDDING_DIR / "unmatched_tracks.csv"

_MERT_INFER_LOCK = threading.Lock()


def load_mert_model(
    model_id: str = MERT_MODEL_ID,
    device: Optional[torch.device] = None,
) -> Tuple[Wav2Vec2FeatureExtractor, AutoModel, torch.device]:
    """Loads the frozen MERT feature extractor and transformer onto the target device."""
    target_device = device or get_device()
    try:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_id, trust_remote_code=True, local_files_only=True
        )
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            output_hidden_states=True,
            local_files_only=True,
        )
    except OSError:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            output_hidden_states=True,
        )
    model.config.output_hidden_states = True
    model.to(target_device)
    model.eval()
    return processor, model, target_device


def extract_layer_embeddings(
    waveforms: List[np.ndarray],
    processor: Wav2Vec2FeatureExtractor,
    model: AutoModel,
    device: torch.device,
    sample_rate: int = MERT_SAMPLE_RATE,
) -> np.ndarray:
    """Runs frozen MERT inference on a batch of 1D waveforms.

    Uses PyTorch forward hooks on MERT's 12 transformer blocks (capturing the
    encoder input + all 12 block outputs) so all 13 hidden layers are extracted
    reliably regardless of Hugging Face transformers version.

    Returns:
        np.ndarray of shape (batch_size, 13, 768) in float32, mean-pooled over time frames.
    """
    inputs = processor(
        waveforms,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    layer_means: List[torch.Tensor] = []
    hooks = []

    def _capture_input_and_output(mod, inp, out):
        # Capture layer 0 input (post-projection + pos_conv_embed + layer_norm)
        x_in = inp[0]
        layer_means.append(x_in.mean(dim=1).float().cpu())
        x_out = out[0] if isinstance(out, tuple) else out
        layer_means.append(x_out.mean(dim=1).float().cpu())

    def _capture_output(mod, inp, out):
        x_out = out[0] if isinstance(out, tuple) else out
        layer_means.append(x_out.mean(dim=1).float().cpu())

    with _MERT_INFER_LOCK:
        for idx, layer in enumerate(model.encoder.layers):
            fn = _capture_input_and_output if idx == 0 else _capture_output
            hooks.append(layer.register_forward_hook(fn))

        try:
            use_amp = device.type == "cuda"
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                model(**inputs)
        finally:
            for h in hooks:
                h.remove()

    # layer_means has 13 tensors of shape (batch_size, 768) -> (batch_size, 13, 768)
    return torch.stack(layer_means, dim=1).numpy()


def extract_window_embeddings(
    waveforms: List[np.ndarray],
    processor: Wav2Vec2FeatureExtractor,
    model: AutoModel,
    device: torch.device,
    num_windows: int = 6,
    layer_indices: Tuple[int, ...] = (3, 4, 5, 6, 7, 8),
    sample_rate: int = MERT_SAMPLE_RATE,
) -> np.ndarray:
    """Extracts (batch_size, num_windows, 768) embeddings pooled over mid MERT layers."""
    inputs = processor(
        waveforms,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    window_accum = None
    hooks = []
    target_set = set(layer_indices)

    def _capture_windows(mod, inp, out):
        nonlocal window_accum
        x_out = out[0] if isinstance(out, tuple) else out  # (batch, time, 768)
        chunks = torch.tensor_split(x_out, num_windows, dim=1)
        win_tensor = torch.stack([c.mean(dim=1) for c in chunks], dim=1).float().cpu()
        window_accum = win_tensor if window_accum is None else (window_accum + win_tensor)

    # model.encoder.layers[i] outputs hidden layer i+1; layers[2..7] = hidden layers 3..8
    with _MERT_INFER_LOCK:
        for idx, layer in enumerate(model.encoder.layers):
            if (idx + 1) in target_set:
                hooks.append(layer.register_forward_hook(_capture_windows))

        try:
            use_amp = device.type == "cuda"
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                model(**inputs)
        finally:
            for h in hooks:
                h.remove()

    return (window_accum / len(target_set)).numpy()


def load_embedding_cache(
    npz_path: Path = DEFAULT_NPZ_PATH,
    manifest_path: Optional[Path] = DEFAULT_MANIFEST_PATH,
) -> Tuple[Dict[int, np.ndarray], Dict[str, dict]]:
    """Loads existing cached embeddings and optional metadata manifest from disk."""
    cached_vectors: Dict[int, np.ndarray] = {}
    manifest: Dict[str, dict] = {}

    if npz_path.exists():
        with np.load(npz_path) as data:
            track_ids = data["track_ids"]
            embeddings = data["embeddings"]
            for tid, emb in zip(track_ids, embeddings):
                cached_vectors[int(tid)] = emb.astype(np.float32, copy=False)

    if manifest_path is not None and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    return cached_vectors, manifest


def save_embedding_cache(
    cached_vectors: Dict[int, np.ndarray],
    manifest: Optional[Dict[str, dict]] = None,
    npz_path: Path = DEFAULT_NPZ_PATH,
    manifest_path: Optional[Path] = DEFAULT_MANIFEST_PATH,
    empty_shape: Tuple[int, ...] = (0, 13, 768),
) -> None:
    """Atomically saves cached embeddings (.npz) and optional metadata manifest (.json)."""
    ensure_data_dirs()
    ordered_ids = sorted(cached_vectors.keys())
    if ordered_ids:
        id_arr = np.array(ordered_ids, dtype=np.int64)
        emb_arr = np.stack([cached_vectors[tid] for tid in ordered_ids], axis=0).astype(np.float32)
    else:
        id_arr = np.empty((0,), dtype=np.int64)
        emb_arr = np.empty(empty_shape, dtype=np.float32)

    tmp_npz = npz_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_npz, track_ids=id_arr, embeddings=emb_arr)
    tmp_npz.replace(npz_path)

    if manifest is not None and manifest_path is not None:
        tmp_manifest = manifest_path.with_suffix(".tmp.json")
        tmp_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_manifest.replace(manifest_path)


def _write_unmatched_csv(unmatched: List[TrackRecord], path: Path = DEFAULT_UNMATCHED_PATH) -> None:
    """Writes unresolved tracks to CSV so missing previews can be supplied via audio_overrides/."""
    ensure_data_dirs()
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "artist_name", "album_title", "track_title", "rating"])
        for t in unmatched:
            writer.writerow([t.track_id, t.artist_name, t.album_title, t.track_title, t.rating])


def build_embedding_cache(
    tracks: Optional[List[TrackRecord]] = None,
    batch_size: int = 4,
    save_every: int = 80,
    limit: Optional[int] = None,
    mert_bundle: Optional[Tuple] = None,
    retry_unmatched: bool = False,
) -> Dict:
    """Incrementally fetches audio previews and computes MERT embeddings on GPU/CPU."""
    ensure_data_dirs()
    all_tracks = tracks if tracks is not None else load_canonical_rated_tracks()
    if limit is not None:
        all_tracks = all_tracks[:limit]

    prev_unmatched_ids = set()
    if DEFAULT_UNMATCHED_PATH.exists():
        with DEFAULT_UNMATCHED_PATH.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    prev_unmatched_ids.add(int(row["track_id"]))
                except (KeyError, ValueError):
                    pass

    cached_vectors, manifest = load_embedding_cache()
    all_pending = [t for t in all_tracks if t.track_id not in cached_vectors]
    skipped_unmatched: List[TrackRecord] = []
    if not retry_unmatched and prev_unmatched_ids:
        pending = []
        for t in all_pending:
            if t.track_id in prev_unmatched_ids and not _find_existing_local(AUDIO_OVERRIDES_DIR, t.track_id):
                skipped_unmatched.append(t)
            else:
                pending.append(t)
    else:
        pending = all_pending

    print(
        f"Total target tracks: {len(all_tracks)} | Already cached: {len(all_tracks) - len(all_pending)} "
        f"| Pending: {len(pending)}"
        + (f" | Skipped prior unmatched: {len(skipped_unmatched)}" if skipped_unmatched else "")
    )

    if not pending:
        return {
            "total": len(all_tracks),
            "cached": len(cached_vectors),
            "newly_embedded": 0,
            "unmatched": len(skipped_unmatched),
            "newly_unmatched": [],
        }

    if mert_bundle is not None:
        processor, model, device = mert_bundle
    else:
        device = get_device()
        gpu_label = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
        print(f"Loading {MERT_MODEL_ID} on {device} ({gpu_label})...")
        processor, model, device = load_mert_model(device=device)

    unmatched: List[TrackRecord] = []
    batch_items: List[Tuple[TrackRecord, str, np.ndarray]] = []
    newly_embedded = 0
    t0 = time.perf_counter()

    def flush_batch() -> None:
        nonlocal newly_embedded
        if not batch_items:
            return
        waves = [item[2] for item in batch_items]
        embs = extract_layer_embeddings(waves, processor, model, device)
        for (rec, src, _), vec in zip(batch_items, embs):
            cached_vectors[rec.track_id] = vec
            manifest[str(rec.track_id)] = {
                "track_id": rec.track_id,
                "artist_id": rec.artist_id,
                "artist_name": rec.artist_name,
                "artist_type": rec.artist_type,
                "vocal_group_key": rec.vocal_group_key,
                "album_id": rec.album_id,
                "album_title": rec.album_title,
                "album_type": rec.album_type,
                "release_date": rec.release_date,
                "track_title": rec.track_title,
                "track_type": rec.track_type,
                "rating": rec.rating,
                "is_title_track": rec.is_title_track,
                "audio_source": src,
            }
            newly_embedded += 1
        batch_items.clear()

    for rec in tqdm(pending, desc="Embedding tracks", unit="track"):
        match = resolve_track_audio(rec.track_id, rec.artist_name, rec.track_title)
        if not match:
            unmatched.append(rec)
            continue
        try:
            wave_tensor = decode_audio_to_tensor(match.local_path, target_sr=MERT_SAMPLE_RATE)
            batch_items.append((rec, match.source, wave_tensor.numpy()))
        except Exception as e:
            print(f"\n[WARN] Decode failed for {rec.artist_name} - {rec.track_title} ({rec.track_id}): {e}")
            unmatched.append(rec)
            continue

        if len(batch_items) >= batch_size:
            flush_batch()
            if newly_embedded % save_every == 0:
                save_embedding_cache(cached_vectors, manifest)

    flush_batch()
    save_embedding_cache(cached_vectors, manifest)
    all_unmatched = skipped_unmatched + unmatched
    if limit is None:
        _write_unmatched_csv(all_unmatched)

    elapsed = time.perf_counter() - t0
    print(
        f"\nDone in {elapsed:.1f}s | Newly embedded: {newly_embedded} | "
        f"Total cached: {len(cached_vectors)} | Unmatched: {len(all_unmatched)}"
    )
    if limit is None and all_unmatched:
        print(f"Unmatched track log written to: {DEFAULT_UNMATCHED_PATH}")

    newly_unmatched = [
        {"track_id": t.track_id, "artist": t.artist_name, "title": t.track_title}
        for t in unmatched
        if t.track_id not in prev_unmatched_ids
    ]

    return {
        "total": len(all_tracks),
        "cached": len(cached_vectors),
        "newly_embedded": newly_embedded,
        "unmatched": len(all_unmatched),
        "newly_unmatched": newly_unmatched,
    }


def build_window_cache(
    batch_size: int = 16,
    num_windows: int = 6,
    npz_path: Path = DEFAULT_WINDOWS_NPZ_PATH,
    mert_bundle: Optional[Tuple] = None,
) -> Dict[str, int]:
    """Extracts (N, 6, 768) 5-second window embeddings from locally cached audio on GPU."""
    from audio import _find_existing_local
    from config import AUDIO_CACHE_DIR, AUDIO_OVERRIDES_DIR

    ensure_data_dirs()
    base_vecs, _ = load_embedding_cache()
    existing_win, _ = load_embedding_cache(npz_path=npz_path, manifest_path=None)
    pending_ids = [tid for tid in sorted(base_vecs.keys()) if tid not in existing_win]

    print(f"Window cache: {len(existing_win)} cached | {len(pending_ids)} pending")
    if not pending_ids:
        return {"cached": len(existing_win), "newly_embedded": 0}

    if mert_bundle is not None:
        processor, model, device = mert_bundle
    else:
        device = get_device()
        gpu_label = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
        print(f"Loading {MERT_MODEL_ID} on {device} ({gpu_label}) for {num_windows}-window extraction...")
        processor, model, device = load_mert_model(device=device)

    batch_items: List[Tuple[int, np.ndarray]] = []
    newly_embedded = 0
    t0 = time.perf_counter()

    def flush_win_batch() -> None:
        nonlocal newly_embedded
        if not batch_items:
            return
        tids = [b[0] for b in batch_items]
        waves = [b[1] for b in batch_items]
        embs = extract_window_embeddings(waves, processor, model, device, num_windows=num_windows)
        for tid, vec in zip(tids, embs):
            existing_win[tid] = vec
            newly_embedded += 1
        batch_items.clear()

    for tid in tqdm(pending_ids, desc="Extracting 5s windows", unit="track"):
        local = _find_existing_local(AUDIO_OVERRIDES_DIR, tid) or _find_existing_local(AUDIO_CACHE_DIR, tid)
        if not local:
            continue
        try:
            wave = decode_audio_to_tensor(local, target_sr=MERT_SAMPLE_RATE).numpy()
            batch_items.append((tid, wave))
        except Exception:
            continue
        if len(batch_items) >= batch_size:
            flush_win_batch()

    flush_win_batch()
    save_embedding_cache(
        existing_win,
        manifest=None,
        npz_path=npz_path,
        manifest_path=None,
        empty_shape=(0, num_windows, 768),
    )

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s | Newly extracted: {newly_embedded} | Total window cache: {len(existing_win)}")
    return {"cached": len(existing_win), "newly_embedded": newly_embedded}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MERT-v1-95M audio embeddings on GPU/CPU.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of tracks to process.")
    parser.add_argument("--batch-size", type=int, default=4, help="GPU inference batch size (default: 4).")
    parser.add_argument("--windows", action="store_true", help="Extract (N, 6, 768) 5s window embeddings from local cache.")
    parser.add_argument("--retry-unmatched", action="store_true", help="Re-query Deezer/iTunes for previously unmatched tracks.")
    args = parser.parse_args()
    if args.windows:
        build_window_cache(batch_size=max(args.batch_size, 8))
    else:
        build_embedding_cache(batch_size=args.batch_size, limit=args.limit, retry_unmatched=args.retry_unmatched)


if __name__ == "__main__":
    main()
