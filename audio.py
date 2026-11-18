"""30-second studio preview resolver, local audio cache, and PyAV waveform decoder."""

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import re
from typing import Iterable, List, Optional, Tuple
import unicodedata

import numpy as np
import requests

from config import (
    ARTIST_ALIASES,
    AUDIO_CACHE_DIR,
    AUDIO_OVERRIDES_DIR,
    CLIP_SECONDS,
    MERT_SAMPLE_RATE,
    ensure_data_dirs,
)

# Reject non-canonical preview versions unless the source title itself asks for them
_REJECT_PREVIEW_RE = re.compile(
    r"\b(instrumental|inst\.?|karaoke|acapella|a\s*cappella|remix|sped\s*up|slowed|reverb|live\s+ver)\b",
    re.IGNORECASE,
)

_PAREN_RE = re.compile(r"[\(\[\{].*?[\)\]\}]")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_AUDIO_EXTS = (".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus")
_HTTP_SESSION: Optional[requests.Session] = None


def _get_http_session() -> requests.Session:
    """Returns a module-level requests.Session for keep-alive HTTP connection pooling."""
    global _HTTP_SESSION
    if _HTTP_SESSION is None:
        _HTTP_SESSION = requests.Session()
    return _HTTP_SESSION


@dataclass
class PreviewMatch:
    """Metadata for a resolved audio file or preview URL."""

    source: str  # "override", "cache", "deezer", "itunes"
    local_path: Path
    matched_artist: str = ""
    matched_title: str = ""
    preview_url: str = ""


def _normalize_text(text: str) -> str:
    """Normalizes Unicode accents, punctuation, and case for fuzzy matching."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _PUNCT_RE.sub(" ", text.casefold())
    return " ".join(text.split())


def _strip_parens(title: str) -> str:
    """Removes parenthetical suffixes like '(feat. ...)' or '(Prod. ...)' for search queries."""
    cleaned = _PAREN_RE.sub(" ", title or "")
    return " ".join(cleaned.split()).strip()


def _extract_paren_candidates(title: str) -> List[str]:
    """Extracts alternate English/romanized titles inside parentheses (e.g. '달라달라 (Dalla Dalla)')."""
    cands: List[str] = []
    for raw in _PAREN_RE.findall(title or ""):
        inner = raw[1:-1].strip()
        if not inner:
            continue
        low = inner.casefold()
        if low.startswith(("feat", "ft.", "prod", "with ", "ver.", "version")):
            continue
        if len(inner) >= 2:
            cands.append(inner)
    return cands


def _artist_matches(expected_artist: str, candidate_artist: str) -> bool:
    """Returns True if candidate_artist matches expected_artist or a known K-Pop alias."""
    exp_norm = _normalize_text(expected_artist)
    cand_norm = _normalize_text(candidate_artist)
    if not exp_norm or not cand_norm:
        return False
    if exp_norm in cand_norm or cand_norm in exp_norm:
        return True

    aliases = ARTIST_ALIASES.get(expected_artist.strip().casefold(), [])
    for alias in aliases:
        alias_norm = _normalize_text(alias)
        if alias_norm and (alias_norm in cand_norm or cand_norm in alias_norm):
            return True

    return SequenceMatcher(None, exp_norm, cand_norm).ratio() >= 0.80


def _title_similarity(expected_title: str, candidate_title: str) -> float:
    """Scores title similarity after normalizing and also comparing stripped-paren forms."""
    exp_full = _normalize_text(expected_title)
    cand_full = _normalize_text(candidate_title)
    if not exp_full or not cand_full:
        return 0.0
    if exp_full == cand_full:
        return 1.0

    exp_base = _normalize_text(_strip_parens(expected_title))
    cand_base = _normalize_text(_strip_parens(candidate_title))
    if exp_base and exp_base == cand_base:
        return 0.96

    best = max(
        SequenceMatcher(None, exp_full, cand_full).ratio(),
        SequenceMatcher(None, exp_base, cand_base).ratio() if exp_base and cand_base else 0.0,
    )
    for paren in _extract_paren_candidates(expected_title):
        p_norm = _normalize_text(paren)
        if not p_norm:
            continue
        if p_norm == cand_base or p_norm == cand_full:
            best = max(best, 0.95)
        elif cand_base:
            best = max(best, SequenceMatcher(None, p_norm, cand_base).ratio())
    return best


def _find_existing_local(directory: Path, track_id: int) -> Optional[Path]:
    """Finds an existing cached or override audio file for track_id."""
    for ext in _AUDIO_EXTS:
        candidate = directory / f"{track_id}{ext}"
        if candidate.exists() and candidate.stat().st_size > 1024:
            return candidate
    return None


def _select_best_preview(
    artist: str,
    title: str,
    candidates: Iterable[Tuple[str, str, str]],
) -> Optional[Tuple[str, str, str]]:
    """Selects the highest-similarity verified (preview_url, artist, title) candidate."""
    best_score = 0.0
    best_hit: Optional[Tuple[str, str, str]] = None
    reject_source = bool(_REJECT_PREVIEW_RE.search(title))
    for preview_url, cand_artist, cand_title in candidates:
        if not preview_url:
            continue
        if _REJECT_PREVIEW_RE.search(cand_title) and not reject_source:
            continue
        if not _artist_matches(artist, cand_artist):
            continue
        sim = _title_similarity(title, cand_title)
        if sim >= 0.72 and sim > best_score:
            best_score = sim
            best_hit = (preview_url, cand_artist, cand_title)
    return best_hit


def _search_deezer(artist: str, title: str, timeout: float = 8.0) -> Optional[Tuple[str, str, str]]:
    """Queries Deezer's free search API for a matching 30s MP3 preview URL."""
    clean_title = _strip_parens(title) or title
    queries = [
        f'artist:"{artist}" track:"{clean_title}"',
        f"{artist} {clean_title}",
    ]
    for paren in _extract_paren_candidates(title):
        queries.append(f"{artist} {paren}")
    session = _get_http_session()
    for q in queries:
        try:
            resp = session.get(
                "https://api.deezer.com/search",
                params={"q": q, "limit": 10},
                timeout=timeout,
            )
            if resp.status_code != 200:
                continue
            data = resp.json().get("data", [])
        except Exception:
            continue

        candidates = (
            (item.get("preview", ""), (item.get("artist") or {}).get("name", ""), item.get("title", ""))
            for item in data
        )
        best_hit = _select_best_preview(artist, title, candidates)
        if best_hit:
            return best_hit
    return None


def _search_itunes(artist: str, title: str, timeout: float = 8.0) -> Optional[Tuple[str, str, str]]:
    """Queries Apple iTunes' free search API for a matching 30s M4A preview URL."""
    clean_title = _strip_parens(title) or title
    terms = [f"{artist} {clean_title}"] + [f"{artist} {p}" for p in _extract_paren_candidates(title)]
    session = _get_http_session()

    for term in terms:
        try:
            resp = session.get(
                "https://itunes.apple.com/search",
                params={
                    "term": term,
                    "media": "music",
                    "entity": "song",
                    "limit": 10,
                    "country": "US",
                },
                timeout=timeout,
            )
            if resp.status_code != 200:
                continue
            results = resp.json().get("results", [])
        except Exception:
            continue

        candidates = (
            (item.get("previewUrl", ""), item.get("artistName", ""), item.get("trackName", ""))
            for item in results
        )
        best_hit = _select_best_preview(artist, title, candidates)
        if best_hit:
            return best_hit
    return None


def resolve_track_audio(track_id: int, artist_name: str, track_title: str) -> Optional[PreviewMatch]:
    """Resolves a track's audio file via override -> cache -> Deezer -> iTunes."""
    ensure_data_dirs()

    override_path = _find_existing_local(AUDIO_OVERRIDES_DIR, track_id)
    if override_path:
        return PreviewMatch(source="override", local_path=override_path)

    cached_path = _find_existing_local(AUDIO_CACHE_DIR, track_id)
    if cached_path:
        return PreviewMatch(source="cache", local_path=cached_path)

    # Try Deezer first (MP3), then iTunes (M4A)
    hit = _search_deezer(artist_name, track_title)
    source_name = "deezer"
    ext = ".mp3"
    if not hit:
        hit = _search_itunes(artist_name, track_title)
        source_name = "itunes"
        ext = ".m4a"

    if not hit:
        return None

    preview_url, matched_artist, matched_title = hit
    dest_path = AUDIO_CACHE_DIR / f"{track_id}{ext}"
    try:
        dl = _get_http_session().get(preview_url, timeout=15)
        if dl.status_code != 200 or len(dl.content) < 2048:
            return None
        dest_path.write_bytes(dl.content)
    except Exception:
        return None

    return PreviewMatch(
        source=source_name,
        local_path=dest_path,
        matched_artist=matched_artist,
        matched_title=matched_title,
        preview_url=preview_url,
    )


def decode_audio_to_tensor(
    audio_path: Path,
    target_sr: int = MERT_SAMPLE_RATE,
    clip_seconds: float = CLIP_SECONDS,
):
    """Decodes an audio file (.mp3, .m4a, .wav, .flac) into a normalized 1D float32 Tensor.

    Uses PyAV's embedded FFmpeg libraries (no system PATH ffmpeg required), resamples
    to mono at target_sr, and center-crops or zero-pads to exact clip length.
    """
    import av
    import torch

    resampler = av.AudioResampler(format="flt", layout="mono", rate=target_sr)
    chunks: List[np.ndarray] = []

    with av.open(str(audio_path)) as container:
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is None:
            raise ValueError(f"No audio stream found in {audio_path}")

        for frame in container.decode(audio_stream):
            for resampled in resampler.resample(frame):
                arr = resampled.to_ndarray()
                # Planar/mono float array -> flatten to 1D
                chunks.append(arr.reshape(-1).astype(np.float32, copy=False))

        # Flush resampler tail
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray().reshape(-1).astype(np.float32, copy=False))

    if not chunks:
        raise ValueError(f"Decoded 0 audio samples from {audio_path}")

    waveform = np.concatenate(chunks, axis=0)
    target_samples = int(target_sr * clip_seconds)

    if len(waveform) > target_samples:
        start = (len(waveform) - target_samples) // 2
        waveform = waveform[start : start + target_samples]
    elif len(waveform) < target_samples:
        pad = target_samples - len(waveform)
        waveform = np.pad(waveform, (0, pad), mode="constant")

    peak = float(np.max(np.abs(waveform)))
    if peak > 1e-6:
        waveform = waveform / peak

    return torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))
