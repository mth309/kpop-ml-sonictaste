"""Central configuration for SonicTaste K-Pop Classifier (STC)."""

from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent


def _resolve_dat_dir() -> Path:
    """Finds the sibling K-Pop DAT directory across supported folder names."""
    for name in ("KPop-DAT", "kpop-discography-analytics", "KPop App"):
        cand = BASE_DIR.parent / name
        if cand.exists():
            return cand
    return BASE_DIR.parent / "KPop App"


KPOP_APP_DIR = _resolve_dat_dir()
KPOP_DB_PATH = KPOP_APP_DIR / "kpop_ratings.db"

DATA_DIR = BASE_DIR / "data"
AUDIO_CACHE_DIR = DATA_DIR / "audio_cache"
AUDIO_OVERRIDES_DIR = DATA_DIR / "audio_overrides"
EMBEDDING_DIR = DATA_DIR / "embeddings"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"

# Primary audio foundation model (music-specific, 13 transformer layers x 768D)
MERT_MODEL_ID = "m-a-p/MERT-v1-95M"
MERT_SAMPLE_RATE = 24000
CLIP_SECONDS = 30.0

# Known artist name aliases for matching Deezer / iTunes preview results
ARTIST_ALIASES: Dict[str, List[str]] = {
    "i-dle": ["(g)i-dle", "g-i-dle", "gidle", "idle"],
    "(g)i-dle": ["i-dle", "g-i-dle", "gidle"],
    "rosé": ["rose"],
    "jennie": ["jennie", "jennie ruby jane"],
    "lisa": ["lisa", "lalisa"],
    "jisoo": ["jisoo"],
    "soyeon": ["soyeon", "jeon soyeon"],
    "jeon somi": ["somi", "jeon somi"],
    "yena": ["yena", "choi yena"],
    "le sserafim": ["le sserafim", "lesserafim"],
    "aespa": ["aespa"],
    "itzy": ["itzy"],
    "ive": ["ive"],
    "nmixx": ["nmixx"],
    "twice": ["twice"],
    "blackpink": ["blackpink"],
    "red velvet": ["red velvet"],
    "mamamoo": ["mamamoo"],
    "babymonster": ["babymonster", "baby monster"],
    "illit": ["illit"],
    "stayc": ["stayc"],
    "fromis_9": ["fromis_9", "fromis 9"],
    "oh my girl": ["oh my girl"],
    "kiss of life": ["kiss of life"],
    "triples": ["triples"],
}


def ensure_data_dirs() -> None:
    """Creates local runtime data directories if they do not exist."""
    for d in (DATA_DIR, AUDIO_CACHE_DIR, AUDIO_OVERRIDES_DIR, EMBEDDING_DIR, CHECKPOINT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def get_device():
    """Returns the best available PyTorch device (CUDA GPU if available, else CPU)."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
