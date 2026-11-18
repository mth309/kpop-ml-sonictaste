# SonicTaste K-Pop Classifier

A local PyTorch + Hugging Face `transformers` audio taste predictor (**SonicTaste / STC**) that learns personal song preferences (`1.0–5.0★` and `>=4.0★` Hit probability) from 30-second studio audio previews. SonicTaste can run via CLI or as a managed HTTP sidecar inside the K-Pop DAT UI.

> **Prerequisite — Requires K-Pop DAT:** SonicTaste **cannot operate standalone** without the companion **[K-Pop DAT (Discography & Analytics Tracker)](https://github.com/mth309/kpop-discography-analytics)**. It reads your rated catalog, canonical deduplication rules, and soloist-to-group vocal links directly from `kpop_ratings.db` in strict **read-only** mode (`?mode=ro`).

---

## How It Works

1. **Frozen Audio Backbone (`m-a-p/MERT-v1-95M`):** Extracts 13 hidden-layer embeddings (`(13, 768)`) plus 5-second temporal windows (`(6, 768)`) from 30-second audio previews (Deezer / iTunes / local `data/audio_overrides/`) on CUDA GPU or CPU.
2. **Champion Ensemble (`60% RidgeCV(mid+5sStd) + 40% PyTorch LayerAttnLin`):**
   * **`RidgeCV (mid+5sStd)`:** Regularized linear regression over middle MERT layers (`3..8`) concatenated with 5-second temporal standard deviation (*hook contrast*).
   * **`PyTorch LayerAttentionLinear`:** Learns softmax attention across all 13 MERT layers with dual-head output (star rating + `>=4.0★` Hit probability).
   * **`Cosine k-NN (k=12, mid)`:** Provides "Sonic Twins" reference tracks from your library.
3. **Leakage-Free Validation:** Uses `GroupKFold` grouped by `vocal_group_key` (linking soloists to their parent groups) so the model cannot cheat by recognizing a vocalist's voice across train/validation splits.
4. **Fingerprinted Checkpoints:** Caches fitted models in `data/checkpoints/` keyed by a SHA-256 hash of canonical track IDs and ratings, automatically retraining (~2s) whenever your ratings change.

---

## Setup

> **Folder Layout:** Clone `KPop-DAT` and `SonicTaste` under the same parent folder so `kpop_ratings.db` resolves automatically.

```powershell
cd SonicTaste
python -m venv venv
.\venv\Scripts\Activate.ps1

# 1. Install PyTorch with CUDA first (adjust CUDA version for your GPU, or omit --index-url for CPU):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. Install remaining dependencies:
pip install -r requirements.txt
```

---

## Common Commands

### 1. Build / Sync Audio Embeddings
```powershell
# Extract 13-layer MERT embeddings for newly rated tracks:
python embedder.py --batch-size 8

# Extract 5-second Hook Contrast windows:
python embedder.py --windows
```
*(Or click **Sync & Retrain Model** inside the K-Pop DAT UI → **Library → Import / Export**.)*

### 2. Predict New Songs & Artists
```powershell
# Predict score for a song via 30s preview lookup:
python predict.py --query "IVE - Love Dive"

# Predict score for an artist based on their top 10 Deezer tracks:
python predict.py --artist "KiiiKiii" --top 10

# Predict score for a local audio file:
python predict.py --file "C:\path\to\song.mp3"
```

### 3. Evaluate Model Performance
```powershell
# 5-fold Artist-Grouped CV with 5s Hook Contrast:
python evaluate.py --windows

# Cold-start artist holdout report:
python evaluate.py --holdout "aespa"
```

### 4. Supplying Missing Audio Manually
If any tracks cannot be matched automatically via Deezer/iTunes, they are listed in `data/embeddings/unmatched_tracks.csv`. Drop a matching `.mp3`, `.m4a`, `.wav`, or `.flac` named `{track_id}.mp3` into `data/audio_overrides/` and re-run `embedder.py` (or click **Sync & Retrain Model** in K-Pop DAT).

---

## Running Tests

Unit tests use synthetic tensors and temporary directories (no GPU, network calls, or Hugging Face downloads required):

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```
