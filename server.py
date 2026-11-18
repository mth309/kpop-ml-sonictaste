"""Lightweight local HTTP sidecar server for K-Pop DAT Watchlist predictions."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import sys
import threading
from typing import Optional, Tuple

from dataset import load_aligned_dataset
from embedder import (
    DEFAULT_WINDOWS_NPZ_PATH,
    build_embedding_cache,
    build_window_cache,
    load_mert_model,
)
from predict import _load_or_fit_ensemble, predict_artist_top_tracks_data

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_MERT_LOCK = threading.Lock()
_SYNC_LOCK = threading.Lock()
_SYNC_STATE_LOCK = threading.Lock()
_MERT_BUNDLE: Optional[Tuple] = None
_SYNC_STATE: dict = {
    "running": False,
    "status": "idle",
    "result": None,
    "error": None,
}


def _get_mert_bundle() -> Tuple:
    """Lazily loads MERT-v1-95M once and keeps it warm in VRAM/RAM."""
    global _MERT_BUNDLE
    with _MERT_LOCK:
        if _MERT_BUNDLE is None:
            _MERT_BUNDLE = load_mert_model()
        return _MERT_BUNDLE


def _run_sync_job() -> None:
    """Runs incremental embedding + ensemble refresh in a background thread."""
    try:
        bundle = _get_mert_bundle()
        emb_stats = build_embedding_cache(batch_size=8, mert_bundle=bundle)
        win_stats = build_window_cache(batch_size=16, mert_bundle=bundle)
        base_ds = load_aligned_dataset(sync_from_db=True)
        win_ds = (
            load_aligned_dataset(npz_path=DEFAULT_WINDOWS_NPZ_PATH, sync_from_db=True)
            if DEFAULT_WINDOWS_NPZ_PATH.exists()
            else None
        )
        _load_or_fit_ensemble(base_ds, win_ds=win_ds)
        result = {
            "status": "ok",
            "newly_embedded": emb_stats.get("newly_embedded", 0),
            "newly_windows": win_stats.get("newly_embedded", 0),
            "total_cached": len(base_ds.track_ids),
            "raw_cached": emb_stats.get("cached", len(base_ds.track_ids)),
            "unmatched_total": emb_stats.get("unmatched", 0),
            "newly_unmatched": emb_stats.get("newly_unmatched", []),
        }
        with _SYNC_STATE_LOCK:
            _SYNC_STATE.update({"running": False, "status": "ok", "result": result, "error": None})
    except Exception as e:
        with _SYNC_STATE_LOCK:
            _SYNC_STATE.update({"running": False, "status": "error", "result": None, "error": str(e)})
    finally:
        _SYNC_LOCK.release()


class _PredictHandler(BaseHTTPRequestHandler):
    """Handles /health, /sync, /sync/status, and POST /predict/artist requests from K-Pop DAT."""

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "model_loaded": _MERT_BUNDLE is not None})
            return
        if self.path == "/sync/status":
            with _SYNC_STATE_LOCK:
                snapshot = dict(_SYNC_STATE)
            self._send_json(200, snapshot)
            return
        self._send_json(404, {"detail": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/sync":
            if not _SYNC_LOCK.acquire(blocking=False):
                self._send_json(200, {"status": "already_running", "running": True})
                return
            with _SYNC_STATE_LOCK:
                _SYNC_STATE.update({"running": True, "status": "running", "result": None, "error": None})
            threading.Thread(target=_run_sync_job, daemon=True).start()
            self._send_json(202, {"status": "started", "running": True})
            return

        if self.path != "/predict/artist":
            self._send_json(404, {"detail": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send_json(400, {"detail": "Invalid JSON body"})
            return

        artist = str(data.get("artist", "")).strip()
        top_n = max(1, min(int(data.get("top_n", 10)), 15))
        if not artist:
            self._send_json(400, {"detail": "Missing 'artist' field"})
            return

        try:
            bundle = _get_mert_bundle()
            result = predict_artist_top_tracks_data(artist, top_n=top_n, mert_bundle=bundle)
            self._send_json(200, result)
        except RuntimeError as e:
            self._send_json(404, {"detail": str(e)})
        except Exception as e:
            self._send_json(500, {"detail": str(e)})

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        # Keep sidecar output quiet unless debugging
        pass


def main() -> None:
    port = int(os.environ.get("STC_SERVER_PORT") or os.environ.get("MTC_SERVER_PORT", "5001"))
    server = ThreadingHTTPServer(("127.0.0.1", port), _PredictHandler)
    # Warm MERT in background thread so /health is ready immediately
    threading.Thread(target=_get_mert_bundle, daemon=True).start()
    print(f"[STC Sidecar] Listening on http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
