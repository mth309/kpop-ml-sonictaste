"""Unit tests for audio matching, embedding cache I/O, and taste prediction models."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

STC_DIR = Path(__file__).resolve().parent.parent
if str(STC_DIR) not in sys.path:
    sys.path.insert(0, str(STC_DIR))

from audio import _REJECT_PREVIEW_RE, _artist_matches, _select_best_preview, _title_similarity  # noqa: E402
from dataset import AlignedDataset  # noqa: E402
from embedder import load_embedding_cache, save_embedding_cache  # noqa: E402
from models.knn import CosineKNNModel  # noqa: E402
from models.sklearn_heads import RidgeBaselineModel  # noqa: E402
from models.torch_head import TorchTasteModel  # noqa: E402
from predict import _predict_with_ensemble  # noqa: E402


class TestModelsAndAudio(unittest.TestCase):
    def test_audio_matching_aliases_bilingual_and_non_canon_rejection(self) -> None:
        self.assertTrue(_artist_matches("i-dle", "(G)I-DLE"))
        self.assertTrue(_artist_matches("Rosé", "ROSE"))
        self.assertGreaterEqual(_title_similarity("달라달라 (Dalla Dalla)", "Dalla Dalla"), 0.90)
        self.assertGreaterEqual(_title_similarity("뚜두뚜두 (DDU-DU DDU-DU)", "DDU-DU DDU-DU"), 0.90)
        self.assertIsNotNone(_REJECT_PREVIEW_RE.search("LOVE DIVE (Instrumental)"))
        self.assertIsNotNone(_REJECT_PREVIEW_RE.search("Supernova (Remix)"))
        self.assertIsNone(_REJECT_PREVIEW_RE.search("LOVE DIVE"))

    def test_embedding_cache_atomic_round_trip(self) -> None:
        rng = np.random.default_rng(7)
        vecs = {
            10: rng.normal(size=(13, 768)).astype(np.float32),
            20: rng.normal(size=(13, 768)).astype(np.float32),
        }
        manifest = {"10": {"track_id": 10, "rating": 5.0}, "20": {"track_id": 20, "rating": 2.0}}

        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "embs.npz"
            man_path = Path(tmp) / "manifest.json"
            save_embedding_cache(vecs, manifest, npz_path=npz_path, manifest_path=man_path)
            loaded_vecs, loaded_man = load_embedding_cache(npz_path=npz_path, manifest_path=man_path)

            self.assertEqual(sorted(loaded_vecs.keys()), [10, 20])
            np.testing.assert_allclose(loaded_vecs[10], vecs[10], atol=1e-6)
            self.assertEqual(loaded_man["20"]["rating"], 2.0)

    def test_knn_ridge_and_torch_models_on_synthetic_embeddings(self) -> None:
        rng = np.random.default_rng(42)
        n = 48
        # Construct linearly separable synthetic 13-layer embeddings
        signal = rng.normal(size=(n, 1, 1)).astype(np.float32)
        embs = rng.normal(scale=0.1, size=(n, 13, 768)).astype(np.float32)
        embs[:, :, :16] += signal
        ratings = np.clip(3.0 + signal[:, 0, 0] * 1.5, 1.0, 5.0).astype(np.float32)
        records = [{"artist_name": f"Art{i%4}", "track_title": f"Song{i}"} for i in range(n)]

        knn = CosineKNNModel(k=5, layer_mode="mid").fit(embs, ratings, records=records)
        knn_preds = knn.predict(embs[:6])
        self.assertEqual(knn_preds.shape, (6,))
        neighbors = knn.explain(embs[0], top_k=3)
        self.assertEqual(len(neighbors), 3)
        self.assertGreaterEqual(neighbors[0].similarity, neighbors[1].similarity)

        ridge = RidgeBaselineModel(layer_mode="mid").fit(embs, ratings)
        ridge_preds = ridge.predict(embs[:6])
        self.assertTrue(np.all((ridge_preds >= 1.0) & (ridge_preds <= 5.0)))

        for arch in ("linear", "mlp"):
            tm = TorchTasteModel(arch=arch, epochs=10, device=torch.device("cpu")).fit(embs, ratings)
            preds, probs = tm.predict_with_hit_prob(embs[:6])
            self.assertEqual(preds.shape, (6,))
            self.assertEqual(probs.shape, (6,))
            self.assertTrue(np.all((preds >= 1.0) & (preds <= 5.0)))
            self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))
            weights = tm.get_layer_weights()
            self.assertEqual(weights.shape, (13,))
            self.assertAlmostEqual(float(weights.sum()), 1.0, places=5)

            with tempfile.TemporaryDirectory() as tmp:
                ckpt = Path(tmp) / f"{arch}.pt"
                tm.save(ckpt)
                reloaded = TorchTasteModel.load(ckpt, device=torch.device("cpu"))
                np.testing.assert_allclose(reloaded.predict(embs[:6]), preds, atol=1e-5)

    def test_select_best_preview_and_custom_shape_checkpoint(self) -> None:
        candidates = [
            ("https://ex/inst.mp3", "IVE", "LOVE DIVE (Instrumental)"),
            ("https://ex/other.mp3", "OtherArtist", "LOVE DIVE"),
            ("https://ex/good.mp3", "IVE", "LOVE DIVE"),
        ]
        hit = _select_best_preview("IVE", "LOVE DIVE", candidates)
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], "https://ex/good.mp3")

        rng = np.random.default_rng(9)
        embs_small = rng.normal(size=(16, 5, 64)).astype(np.float32)
        ratings_small = rng.uniform(1.0, 5.0, size=(16,)).astype(np.float32)
        tm = TorchTasteModel(arch="linear", epochs=3, device=torch.device("cpu")).fit(embs_small, ratings_small)
        preds = tm.predict(embs_small[:4])
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "custom_shape.pt"
            tm.save(ckpt)
            reloaded = TorchTasteModel.load(ckpt, device=torch.device("cpu"))
            np.testing.assert_allclose(reloaded.predict(embs_small[:4]), preds, atol=1e-5)

            win_npz = Path(tmp) / "windows.npz"
            save_embedding_cache({1: embs_small[0]}, manifest=None, npz_path=win_npz, manifest_path=None)
            loaded_vecs, loaded_man = load_embedding_cache(npz_path=win_npz, manifest_path=None)
            self.assertEqual(loaded_man, {})
            np.testing.assert_allclose(loaded_vecs[1], embs_small[0], atol=1e-6)

    def test_ensemble_checkpoint_save_load_and_invalidation(self) -> None:
        rng = np.random.default_rng(11)
        n = 24
        tids = np.arange(100, 100 + n, dtype=np.int64)
        embs = rng.normal(size=(n, 13, 768)).astype(np.float32)
        wins = rng.normal(size=(n, 6, 768)).astype(np.float32)
        ratings = rng.uniform(1.0, 5.0, size=(n,)).astype(np.float32)
        records = [{"artist_name": f"Art{i%3}", "track_title": f"Track{i}"} for i in range(n)]

        base_ds = AlignedDataset(
            track_ids=tids,
            embeddings=embs,
            ratings=ratings.copy(),
            vocal_groups=np.array([f"vg_{i%3}" for i in range(n)]),
            artists=np.array([r["artist_name"] for r in records]),
            titles=np.array([r["track_title"] for r in records]),
            records=records,
        )
        win_ds = AlignedDataset(
            track_ids=tids,
            embeddings=wins,
            ratings=ratings.copy(),
            vocal_groups=base_ds.vocal_groups,
            artists=base_ds.artists,
            titles=base_ds.titles,
            records=records,
        )

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpts"
            ens1 = _predict_with_ensemble(
                base_ds, embs[:2], query_win_embs=wins[:2], win_ds=win_ds, checkpoint_dir=ckpt_dir
            )
            self.assertTrue(ens1.used_contrast)
            self.assertTrue((ckpt_dir / "meta.json").exists())
            self.assertTrue((ckpt_dir / "torch_linear.pt").exists())

            # Second call with same fingerprint loads cached models identically
            ens2 = _predict_with_ensemble(
                base_ds, embs[:2], query_win_embs=wins[:2], win_ds=win_ds, checkpoint_dir=ckpt_dir
            )
            np.testing.assert_allclose(ens1.blends, ens2.blends, atol=1e-5)

            # Mutating a rating invalidates the fingerprint and retrains
            base_ds.ratings[0] = 5.0 if base_ds.ratings[0] < 4.5 else 1.0
            ens3 = _predict_with_ensemble(
                base_ds, embs[:2], query_win_embs=wins[:2], win_ds=win_ds, checkpoint_dir=ckpt_dir
            )
            self.assertFalse(np.allclose(ens1.ridge_preds, ens3.ridge_preds, atol=1e-6))

    def test_sidecar_http_endpoints(self) -> None:
        from http.server import ThreadingHTTPServer
        import json
        import threading
        from unittest.mock import patch
        import urllib.error
        import urllib.request

        import server

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server._PredictHandler)
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                self.assertEqual(json.loads(resp.read().decode("utf-8"))["status"], "ok")

            fake_pred = {"matched_artist": "IVE", "avg_score": 4.2, "hit_count": 8, "total_tracks": 10, "tracks": []}
            with patch("server._get_mert_bundle", return_value=(None, None, None)), patch(
                "server.predict_artist_top_tracks_data", return_value=fake_pred
            ):
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/predict/artist",
                    data=json.dumps({"artist": "IVE", "top_n": 10}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    self.assertEqual(resp.status, 200)
                    self.assertEqual(json.loads(resp.read().decode("utf-8"))["matched_artist"], "IVE")

            with patch("server._get_mert_bundle", return_value=(None, None, None)), patch(
                "server.build_embedding_cache",
                return_value={"newly_embedded": 2, "cached": 100, "unmatched": 0, "newly_unmatched": []},
            ), patch("server.build_window_cache", return_value={"newly_embedded": 2, "cached": 100}), patch(
                "server.load_aligned_dataset", return_value=type("DS", (), {"track_ids": [1, 2]})()
            ), patch("server._load_or_fit_ensemble"):
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/sync", data=b"{}", headers={"Content-Type": "application/json"}, method="POST"
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    self.assertEqual(resp.status, 202)
                    start_body = json.loads(resp.read().decode("utf-8"))
                    self.assertEqual(start_body["status"], "started")
                    self.assertTrue(start_body["running"])

                status_body = {}
                for _ in range(50):
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/sync/status", timeout=5) as resp:
                        status_body = json.loads(resp.read().decode("utf-8"))
                    if not status_body.get("running"):
                        break
                    import time
                    time.sleep(0.02)

                self.assertFalse(status_body["running"])
                self.assertEqual(status_body["status"], "ok")
                self.assertEqual(status_body["result"]["newly_embedded"], 2)
                self.assertEqual(status_body["result"]["total_cached"], 2)

            bad_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/predict/artist",
                data=json.dumps({"artist": ""}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(bad_req, timeout=5)
            self.assertEqual(ctx.exception.code, 400)
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
