"""Unit tests for read-only catalog ingestion, canonical filtering, and GroupKFold isolation."""

from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

import numpy as np

STC_DIR = Path(__file__).resolve().parent.parent
if str(STC_DIR) not in sys.path:
    sys.path.insert(0, str(STC_DIR))

from catalog import load_canonical_rated_tracks  # noqa: E402
from config import KPOP_APP_DIR  # noqa: E402
from dataset import (  # noqa: E402
    AlignedDataset,
    iter_group_kfold_splits,
    make_holdout_split,
    pool_layers,
)


def _create_temp_kpop_db(db_path: Path) -> None:
    schema_sql = (KPOP_APP_DIR / "schema.sql").read_text(encoding="utf-8")
    conn = sqlite3.connect(str(db_path))
    conn.executescript(schema_sql)

    # Insert 2 groups and 1 soloist linked to Group A
    conn.execute("INSERT INTO artists (id, name, artist_type) VALUES (1, 'GroupA', 'group')")
    conn.execute("INSERT INTO artists (id, name, artist_type) VALUES (2, 'SoloA', 'solo')")
    conn.execute("INSERT INTO artists (id, name, artist_type) VALUES (3, 'GroupB', 'group')")
    conn.execute(
        "INSERT INTO artist_members (group_id, member_name, solo_artist_id) VALUES (1, 'SoloA', 2)"
    )

    conn.execute(
        "INSERT INTO albums (id, artist_id, title, album_type, release_date) VALUES (10, 1, 'Mini 1', 'EP', '2024-01-01')"
    )
    conn.execute(
        "INSERT INTO albums (id, artist_id, title, album_type, release_date) VALUES (20, 2, 'Solo 1', 'Single', '2024-06-01')"
    )
    conn.execute(
        "INSERT INTO albums (id, artist_id, title, album_type, release_date) VALUES (30, 3, 'Mini B', 'EP', '2024-03-01')"
    )

    # Canonical rated tracks + tracks that MUST be filtered out
    conn.execute(
        "INSERT INTO tracks (id, album_id, track_number, title, rating, track_type) VALUES (101, 10, 1, 'Title A', 5.0, 'core')"
    )
    conn.execute(
        "INSERT INTO tracks (id, album_id, track_number, title, rating, track_type) VALUES (102, 10, 2, 'Intro Skit', 2.0, 'non_canon')"
    )
    conn.execute(
        "INSERT INTO tracks (id, album_id, track_number, title, rating, track_type) VALUES (103, 10, 3, 'Title A (JP)', 4.0, 'alt_lang')"
    )
    conn.execute(
        "INSERT INTO tracks (id, album_id, track_number, title, rating, track_type) VALUES (104, 10, 4, 'Unrated B-Side', NULL, 'core')"
    )
    conn.execute(
        "INSERT INTO tracks (id, album_id, track_number, title, rating, track_type) VALUES (201, 20, 1, 'Solo Song', 4.0, 'core')"
    )
    conn.execute(
        "INSERT INTO tracks (id, album_id, track_number, title, rating, track_type) VALUES (301, 30, 1, 'Title B', 3.0, 'core')"
    )
    conn.commit()
    conn.close()


class TestCatalogAndSplits(unittest.TestCase):
    def test_canonical_filtering_and_vocal_group_linking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test_kpop.db"
            _create_temp_kpop_db(db_path)

            tracks = load_canonical_rated_tracks(db_path=db_path)
            ids = [t.track_id for t in tracks]
            self.assertEqual(ids, [101, 301, 201])

            by_id = {t.track_id: t for t in tracks}
            # GroupA and linked SoloA must share the exact same vocal_group_key
            self.assertEqual(by_id[101].vocal_group_key, by_id[201].vocal_group_key)
            self.assertNotEqual(by_id[101].vocal_group_key, by_id[301].vocal_group_key)

    def test_transitive_multi_group_vocal_linking(self) -> None:
        """Connected components via Union-Find must merge i-dle <-> Soyeon <-> K/DA regardless of ID order."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test_kpop_transitive.db"
            schema_sql = (KPOP_APP_DIR / "schema.sql").read_text(encoding="utf-8")
            conn = sqlite3.connect(str(db_path))
            conn.executescript(schema_sql)

            # Intentionally give soloist smallest ID and second group largest ID
            conn.execute("INSERT INTO artists (id, name, artist_type) VALUES (1, 'Soyeon', 'solo')")
            conn.execute("INSERT INTO artists (id, name, artist_type) VALUES (2, 'i-dle', 'group')")
            conn.execute("INSERT INTO artists (id, name, artist_type) VALUES (3, 'K/DA', 'group')")
            conn.execute("INSERT INTO artists (id, name, artist_type) VALUES (4, 'UnrelatedSolo', 'solo')")
            conn.execute("INSERT INTO artist_members (group_id, member_name, solo_artist_id) VALUES (2, 'Soyeon', 1)")
            conn.execute("INSERT INTO artist_members (group_id, member_name, solo_artist_id) VALUES (3, 'Soyeon', 1)")

            conn.execute(
                "INSERT INTO albums (id, artist_id, title, album_type, release_date) VALUES (10, 1, 'Windy', 'EP', '2021-07-05')"
            )
            conn.execute(
                "INSERT INTO albums (id, artist_id, title, album_type, release_date) VALUES (20, 2, 'I NEVER DIE', 'LP', '2022-03-14')"
            )
            conn.execute(
                "INSERT INTO albums (id, artist_id, title, album_type, release_date) VALUES (30, 3, 'ALL OUT', 'EP', '2020-11-06')"
            )
            conn.execute(
                "INSERT INTO albums (id, artist_id, title, album_type, release_date) VALUES (40, 4, 'Other', 'Single', '2023-01-01')"
            )

            conn.execute(
                "INSERT INTO tracks (id, album_id, track_number, title, rating, track_type) VALUES (101, 10, 1, 'BEAM BEAM', 4.5, 'core')"
            )
            conn.execute(
                "INSERT INTO tracks (id, album_id, track_number, title, rating, track_type) VALUES (201, 20, 1, 'TOMBOY', 5.0, 'core')"
            )
            conn.execute(
                "INSERT INTO tracks (id, album_id, track_number, title, rating, track_type) VALUES (301, 30, 1, 'MORE', 4.5, 'core')"
            )
            conn.execute(
                "INSERT INTO tracks (id, album_id, track_number, title, rating, track_type) VALUES (401, 40, 1, 'Solo Track', 3.5, 'core')"
            )
            conn.commit()
            conn.close()

            tracks = load_canonical_rated_tracks(db_path=db_path)
            by_id = {t.track_id: t for t in tracks}

            self.assertEqual(by_id[101].vocal_group_key, by_id[201].vocal_group_key)
            self.assertEqual(by_id[201].vocal_group_key, by_id[301].vocal_group_key)
            self.assertNotEqual(by_id[101].vocal_group_key, by_id[401].vocal_group_key)

            # Verify holdout on "i-dle" isolates both Soyeon and K/DA as well
            rng = np.random.default_rng(1)
            ds = AlignedDataset(
                track_ids=np.array([t.track_id for t in tracks], dtype=np.int64),
                embeddings=rng.normal(size=(len(tracks), 13, 768)).astype(np.float32),
                ratings=np.array([t.rating for t in tracks], dtype=np.float32),
                vocal_groups=np.array([t.vocal_group_key for t in tracks]),
                artists=np.array([t.artist_name for t in tracks]),
                titles=np.array([t.track_title for t in tracks]),
                records=[{"artist_name": t.artist_name, "track_title": t.track_title} for t in tracks],
            )
            train_idx, val_idx, extra = make_holdout_split(ds, ["i-dle"], isolate_vocal_groups=True)
            self.assertEqual(set(extra), {"Soyeon", "K/DA"})
            self.assertEqual(set(ds.artists[val_idx].tolist()), {"i-dle", "Soyeon", "K/DA"})
            self.assertEqual(set(ds.artists[train_idx].tolist()), {"UnrelatedSolo"})

    def test_group_kfold_and_holdout_prevent_vocal_leakage(self) -> None:
        rng = np.random.default_rng(0)
        n = 12
        ds = AlignedDataset(
            track_ids=np.arange(n, dtype=np.int64),
            embeddings=rng.normal(size=(n, 13, 768)).astype(np.float32),
            ratings=np.array([5, 5, 4, 4, 3, 3, 2, 2, 5, 4, 3, 2], dtype=np.float32),
            vocal_groups=np.array(["g1", "g1", "g1", "g1", "g2", "g2", "g2", "g2", "g3", "g3", "g4", "g4"]),
            artists=np.array(["GroupA", "GroupA", "SoloA", "SoloA", "GroupB", "GroupB", "GroupB", "GroupB", "GroupC", "GroupC", "GroupD", "GroupD"]),
            titles=np.array([f"Track {i}" for i in range(n)]),
            records=[{"artist_name": "X", "track_title": f"T{i}"} for i in range(n)],
        )

        for train_idx, val_idx in iter_group_kfold_splits(ds, n_splits=3):
            train_groups = set(ds.vocal_groups[train_idx].tolist())
            val_groups = set(ds.vocal_groups[val_idx].tolist())
            self.assertTrue(train_groups.isdisjoint(val_groups))

        train_idx, val_idx, extra = make_holdout_split(ds, ["GroupA"], isolate_vocal_groups=True)
        self.assertEqual(extra, ["SoloA"])
        self.assertTrue(set(ds.artists[val_idx].tolist()) == {"GroupA", "SoloA"})
        self.assertTrue(set(ds.vocal_groups[train_idx].tolist()).isdisjoint({"g1"}))

    def test_pool_layers_shapes(self) -> None:
        embs = np.ones((5, 13, 768), dtype=np.float32)
        for mode in ("mean", "upper", "mid", "last"):
            pooled = pool_layers(embs, mode=mode)
            self.assertEqual(pooled.shape, (5, 768))


if __name__ == "__main__":
    unittest.main()
