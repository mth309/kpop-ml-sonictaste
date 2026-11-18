"""Read-only catalog ingestion from K-Pop DAT's kpop_ratings.db."""

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import sys
from typing import Dict, List, Optional

from config import KPOP_APP_DIR, KPOP_DB_PATH

# Reuse K-Pop DAT's canonical deduplication & UDFs without duplicating SQL invariants
if str(KPOP_APP_DIR) not in sys.path:
    sys.path.insert(0, str(KPOP_APP_DIR))

import database as kpop_db  # noqa: E402


@dataclass(frozen=True)
class TrackRecord:
    """Represents a single canonical, rated track from the K-Pop DAT catalog."""

    track_id: int
    artist_id: int
    artist_name: str
    artist_type: str
    vocal_group_key: str
    album_id: int
    album_title: str
    album_type: str
    release_date: str
    track_title: str
    track_type: str
    rating: float
    is_title_track: bool


def _open_readonly_connection(db_path: Path) -> sqlite3.Connection:
    """Opens kpop_ratings.db in strict read-only URI mode with required UDFs."""
    if not db_path.exists():
        raise FileNotFoundError(f"K-Pop DAT database not found at: {db_path}")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.create_function("is_alt_lang", 1, kpop_db.is_alt_lang_text)
    conn.create_function("release_day_gap", 2, kpop_db.release_day_gap)
    conn.create_function("py_lower", 1, lambda s: s.casefold() if s else "")
    return conn


def _build_vocal_group_map(conn: sqlite3.Connection) -> Dict[int, str]:
    """Maps each artist_id to a shared vocal_group_key for leakage-free CV splits.

    Uses Union-Find over (solo_artist_id, group_id) edges in artist_members so
    connected components of linked groups and soloists (e.g., i-dle <-> Soyeon <-> K/DA)
    all receive the exact same vocal_group_key regardless of ID or name order.
    """
    artists = conn.execute("SELECT id, artist_type FROM artists").fetchall()
    artist_types = {int(a["id"]): a["artist_type"] for a in artists}
    parent = {aid: aid for aid in artist_types}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    edges = conn.execute(
        """
        SELECT am.solo_artist_id, am.group_id
        FROM artist_members am
        WHERE am.solo_artist_id IS NOT NULL AND am.group_id IS NOT NULL
        """
    ).fetchall()
    for row in edges:
        solo_id = int(row["solo_artist_id"])
        group_id = int(row["group_id"])
        if solo_id in parent and group_id in parent:
            union(solo_id, group_id)

    has_group = {find(aid) for aid, atype in artist_types.items() if atype == "group"}

    mapping: Dict[int, str] = {}
    for aid, atype in artist_types.items():
        root = find(aid)
        if root in has_group or atype == "group":
            mapping[aid] = f"group_{root}"
        else:
            mapping[aid] = f"solo_{root}"
    return mapping


def load_canonical_rated_tracks(
    db_path: Optional[Path] = None,
    artists_filter: Optional[List[str]] = None,
) -> List[TrackRecord]:
    """Loads rated, deduplicated canonical tracks from kpop_ratings.db (read-only).

    Args:
        db_path: Optional override path to SQLite database. Defaults to config.KPOP_DB_PATH.
        artists_filter: Optional list of artist names (case-insensitive) to restrict to.

    Returns:
        List of TrackRecord entries ordered by artist, release_date, album_id, track_number.
    """
    target_db = Path(db_path) if db_path else KPOP_DB_PATH
    conn = _open_readonly_connection(target_db)
    try:
        vocal_map = _build_vocal_group_map(conn)

        query = f"""
            SELECT
                tr.id AS track_id,
                ar.id AS artist_id,
                ar.name AS artist_name,
                COALESCE(ar.artist_type, 'group') AS artist_type,
                al.id AS album_id,
                al.title AS album_title,
                COALESCE(al.album_type, 'EP') AS album_type,
                COALESCE(al.release_date, '') AS release_date,
                tr.title AS track_title,
                COALESCE(tr.track_type, 'core') AS track_type,
                tr.rating AS rating,
                COALESCE(tr.is_title_track, 0) AS is_title_track
            FROM tracks tr
            JOIN albums al ON tr.album_id = al.id
            JOIN artists ar ON al.artist_id = ar.id
            WHERE tr.rating IS NOT NULL
              AND {kpop_db.CANONICAL_UNIQUE_TRACK_SQL}
            ORDER BY ar.name COLLATE NOCASE, al.release_date, al.id, tr.track_number, tr.id
        """
        rows = conn.execute(query).fetchall()
    finally:
        conn.close()

    filter_set = {a.strip().casefold() for a in artists_filter} if artists_filter else None

    records: List[TrackRecord] = []
    for r in rows:
        artist_name = r["artist_name"]
        if filter_set and artist_name.strip().casefold() not in filter_set:
            continue
        aid = int(r["artist_id"])
        records.append(
            TrackRecord(
                track_id=int(r["track_id"]),
                artist_id=aid,
                artist_name=artist_name,
                artist_type=r["artist_type"],
                vocal_group_key=vocal_map.get(aid, f"artist_{aid}"),
                album_id=int(r["album_id"]),
                album_title=r["album_title"],
                album_type=r["album_type"],
                release_date=r["release_date"],
                track_title=r["track_title"],
                track_type=r["track_type"],
                rating=float(r["rating"]),
                is_title_track=bool(r["is_title_track"]),
            )
        )
    return records
