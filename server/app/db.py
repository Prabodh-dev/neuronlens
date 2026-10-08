import sqlite3
import threading
from pathlib import Path

from .errors import ApiError


class Database:
    """Read-only SQLite handle shared across request threads."""

    def __init__(self, path: Path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"database not found: {path}")
        self.path = Path(path)
        self._conn = sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True,
                                     check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def one(self, sql: str, params: tuple = ()):
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def close(self) -> None:
        self._conn.close()


REQUIRED_META = ("schema_version", "run_id", "model_name", "hook_layer", "d_model",
                 "d_sae", "k", "created_at", "is_synthetic")


def load_meta(db: Database) -> dict:
    try:
        meta = {r["key"]: r["value"] for r in db.query("SELECT key, value FROM meta")}
    except Exception as e:
        raise ApiError("SCHEMA_MISMATCH", f"cannot read meta table: {e}")
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise ApiError("SCHEMA_MISMATCH", f"meta is missing keys: {', '.join(missing)}")
    major = meta["schema_version"].split(".")[0]
    if major != "1":
        raise ApiError("SCHEMA_MISMATCH",
                       f"schema_version {meta['schema_version']!r} is not supported (need 1.x)")
    return meta
