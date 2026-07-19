from __future__ import annotations

import hashlib
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

MATCHED_BY_FILE_NAME = "file_name"
MATCHED_BY_FILE_HASH = "file_hash"

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash TEXT NOT NULL,
    file_name TEXT,
    sender_id INTEGER,
    sender_username TEXT NOT NULL,
    transcript TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_videos_file_hash ON videos (file_hash);
CREATE INDEX IF NOT EXISTS idx_videos_file_name ON videos (file_name);
"""


@dataclass(frozen=True)
class DuplicateMatch:
    first_sender: str
    matched_by: str


class VideoRegistry:
    """Records every processed media file so re-sent files can be flagged as duplicates.

    Methods are synchronous; the bot invokes them via asyncio.to_thread, so a
    lock serializes access to the shared connection across worker threads.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def find_duplicate(self, file_name: str | None, file_hash: str) -> DuplicateMatch | None:
        with self._lock:
            if file_name is not None and file_name.strip():
                row = self._connection.execute(
                    "SELECT sender_username FROM videos WHERE file_name = ? ORDER BY id LIMIT 1",
                    (file_name,),
                ).fetchone()
                if row is not None:
                    return DuplicateMatch(first_sender=row[0], matched_by=MATCHED_BY_FILE_NAME)
            row = self._connection.execute(
                "SELECT sender_username FROM videos WHERE file_hash = ? ORDER BY id LIMIT 1",
                (file_hash,),
            ).fetchone()
            if row is not None:
                return DuplicateMatch(first_sender=row[0], matched_by=MATCHED_BY_FILE_HASH)
            return None

    def record_video(
        self,
        *,
        file_hash: str,
        file_name: str | None,
        sender_id: int | None,
        sender_username: str,
        transcript: str,
    ) -> int:
        with self._lock:
            cursor = self._connection.execute(
                "INSERT INTO videos (file_hash, file_name, sender_id, sender_username, transcript)"
                " VALUES (?, ?, ?, ?, ?)",
                (file_hash, file_name, sender_id, sender_username, transcript),
            )
            self._connection.commit()
            return int(cursor.lastrowid or 0)

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def compute_file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
