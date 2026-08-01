from __future__ import annotations

import sqlite3
import threading
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from telegram_transcript.runtime_state import RuntimePreferences


SCHEMA = """
CREATE TABLE IF NOT EXISTS scope_preferences (
    scope_key TEXT PRIMARY KEY,
    audio_tempo REAL NOT NULL,
    transcription_model TEXT NOT NULL,
    transcription_refinement_model TEXT NOT NULL,
    translation_enabled INTEGER NOT NULL,
    translation_model TEXT NOT NULL,
    translation_prompt TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class ScopedStateStore:
    """Persists tenant preferences without retaining per-media state."""

    def __init__(self, path: Path, *, defaults: RuntimePreferences) -> None:
        self.path = path
        self.defaults = defaults
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        try:
            self._connection.execute("PRAGMA busy_timeout=5000")
            purge_legacy_job_metadata(self._connection)
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.executescript(SCHEMA)
            self._connection.commit()
        except Exception:
            self._connection.close()
            raise

    def load_preferences(self, scope_key: str) -> RuntimePreferences:
        with self._lock:
            row = self._connection.execute(
                "SELECT audio_tempo, transcription_model, transcription_refinement_model, "
                "translation_enabled, translation_model, translation_prompt "
                "FROM scope_preferences WHERE scope_key = ?",
                (scope_key,),
            ).fetchone()
        if row is None:
            return self.defaults
        return RuntimePreferences(
            audio_tempo=float(row[0]),
            transcription_model=str(row[1]),
            transcription_refinement_model=str(row[2]),
            translation_enabled=bool(row[3]),
            translation_model=str(row[4]),
            translation_prompt=str(row[5]),
        )

    def save_preferences(self, scope_key: str, preferences: RuntimePreferences) -> None:
        values = asdict(preferences)
        now = utc_now()
        with self._lock:
            self._connection.execute(
                "INSERT INTO scope_preferences "
                "(scope_key, audio_tempo, transcription_model, transcription_refinement_model, "
                "translation_enabled, translation_model, translation_prompt, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(scope_key) DO UPDATE SET "
                "audio_tempo=excluded.audio_tempo, transcription_model=excluded.transcription_model, "
                "transcription_refinement_model=excluded.transcription_refinement_model, "
                "translation_enabled=excluded.translation_enabled, translation_model=excluded.translation_model, "
                "translation_prompt=excluded.translation_prompt, updated_at=excluded.updated_at",
                (
                    scope_key,
                    values["audio_tempo"],
                    values["transcription_model"],
                    values["transcription_refinement_model"],
                    int(values["translation_enabled"]),
                    values["translation_model"],
                    values["translation_prompt"],
                    now,
                ),
            )
            self._connection.commit()

    def delete_scope(self, scope_key: str) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM scope_preferences WHERE scope_key = ?", (scope_key,))
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def purge_legacy_job_metadata(connection: sqlite3.Connection) -> bool:
    table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'jobs'"
    ).fetchone()
    if table is None:
        return False

    # This database also contains preferences, so remove only the legacy media
    # table and rewrite the remaining database instead of deleting the file.
    connection.execute("PRAGMA secure_delete=ON")
    connection.execute("DROP TABLE jobs")
    connection.commit()
    connection.execute("VACUUM")
    checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if checkpoint is not None and int(checkpoint[0]) != 0:
        raise sqlite3.OperationalError("Unable to truncate legacy job metadata from the SQLite WAL.")
    return True
