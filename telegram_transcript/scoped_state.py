from __future__ import annotations

import hashlib
import sqlite3
import threading
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from telegram_transcript.runtime_state import RuntimePreferences


RETENTION_DAYS = 7

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
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    scope_key TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    file_unique_hash TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    UNIQUE (chat_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_jobs_scope ON jobs (scope_key);
CREATE INDEX IF NOT EXISTS idx_jobs_expires ON jobs (expires_at);
"""


class ScopedStateStore:
    """Persists tenant preferences and short-lived, content-free job metadata."""

    def __init__(self, path: Path, *, defaults: RuntimePreferences) -> None:
        self.path = path
        self.defaults = defaults
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.executescript(SCHEMA)
        self._connection.commit()

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

    def record_job(
        self,
        *,
        job_id: str,
        scope_key: str,
        chat_id: int,
        message_id: int,
        file_unique_id: str | None,
    ) -> bool:
        now = datetime.now(UTC)
        expires_at = now + timedelta(days=RETENTION_DAYS)
        unique_hash = (
            hashlib.sha256(file_unique_id.encode("utf-8")).hexdigest()
            if file_unique_id
            else None
        )
        with self._lock:
            cursor = self._connection.execute(
                "INSERT OR IGNORE INTO jobs "
                "(job_id, scope_key, chat_id, message_id, file_unique_hash, status, created_at, updated_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?)",
                (
                    job_id,
                    scope_key,
                    chat_id,
                    message_id,
                    unique_hash,
                    now.isoformat(),
                    now.isoformat(),
                    expires_at.isoformat(),
                ),
            )
            self._connection.commit()
            return cursor.rowcount == 1

    def set_job_status(self, job_id: str, status: str) -> None:
        now = datetime.now(UTC)
        expires_at = now + timedelta(days=RETENTION_DAYS)
        with self._lock:
            self._connection.execute(
                "UPDATE jobs SET status = ?, updated_at = ?, expires_at = ? WHERE job_id = ?",
                (status, now.isoformat(), expires_at.isoformat(), job_id),
            )
            self._connection.commit()

    def delete_scope(self, scope_key: str) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM scope_preferences WHERE scope_key = ?", (scope_key,))
            self._connection.execute("DELETE FROM jobs WHERE scope_key = ?", (scope_key,))
            self._connection.commit()

    def purge_expired_jobs(self) -> int:
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM jobs WHERE expires_at <= ?",
                (utc_now(),),
            )
            self._connection.commit()
            return cursor.rowcount

    def fail_incomplete_jobs(self) -> int:
        """Mark jobs interrupted by a process restart as failed without retaining content."""
        now = datetime.now(UTC)
        expires_at = now + timedelta(days=RETENTION_DAYS)
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE jobs SET status = 'failed', updated_at = ?, expires_at = ? "
                "WHERE status IN ('queued', 'running')",
                (now.isoformat(), expires_at.isoformat()),
            )
            self._connection.commit()
            return cursor.rowcount

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()
