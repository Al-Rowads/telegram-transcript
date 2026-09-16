from __future__ import annotations

import sqlite3
import threading
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from telegram_transcript.runtime_state import RuntimePreferences, normalize_legacy_gpt_model
from telegram_transcript.model_catalog import GEMINI_FLASH_LITE_MODEL, RETIRED_TEXT_MODELS


SCHEMA = """
CREATE TABLE IF NOT EXISTS scope_preferences (
    scope_key TEXT PRIMARY KEY,
    audio_tempo REAL NOT NULL,
    transcription_model TEXT NOT NULL,
    transcription_refinement_model TEXT NOT NULL,
    translation_enabled INTEGER NOT NULL,
    translation_model TEXT NOT NULL,
    translation_prompt TEXT NOT NULL,
    transcription_refinement_enabled INTEGER NOT NULL DEFAULT 0,
    audio_correction_enabled INTEGER NOT NULL DEFAULT 0,
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
            with self._connection:
                self._migrate_cost_preferences()
        except Exception:
            self._connection.close()
            raise

    def _migrate_cost_preferences(self) -> None:
        # Include the reset and version marker in the same transaction so a
        # restart never resets a user's new, explicit provider selection.
        self._connection.execute("BEGIN IMMEDIATE")
        if self._connection.execute("PRAGMA user_version").fetchone()[0] >= 1:
            return
        columns = {row[1] for row in self._connection.execute("PRAGMA table_info(scope_preferences)")}
        for column in ("transcription_refinement_enabled", "audio_correction_enabled"):
            if column not in columns:
                self._connection.execute(
                    f"ALTER TABLE scope_preferences ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0"
                )
        self._connection.execute(
            "UPDATE scope_preferences SET transcription_model = 'deepgram', "
            "transcription_refinement_enabled = 0, audio_correction_enabled = 0"
        )
        for field in ("translation_model", "transcription_refinement_model"):
            for model in RETIRED_TEXT_MODELS:
                self._connection.execute(
                    f"UPDATE scope_preferences SET {field} = ? WHERE {field} = ?",
                    (GEMINI_FLASH_LITE_MODEL, model),
                )
        self._connection.execute("PRAGMA user_version=1")

    def load_preferences(self, scope_key: str) -> RuntimePreferences:
        with self._lock:
            row = self._connection.execute(
                "SELECT audio_tempo, transcription_model, transcription_refinement_model, "
                "translation_enabled, translation_model, translation_prompt, "
                "transcription_refinement_enabled, audio_correction_enabled "
                "FROM scope_preferences WHERE scope_key = ?",
                (scope_key,),
            ).fetchone()
        if row is None:
            return self.defaults
        return RuntimePreferences(
            audio_tempo=float(row[0]),
            transcription_model=str(row[1]),
            transcription_refinement_model=normalize_legacy_gpt_model(str(row[2])),
            translation_enabled=bool(row[3]),
            translation_model=normalize_legacy_gpt_model(str(row[4])),
            translation_prompt=str(row[5]),
            transcription_refinement_enabled=bool(row[6]),
            audio_correction_enabled=bool(row[7]),
        )

    def save_preferences(self, scope_key: str, preferences: RuntimePreferences) -> None:
        values = asdict(preferences)
        now = utc_now()
        with self._lock:
            self._connection.execute(
                "INSERT INTO scope_preferences "
                "(scope_key, audio_tempo, transcription_model, transcription_refinement_model, "
                "translation_enabled, translation_model, translation_prompt, updated_at, "
                "transcription_refinement_enabled, audio_correction_enabled) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(scope_key) DO UPDATE SET "
                "audio_tempo=excluded.audio_tempo, transcription_model=excluded.transcription_model, "
                "transcription_refinement_model=excluded.transcription_refinement_model, "
                "translation_enabled=excluded.translation_enabled, translation_model=excluded.translation_model, "
                "translation_prompt=excluded.translation_prompt, updated_at=excluded.updated_at, "
                "transcription_refinement_enabled=excluded.transcription_refinement_enabled, "
                "audio_correction_enabled=excluded.audio_correction_enabled",
                (
                    scope_key,
                    values["audio_tempo"],
                    values["transcription_model"],
                    values["transcription_refinement_model"],
                    int(values["translation_enabled"]),
                    values["translation_model"],
                    values["translation_prompt"],
                    now,
                    int(values["transcription_refinement_enabled"]),
                    int(values["audio_correction_enabled"]),
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
