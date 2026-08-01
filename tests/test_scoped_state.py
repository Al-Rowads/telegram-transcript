from __future__ import annotations

import sqlite3
from pathlib import Path

from telegram_transcript.runtime_state import RuntimePreferences
from telegram_transcript.scoped_state import ScopedStateStore


def default_preferences() -> RuntimePreferences:
    return RuntimePreferences(
        audio_tempo=1.0,
        transcription_model="gemini",
        transcription_refinement_model="openai/gpt-5.5",
        translation_enabled=True,
        translation_model="openai/gpt-5.5",
        translation_prompt="natural",
    )


def test_scoped_preferences_are_isolated(tmp_path: Path) -> None:
    store = ScopedStateStore(tmp_path / "state.sqlite3", defaults=default_preferences())
    changed = RuntimePreferences(
        audio_tempo=0.8,
        transcription_model="deepgram",
        transcription_refinement_model="openai/gpt-5.5",
        translation_enabled=False,
        translation_model="openai/gpt-5.5",
        translation_prompt="literal",
    )
    try:
        store.save_preferences("user:1", changed)

        assert store.load_preferences("user:1") == changed
        assert store.load_preferences("user:2") == default_preferences()
        assert store.load_preferences("chat:-100") == default_preferences()
    finally:
        store.close()


def test_state_database_does_not_create_media_job_storage(tmp_path: Path) -> None:
    database = tmp_path / "state.sqlite3"
    store = ScopedStateStore(database, defaults=default_preferences())
    store.close()

    connection = sqlite3.connect(database)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        connection.close()

    assert tables == {"scope_preferences"}


def test_opening_state_database_purges_legacy_jobs_and_preserves_preferences(tmp_path: Path) -> None:
    database = tmp_path / "state.sqlite3"
    changed = RuntimePreferences(
        audio_tempo=0.9,
        transcription_model="whisper",
        transcription_refinement_model="openai/gpt-5.5",
        translation_enabled=False,
        translation_model="openai/gpt-5.5",
        translation_prompt="literal",
    )
    store = ScopedStateStore(database, defaults=default_preferences())
    try:
        store.save_preferences("user:1", changed)
    finally:
        store.close()

    sentinel = "legacy-media-metadata-sentinel"
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE jobs (metadata TEXT NOT NULL)")
        connection.execute("INSERT INTO jobs VALUES (?)", (sentinel,))
        connection.commit()
    finally:
        connection.close()

    store = ScopedStateStore(database, defaults=default_preferences())
    try:
        assert store.load_preferences("user:1") == changed
    finally:
        store.close()

    connection = sqlite3.connect(database)
    try:
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'jobs'"
        ).fetchone() is None
    finally:
        connection.close()
    assert sentinel.encode() not in database.read_bytes()
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(f"{database}{suffix}")
        assert not sidecar.exists() or sentinel.encode() not in sidecar.read_bytes()


def test_delete_scope_removes_only_preferences(tmp_path: Path) -> None:
    database = tmp_path / "state.sqlite3"
    store = ScopedStateStore(database, defaults=default_preferences())
    try:
        store.save_preferences("user:1", default_preferences())
        store.delete_scope("user:1")
        assert store.load_preferences("user:1") == default_preferences()
    finally:
        store.close()
