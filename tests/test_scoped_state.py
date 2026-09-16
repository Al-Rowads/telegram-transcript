from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from telegram_transcript.runtime_state import RuntimePreferences
from telegram_transcript.model_catalog import GEMINI_FLASH_LITE_MODEL, RETIRED_TEXT_MODELS
from telegram_transcript.scoped_state import ScopedStateStore


def default_preferences() -> RuntimePreferences:
    return RuntimePreferences(
        audio_tempo=1.0,
        transcription_model="gemini",
        transcription_refinement_model="google/gemini-2.5-flash-lite",
        translation_enabled=True,
        translation_model="google/gemini-2.5-flash-lite",
        translation_prompt="natural",
    )


@pytest.mark.parametrize("retired_model", sorted(RETIRED_TEXT_MODELS))
def test_old_sqlite_schema_migrates_once_without_resetting_future_choices(tmp_path: Path, retired_model: str) -> None:
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE scope_preferences (scope_key TEXT PRIMARY KEY, audio_tempo REAL NOT NULL, "
            "transcription_model TEXT NOT NULL, transcription_refinement_model TEXT NOT NULL, "
            "translation_enabled INTEGER NOT NULL, translation_model TEXT NOT NULL, "
            "translation_prompt TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO scope_preferences VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("user:1", 0.9, "gemini", retired_model, 1, retired_model, "literal", "original"),
                ("chat:2", 1.0, "whisper", "custom-model", 0, "google/gemini-3.5-flash", "natural", "original"),
            ],
        )
    store = ScopedStateStore(database, defaults=default_preferences())
    try:
        migrated = store.load_preferences("user:1")
        assert migrated.transcription_model == "deepgram"
        assert migrated.translation_model == GEMINI_FLASH_LITE_MODEL
        assert migrated.transcription_refinement_model == GEMINI_FLASH_LITE_MODEL
        assert migrated.audio_tempo == 0.9 and migrated.translation_prompt == "literal"
        assert migrated.translation_enabled is True
        assert migrated.transcription_refinement_enabled is False
        assert migrated.audio_correction_enabled is False
        other = store.load_preferences("chat:2")
        assert other.translation_model == "google/gemini-3.5-flash"
        assert other.transcription_refinement_model == "custom-model"
        assert other.translation_enabled is False
        selected = replace(migrated, transcription_model="gemini", transcription_refinement_enabled=True,
                           audio_correction_enabled=True)
        store.save_preferences("user:1", selected)
    finally:
        store.close()
    reopened = ScopedStateStore(database, defaults=default_preferences())
    try:
        assert reopened.load_preferences("user:1") == selected
        assert reopened.load_preferences("chat:2") == other
        with sqlite3.connect(database) as connection:
            assert connection.execute("PRAGMA user_version").fetchone() == (1,)
    finally:
        reopened.close()


def test_scoped_preferences_are_isolated(tmp_path: Path) -> None:
    store = ScopedStateStore(tmp_path / "state.sqlite3", defaults=default_preferences())
    changed = RuntimePreferences(
        audio_tempo=0.8,
        transcription_model="deepgram",
        transcription_refinement_model="google/gemini-2.5-flash-lite",
        translation_enabled=False,
        translation_model="google/gemini-2.5-flash-lite",
        translation_prompt="literal",
    )
    try:
        store.save_preferences("user:1", changed)

        assert store.load_preferences("user:1") == changed
        assert store.load_preferences("user:2") == default_preferences()
        assert store.load_preferences("chat:-100") == default_preferences()
    finally:
        store.close()


@pytest.mark.parametrize("legacy_field", ["translation_model", "transcription_refinement_model", "both"])
def test_scoped_preferences_migrate_saved_gpt_models(tmp_path: Path, legacy_field: str) -> None:
    database = tmp_path / "state.sqlite3"
    expected = replace(
        default_preferences(),
        audio_tempo=0.8,
        transcription_model="deepgram",
        translation_enabled=False,
        translation_prompt="literal",
        translation_model=(
            "qwen/qwen3-30b-a3b-instruct-2507"
            if legacy_field == "transcription_refinement_model"
            else "google/gemini-2.5-flash-lite"
        ),
        transcription_refinement_model=(
            "google/gemini-3.5-flash"
            if legacy_field == "translation_model"
            else "google/gemini-2.5-flash-lite"
        ),
    )
    legacy = replace(expected, **{
        field: "openai/gpt-5.5"
        for field in ("translation_model", "transcription_refinement_model")
        if legacy_field in {field, "both"}
    })
    store = ScopedStateStore(database, defaults=default_preferences())
    try:
        store.save_preferences("user:1", legacy)
        store.save_preferences("user:2", expected)
        assert store.load_preferences("user:1") == expected
        assert store.load_preferences("user:1") == expected
        with sqlite3.connect(database) as connection:
            assert connection.execute(
                "SELECT translation_model, transcription_refinement_model FROM scope_preferences WHERE scope_key = ?",
                ("user:1",),
            ).fetchone() == (legacy.translation_model, legacy.transcription_refinement_model)
        store.save_preferences("user:1", store.load_preferences("user:1"))
        with sqlite3.connect(database) as connection:
            assert connection.execute(
                "SELECT translation_model, transcription_refinement_model FROM scope_preferences WHERE scope_key = ?",
                ("user:1",),
            ).fetchone() == (expected.translation_model, expected.transcription_refinement_model)
        assert store.load_preferences("user:2") == expected
    finally:
        store.close()
    store = ScopedStateStore(database, defaults=default_preferences())
    try:
        assert store.load_preferences("user:1") == expected
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
        transcription_refinement_model="google/gemini-2.5-flash-lite",
        translation_enabled=False,
        translation_model="google/gemini-2.5-flash-lite",
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
