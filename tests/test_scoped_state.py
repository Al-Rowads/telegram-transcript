from __future__ import annotations

import hashlib
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


def test_job_metadata_is_content_free_hashed_and_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "state.sqlite3"
    store = ScopedStateStore(database, defaults=default_preferences())
    try:
        assert store.record_job(
            job_id="job-1",
            scope_key="user:1",
            chat_id=1,
            message_id=2,
            file_unique_id="telegram-secret-id",
        )
        assert not store.record_job(
            job_id="job-2",
            scope_key="user:1",
            chat_id=1,
            message_id=2,
            file_unique_id="telegram-secret-id",
        )
    finally:
        store.close()

    connection = sqlite3.connect(database)
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
        row = connection.execute("SELECT file_unique_hash, status FROM jobs").fetchone()
    finally:
        connection.close()

    assert not {"transcript", "translation", "file_name", "sender_name"} & columns
    assert row == (hashlib.sha256(b"telegram-secret-id").hexdigest(), "queued")


def test_restart_marks_incomplete_jobs_failed_and_forget_deletes_scope(tmp_path: Path) -> None:
    store = ScopedStateStore(tmp_path / "state.sqlite3", defaults=default_preferences())
    try:
        store.save_preferences("user:1", default_preferences())
        store.record_job(
            job_id="job-1",
            scope_key="user:1",
            chat_id=1,
            message_id=2,
            file_unique_id=None,
        )
        assert store.fail_incomplete_jobs() == 1
        store.delete_scope("user:1")
    finally:
        store.close()

    connection = sqlite3.connect(tmp_path / "state.sqlite3")
    try:
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM scope_preferences").fetchone() == (0,)
    finally:
        connection.close()
