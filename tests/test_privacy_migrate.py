from __future__ import annotations

import sqlite3
from pathlib import Path

from telegram_transcript.privacy_migrate import (
    delete_legacy_registry_data,
    delete_legacy_registry_files,
)


def test_delete_legacy_registry_data_drops_content_table(tmp_path: Path) -> None:
    database = tmp_path / "videos.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE videos (transcript TEXT, sender_username TEXT)")
    connection.execute("INSERT INTO videos VALUES ('private transcript', '@sender')")
    connection.commit()
    connection.close()

    assert delete_legacy_registry_data(database) is True

    connection = sqlite3.connect(database)
    try:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'videos'"
        ).fetchone()
    finally:
        connection.close()
    assert table is None
    assert b"private transcript" not in database.read_bytes()


def test_delete_legacy_registry_data_does_not_create_missing_database(tmp_path: Path) -> None:
    database = tmp_path / "missing.sqlite3"

    assert delete_legacy_registry_data(database) is False
    assert not database.exists()


def test_delete_legacy_registry_files_removes_database_and_sidecars(tmp_path: Path) -> None:
    database = tmp_path / "videos.sqlite3"
    candidates = [database, Path(f"{database}-wal"), Path(f"{database}-shm"), Path(f"{database}-journal")]
    for candidate in candidates:
        candidate.write_bytes(b"legacy video data")

    assert delete_legacy_registry_files(database) is True
    assert not any(candidate.exists() for candidate in candidates)


def test_delete_legacy_registry_files_returns_false_when_nothing_exists(tmp_path: Path) -> None:
    assert delete_legacy_registry_files(tmp_path / "missing.sqlite3") is False
