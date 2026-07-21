from __future__ import annotations

import sqlite3
from pathlib import Path

from telegram_transcript.privacy_migrate import delete_legacy_registry_data


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


def test_delete_legacy_registry_data_does_not_create_missing_database(tmp_path: Path) -> None:
    database = tmp_path / "missing.sqlite3"

    assert delete_legacy_registry_data(database) is False
    assert not database.exists()
