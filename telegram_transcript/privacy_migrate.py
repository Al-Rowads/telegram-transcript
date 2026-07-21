from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def delete_legacy_registry_data(path: Path) -> bool:
    if not path.exists():
        return False
    connection = sqlite3.connect(path)
    try:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'videos'"
        ).fetchone()
        if table is None:
            return False
        connection.executescript(
            "DROP TABLE videos;"
            "DROP INDEX IF EXISTS idx_videos_file_hash;"
            "DROP INDEX IF EXISTS idx_videos_file_name;"
        )
        connection.commit()
        return True
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Irreversibly delete the legacy transcript and sender registry."
    )
    parser.add_argument("path", type=Path, nargs="?", default=Path("data/videos.sqlite3"))
    parser.add_argument(
        "--confirm-delete",
        action="store_true",
        help="Confirm irreversible deletion of legacy transcript and sender records.",
    )
    args = parser.parse_args()
    if not args.confirm_delete:
        parser.error("--confirm-delete is required; no data was changed")
    removed = delete_legacy_registry_data(args.path)
    print("Legacy registry deleted." if removed else "No legacy registry table was found.")


if __name__ == "__main__":
    main()
