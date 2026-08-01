from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


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
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA secure_delete=ON")
        connection.execute("DROP TABLE videos")
        connection.commit()
        connection.execute("VACUUM")
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint is not None and int(checkpoint[0]) != 0:
            raise sqlite3.OperationalError("Unable to truncate the legacy registry SQLite WAL.")
        return True
    finally:
        connection.close()


def delete_legacy_registry_files(path: Path) -> bool:
    candidates = (path, *(Path(f"{path}{suffix}") for suffix in SQLITE_SIDECAR_SUFFIXES))
    found = any(candidate.exists() for candidate in candidates)
    for candidate in candidates:
        candidate.unlink(missing_ok=True)
    return found


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
