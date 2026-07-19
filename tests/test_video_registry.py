from __future__ import annotations

import hashlib
from pathlib import Path

from telegram_transcript.video_registry import (
    DuplicateMatch,
    VideoRegistry,
    compute_file_sha256,
)


def make_registry(tmp_path: Path) -> VideoRegistry:
    return VideoRegistry(tmp_path / "videos.sqlite3")


def test_find_duplicate_returns_none_for_empty_registry(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)

    assert registry.find_duplicate("clip.mp4", "hash-1") is None


def test_find_duplicate_matches_by_file_name(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    registry.record_video(
        file_hash="hash-1",
        file_name="clip.mp4",
        sender_id=1,
        sender_username="@alice",
        transcript="first transcript",
    )

    match = registry.find_duplicate("clip.mp4", "different-hash")

    assert match == DuplicateMatch(first_sender="@alice", matched_by="file_name")


def test_find_duplicate_falls_back_to_file_hash_on_name_miss(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    registry.record_video(
        file_hash="hash-1",
        file_name="clip.mp4",
        sender_id=1,
        sender_username="@alice",
        transcript="first transcript",
    )

    match = registry.find_duplicate("renamed.mp4", "hash-1")

    assert match == DuplicateMatch(first_sender="@alice", matched_by="file_hash")


def test_find_duplicate_skips_name_check_when_file_name_missing(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    registry.record_video(
        file_hash="hash-1",
        file_name=None,
        sender_id=1,
        sender_username="@alice",
        transcript="first transcript",
    )

    assert registry.find_duplicate(None, "hash-1") == DuplicateMatch(
        first_sender="@alice", matched_by="file_hash"
    )
    assert registry.find_duplicate(None, "hash-2") is None


def test_find_duplicate_reports_earliest_sender(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)
    registry.record_video(
        file_hash="hash-1",
        file_name="clip.mp4",
        sender_id=1,
        sender_username="@alice",
        transcript="first transcript",
    )
    registry.record_video(
        file_hash="hash-1",
        file_name="clip.mp4",
        sender_id=2,
        sender_username="@bob",
        transcript="second transcript",
    )

    match = registry.find_duplicate("clip.mp4", "hash-1")

    assert match is not None
    assert match.first_sender == "@alice"


def test_registry_persists_across_reopen(tmp_path: Path) -> None:
    first = make_registry(tmp_path)
    first.record_video(
        file_hash="hash-1",
        file_name="clip.mp4",
        sender_id=1,
        sender_username="@alice",
        transcript="first transcript",
    )
    first.close()

    reopened = make_registry(tmp_path)
    match = reopened.find_duplicate("clip.mp4", "hash-1")

    assert match is not None
    assert match.first_sender == "@alice"


def test_record_video_returns_incrementing_row_ids(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)

    first_id = registry.record_video(
        file_hash="hash-1",
        file_name=None,
        sender_id=None,
        sender_username="@alice",
        transcript="first transcript",
    )
    second_id = registry.record_video(
        file_hash="hash-2",
        file_name=None,
        sender_id=None,
        sender_username="@bob",
        transcript="second transcript",
    )

    assert second_id > first_id > 0


def test_compute_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    payload = b"sample video bytes" * 1024
    source = tmp_path / "clip.mp4"
    source.write_bytes(payload)

    assert compute_file_sha256(source, chunk_size=64) == hashlib.sha256(payload).hexdigest()
