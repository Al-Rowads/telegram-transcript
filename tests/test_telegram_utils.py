from __future__ import annotations

from telegram_transcript.telegram_utils import should_send_as_text, split_text_for_telegram


def test_split_text_for_telegram_returns_empty_for_blank_text() -> None:
    assert split_text_for_telegram("   ") == []


def test_split_text_for_telegram_splits_long_text() -> None:
    chunks = split_text_for_telegram("word " * 20, limit=25)

    assert len(chunks) > 1
    assert all(len(chunk) <= 25 for chunk in chunks)
    assert " ".join(chunks).replace("  ", " ") == ("word " * 20).strip()


def test_should_send_as_text_checks_limit() -> None:
    assert should_send_as_text("hello", limit=5)
    assert not should_send_as_text("hello!", limit=5)
