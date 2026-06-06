from __future__ import annotations

from types import SimpleNamespace

import pytest

from telegram_transcript.bot import (
    get_attachment_suffix,
    get_video_attachment,
    handle_video_upload,
    is_authorized,
    is_video_document,
    send_transcript,
)
from telegram_transcript.config import Settings


class FakeMessage:
    def __init__(self, *, video: object | None = None, document: object | None = None) -> None:
        self.video = video
        self.document = document
        self.text_replies: list[str] = []
        self.document_replies: list[object] = []

    async def reply_text(self, text: str) -> object:
        self.text_replies.append(text)
        return FakeStatus()

    async def reply_document(self, *, document: object, caption: str) -> None:
        self.document_replies.append((document, caption))


class FakeStatus:
    def __init__(self) -> None:
        self.edits: list[str] = []

    async def edit_text(self, text: str) -> None:
        self.edits.append(text)


def test_is_authorized_allows_everyone_without_allowlist() -> None:
    settings = Settings(telegram_bot_token="token", openai_api_key="key")

    assert is_authorized(settings, user_id=123)


def test_is_authorized_checks_allowlist() -> None:
    settings = Settings(
        telegram_bot_token="token",
        openai_api_key="key",
        allowed_telegram_user_ids=frozenset({123}),
    )

    assert is_authorized(settings, user_id=123)
    assert not is_authorized(settings, user_id=999)


def test_is_video_document_accepts_video_mime_type() -> None:
    document = SimpleNamespace(mime_type="video/mp4", file_name="upload.bin")

    assert is_video_document(document)


def test_is_video_document_accepts_video_extension() -> None:
    document = SimpleNamespace(mime_type="application/octet-stream", file_name="clip.mov")

    assert is_video_document(document)


def test_get_video_attachment_prefers_video() -> None:
    video = SimpleNamespace(file_size=1)
    document = SimpleNamespace(mime_type="video/mp4", file_name="clip.mp4")
    message = SimpleNamespace(video=video, document=document)

    assert get_video_attachment(message) is video


def test_get_attachment_suffix_defaults_to_mp4() -> None:
    assert get_attachment_suffix(SimpleNamespace(file_name="clip.txt")) == ".mp4"


@pytest.mark.asyncio
async def test_send_transcript_sends_short_text() -> None:
    message = FakeMessage()

    await send_transcript(message, "hello")

    assert message.text_replies == ["hello"]
    assert message.document_replies == []


@pytest.mark.asyncio
async def test_send_transcript_sends_long_text_as_document() -> None:
    message = FakeMessage()

    await send_transcript(message, "x" * 4000)

    assert message.text_replies == []
    assert len(message.document_replies) == 1


@pytest.mark.asyncio
async def test_handle_video_upload_rejects_unauthorized_user() -> None:
    message = FakeMessage(video=SimpleNamespace(file_size=1))
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=999))
    context = SimpleNamespace(
        bot_data={
            "settings": Settings(
                telegram_bot_token="token",
                openai_api_key="key",
                allowed_telegram_user_ids=frozenset({123}),
            )
        }
    )

    await handle_video_upload(update, context)

    assert "not enabled" in message.text_replies[0]


@pytest.mark.asyncio
async def test_handle_video_upload_rejects_non_video() -> None:
    message = FakeMessage(document=SimpleNamespace(mime_type="text/plain", file_name="notes.txt"))
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    context = SimpleNamespace(bot_data={"settings": Settings(telegram_bot_token="token", openai_api_key="key")})

    await handle_video_upload(update, context)

    assert message.text_replies == ["Please send a video file to transcribe."]


@pytest.mark.asyncio
async def test_handle_video_upload_rejects_oversized_video() -> None:
    message = FakeMessage(video=SimpleNamespace(file_size=11))
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    context = SimpleNamespace(
        bot_data={
            "settings": Settings(
                telegram_bot_token="token",
                openai_api_key="key",
                max_video_mb=10 / 1024 / 1024,
            )
        }
    )

    await handle_video_upload(update, context)

    assert "larger than" in message.text_replies[0]
