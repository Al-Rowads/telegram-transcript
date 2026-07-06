from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from telegram.constants import ChatType

from telegram_transcript import bot as bot_module
from telegram_transcript.bot import (
    get_attachment_suffix,
    get_video_attachment,
    handle_non_video,
    handle_tempo_command,
    handle_video_upload,
    is_authorized,
    is_video_document,
    process_video_message,
    send_transcript,
)
from telegram_transcript.config import Settings
from telegram_transcript.models import TranscriptionResult, SubtitleCue


class FakeMessage:
    def __init__(
        self,
        *,
        video: object | None = None,
        document: object | None = None,
        chat_type: str = ChatType.PRIVATE,
        message_id: int = 42,
        message_thread_id: int | None = None,
    ) -> None:
        self.video = video
        self.document = document
        self.chat = SimpleNamespace(type=chat_type)
        self.message_id = message_id
        self.message_thread_id = message_thread_id
        self.text_replies: list[str] = []
        self.text_reply_kwargs: list[dict[str, object]] = []
        self.status_replies: list[FakeStatus] = []
        self.document_replies: list[object] = []
        self.document_reply_kwargs: list[dict[str, object]] = []

    async def reply_text(self, text: str, **kwargs: object) -> object:
        self.text_replies.append(text)
        self.text_reply_kwargs.append(kwargs)
        status = FakeStatus()
        self.status_replies.append(status)
        return status

    async def reply_document(self, *, document: object, caption: str, **kwargs: object) -> None:
        self.document_replies.append((document, caption))
        self.document_reply_kwargs.append(kwargs)


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
async def test_send_transcript_replies_to_original_group_message() -> None:
    message = FakeMessage(chat_type=ChatType.SUPERGROUP, message_id=99, message_thread_id=7)

    await send_transcript(message, "hello")

    assert message.text_replies == ["hello"]
    assert message.text_reply_kwargs == [
        {
            "reply_to_message_id": 99,
            "allow_sending_without_reply": True,
            "message_thread_id": 7,
        }
    ]


@pytest.mark.asyncio
async def test_send_transcript_sends_long_text_as_document() -> None:
    message = FakeMessage(chat_type=ChatType.SUPERGROUP, message_id=99, message_thread_id=7)

    await send_transcript(message, "x" * 4000)

    assert message.text_replies == []
    assert len(message.document_replies) == 1
    assert message.document_reply_kwargs == [
        {
            "reply_to_message_id": 99,
            "allow_sending_without_reply": True,
            "message_thread_id": 7,
        }
    ]


@pytest.mark.asyncio
async def test_handle_non_video_ignores_group_messages() -> None:
    message = FakeMessage(chat_type=ChatType.GROUP)
    update = SimpleNamespace(effective_message=message)

    await handle_non_video(update, SimpleNamespace())

    assert message.text_replies == []


@pytest.mark.asyncio
async def test_handle_non_video_ignores_private_messages() -> None:
    message = FakeMessage()
    update = SimpleNamespace(effective_message=message)

    await handle_non_video(update, SimpleNamespace())

    assert message.text_replies == []


@pytest.mark.asyncio
async def test_handle_tempo_command_updates_runtime_tempo_in_group() -> None:
    message = FakeMessage(chat_type=ChatType.SUPERGROUP, message_id=123, message_thread_id=8)
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=999))
    context = SimpleNamespace(
        args=["1.2"],
        bot_data={
            "audio_tempo": 1.0,
            "settings": Settings(
                telegram_bot_token="token",
                openai_api_key="key",
                allowed_telegram_user_ids=frozenset({123}),
            ),
        },
    )

    await handle_tempo_command(update, context)

    assert context.bot_data["audio_tempo"] == 1.2
    assert message.text_replies == ["Tempo set to 1.2x."]
    assert message.text_reply_kwargs == [
        {
            "reply_to_message_id": 123,
            "allow_sending_without_reply": True,
            "message_thread_id": 8,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_type", [ChatType.PRIVATE, ChatType.CHANNEL])
async def test_handle_tempo_command_ignores_private_and_channel_chats(chat_type: str) -> None:
    message = FakeMessage(chat_type=chat_type)
    update = SimpleNamespace(effective_message=message)
    context = SimpleNamespace(args=["1.2"], bot_data={"audio_tempo": 1.0})

    await handle_tempo_command(update, context)

    assert context.bot_data["audio_tempo"] == 1.0
    assert message.text_replies == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args",
    [
        [],
        ["fast"],
        ["1.0", "extra"],
        ["0.49"],
        ["2.01"],
    ],
)
async def test_handle_tempo_command_ignores_invalid_values(args: list[str]) -> None:
    message = FakeMessage(chat_type=ChatType.GROUP)
    update = SimpleNamespace(effective_message=message)
    context = SimpleNamespace(args=args, bot_data={"audio_tempo": 1.0})

    await handle_tempo_command(update, context)

    assert context.bot_data["audio_tempo"] == 1.0
    assert message.text_replies == []


@pytest.mark.asyncio
async def test_handle_video_upload_rejects_unauthorized_user() -> None:
    message = FakeMessage(video=SimpleNamespace(file_size=1), chat_type=ChatType.GROUP, message_id=123)
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
    assert message.text_reply_kwargs[0]["reply_to_message_id"] == 123


@pytest.mark.asyncio
async def test_handle_video_upload_ignores_private_non_video_document() -> None:
    message = FakeMessage(document=SimpleNamespace(mime_type="text/plain", file_name="notes.txt"))
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    context = SimpleNamespace(bot_data={"settings": Settings(telegram_bot_token="token", openai_api_key="key")})

    await handle_video_upload(update, context)

    assert message.text_replies == []


@pytest.mark.asyncio
async def test_handle_video_upload_ignores_group_non_video_document() -> None:
    message = FakeMessage(
        document=SimpleNamespace(mime_type="text/plain", file_name="notes.txt"),
        chat_type=ChatType.GROUP,
    )
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    context = SimpleNamespace(bot_data={"settings": Settings(telegram_bot_token="token", openai_api_key="key")})

    await handle_video_upload(update, context)

    assert message.text_replies == []


@pytest.mark.asyncio
async def test_handle_video_upload_ignores_oversized_video() -> None:
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

    assert message.text_replies == []


@pytest.mark.asyncio
async def test_handle_video_upload_accepts_video_at_exact_size(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        telegram_bot_token="token",
        openai_api_key="key",
        max_video_mb=10 / 1024 / 1024,
    )
    attachment = SimpleNamespace(file_size=10)
    message = FakeMessage(video=attachment, chat_type=ChatType.SUPERGROUP, message_id=123, message_thread_id=8)
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    context = SimpleNamespace(
        bot_data={
            "settings": settings,
            "job_semaphore": asyncio.Semaphore(1),
            "audio_tempo": 1.4,
        }
    )
    processed = False

    async def fake_process_video_message(*args: object) -> None:
        nonlocal processed
        processed = True
        assert args[0] is message
        assert args[1] is attachment
        assert args[6] == 1.4
        context.bot_data["audio_tempo"] = 2.0

    monkeypatch.setattr(bot_module, "process_video_message", fake_process_video_message)

    await handle_video_upload(update, context)

    assert processed
    assert message.text_replies == []


@pytest.mark.asyncio
async def test_process_video_message_reports_step_by_step_flow(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def run_inline(func: object, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)

    class FakeTelegramFile:
        async def download_to_drive(self, *, custom_path: str) -> None:
            Path(custom_path).write_bytes(b"video")

    class FakeAttachment:
        file_name = "clip.mp4"
        file_size = 5

        async def get_file(self) -> FakeTelegramFile:
            return FakeTelegramFile()

    class FakeTranscriber:
        async def transcribe_chunks_async(self, chunks: object, progress_callback: object = None) -> TranscriptionResult:
            chunk = list(chunks)[0]
            assert progress_callback is not None
            await progress_callback(
                "transcribing_chunk",
                {
                    "index": 1,
                    "total": 1,
                    "chunk_bytes": chunk.path.stat().st_size,
                    "model": "gpt-4o-transcribe",
                },
            )
            await progress_callback("chunk_transcribed", {"index": 1, "total": 1, "raw_chars": 12})
            await progress_callback(
                "refining_transcript",
                {
                    "raw_chars": 12,
                    "model": "gpt-5.4-mini",
                },
            )
            await progress_callback("refinement_complete", {"cleaned_chars": 9})
            return TranscriptionResult(
                raw_transcript="هاي خام",
                refined_transcript="هاي مرتبة",
                subtitle_cues=(SubtitleCue(0.0, 1.25, "هاي خام"),),
            )

    def fake_extract_audio(video_path: Path, audio_path: Path, *, audio_tempo: float) -> Path:
        assert video_path.exists()
        assert audio_tempo == 1.4
        audio_path.write_bytes(b"audio")
        return audio_path

    monkeypatch.setattr(bot_module, "extract_audio", fake_extract_audio)
    caplog.set_level("INFO", logger="telegram_transcript.bot")
    message = FakeMessage()
    status_ref: dict[str, object] = {"message": None}
    settings = Settings(telegram_bot_token="token", openai_api_key="key")
    context = SimpleNamespace(bot_data={"transcriber": FakeTranscriber()})

    await process_video_message(message, FakeAttachment(), settings, context, status_ref, "job1234", 1.4)

    assert message.text_replies == ["Video received. Starting transcription...", "هاي خام", "هاي مرتبة"]
    assert [caption for _, caption in message.document_replies] == ["SRT subtitles"]
    status = message.status_replies[0]
    assert status.edits == [
        "Step 2/6: extracting MP3 audio at 1.4x...",
        "Step 3/6: preparing audio chunks...",
        "Step 4/6: transcribing chunk 1/1...",
        "Step 5/6: refining transcript...",
        "Step 6/6: sending transcript...",
        "Transcript ready.",
    ]
    assert "job1234 step 1/6" in caplog.text
    assert "job1234 step 6/6" in caplog.text
    assert "هاي مرتبة" not in caplog.text


@pytest.mark.asyncio
async def test_process_video_message_skips_srt_when_timestamps_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_inline(func: object, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)

    class FakeTelegramFile:
        async def download_to_drive(self, *, custom_path: str) -> None:
            Path(custom_path).write_bytes(b"video")

    class FakeAttachment:
        file_name = "clip.mp4"
        file_size = 5

        async def get_file(self) -> FakeTelegramFile:
            return FakeTelegramFile()

    class FakeTranscriber:
        async def transcribe_chunks_async(self, chunks: object, progress_callback: object = None) -> TranscriptionResult:
            return TranscriptionResult(raw_transcript="raw only")

    def fake_extract_audio(video_path: Path, audio_path: Path, *, audio_tempo: float) -> Path:
        audio_path.write_bytes(b"audio")
        return audio_path

    monkeypatch.setattr(bot_module, "extract_audio", fake_extract_audio)
    message = FakeMessage()
    status_ref: dict[str, object] = {"message": None}
    settings = Settings(telegram_bot_token="token", openai_api_key="key")
    context = SimpleNamespace(bot_data={"transcriber": FakeTranscriber()})

    await process_video_message(message, FakeAttachment(), settings, context, status_ref, "job1234", 1.0)

    assert message.text_replies == ["Video received. Starting transcription...", "raw only"]
    assert message.document_replies == []
    assert message.status_replies[0].edits[-1] == "Transcript ready. SRT unavailable for this provider/model."


@pytest.mark.asyncio
async def test_process_video_message_rejects_downloaded_file_over_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTelegramFile:
        async def download_to_drive(self, *, custom_path: str) -> None:
            Path(custom_path).write_bytes(b"x" * 11)

    class FakeAttachment:
        file_name = "clip.mp4"
        file_size = None

        async def get_file(self) -> FakeTelegramFile:
            return FakeTelegramFile()

    def fail_extract_audio(*_: object, **__: object) -> None:
        raise AssertionError("extract_audio should not run for an oversized downloaded video")

    monkeypatch.setattr(bot_module, "extract_audio", fail_extract_audio)
    message = FakeMessage()
    status_ref: dict[str, object] = {"message": None}
    settings = Settings(
        telegram_bot_token="token",
        openai_api_key="key",
        max_video_mb=10 / 1024 / 1024,
    )
    context = SimpleNamespace(bot_data={"transcriber": object()})

    await process_video_message(message, FakeAttachment(), settings, context, status_ref, "job1234", 1.0)

    assert status_ref["message"] is None
    assert message.text_replies == []
