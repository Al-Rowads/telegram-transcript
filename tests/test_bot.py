from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from telegram_transcript import bot as bot_module
from telegram_transcript.bot import (
    BotState,
    get_attachment_suffix,
    get_audio_suffix,
    get_message_topic_id,
    get_media_attachment,
    handle_new_message,
    handle_noise_command,
    handle_non_media,
    handle_tempo_command,
    handle_media_upload,
    is_audio_document,
    is_audio_message,
    is_authorized,
    is_video_document,
    process_media_message,
    send_transcript,
    send_transcription_result,
)
from telegram_transcript.config import Settings, mb_to_bytes
from telegram_transcript.transcriber import TranscriptionResult


class FakeClient:
    def __init__(self) -> None:
        self.sent_files: list[dict[str, object]] = []

    async def send_file(self, entity: object, file: object, **kwargs: object) -> None:
        self.sent_files.append({"entity": entity, "file": file, "kwargs": kwargs})


class FakeMessage:
    def __init__(
        self,
        *,
        raw_text: str = "",
        file: object | None = None,
        video: object | None = None,
        voice: object | None = None,
        media_bytes: bytes = b"video",
        is_private: bool = True,
        is_group: bool = False,
        is_channel: bool = False,
        id: int = 42,
        chat_id: int = 100,
        sender_id: int = 123,
        reply_to: object | None = None,
        client: FakeClient | None = None,
    ) -> None:
        self.raw_text = raw_text
        self.file = file
        self.video = video
        self.voice = voice
        self.media_bytes = media_bytes
        self.is_private = is_private
        self.is_group = is_group
        self.is_channel = is_channel
        self.id = id
        self.chat_id = chat_id
        self.sender_id = sender_id
        self.reply_to = reply_to
        self.client = client or FakeClient()
        self.text_replies: list[str] = []
        self.status_replies: list[FakeStatus] = []
        self.downloads: list[str] = []

    async def reply(self, text: str) -> object:
        self.text_replies.append(text)
        status = FakeStatus()
        self.status_replies.append(status)
        return status

    async def download_media(self, *, file: str) -> str:
        self.downloads.append(file)
        Path(file).write_bytes(self.media_bytes)
        return file

    async def get_input_chat(self) -> str:
        return f"chat:{self.chat_id}"


class FakeStatus:
    def __init__(self) -> None:
        self.edits: list[str] = []

    async def edit(self, text: str) -> None:
        self.edits.append(text)


def make_settings(**kwargs: object) -> Settings:
    return Settings(
        telegram_bot_token="token",
        telegram_api_id=12345,
        telegram_api_hash="hash",
        openai_api_key="key",
        **kwargs,
    )


def make_state(
    *,
    settings: Settings | None = None,
    transcriber: object | None = None,
    audio_tempo: float | None = None,
) -> BotState:
    settings = settings or make_settings()
    return BotState(
        settings=settings,
        transcriber=transcriber or object(),
        job_semaphore=asyncio.Semaphore(settings.max_concurrent_jobs),
        audio_tempo=settings.audio_tempo if audio_tempo is None else audio_tempo,
    )


def video_file(*, size: int | None = 1, name: str = "clip.mp4", mime_type: str = "video/mp4") -> object:
    return SimpleNamespace(size=size, name=name, mime_type=mime_type)


def audio_file(*, size: int | None = 1, name: str = "clip.mp3", mime_type: str = "audio/mpeg") -> object:
    return SimpleNamespace(size=size, name=name, mime_type=mime_type)


def voice_file(*, size: int | None = 1) -> object:
    return SimpleNamespace(size=size, name="", mime_type="audio/ogg")


def group_message(**kwargs: object) -> FakeMessage:
    return FakeMessage(is_private=False, is_group=True, is_channel=False, **kwargs)


def channel_message(**kwargs: object) -> FakeMessage:
    return FakeMessage(is_private=False, is_group=False, is_channel=True, **kwargs)


def topic_reply(topic_id: int) -> object:
    return SimpleNamespace(forum_topic=True, reply_to_top_id=None, reply_to_msg_id=topic_id)


def test_is_authorized_allows_everyone_without_allowlist() -> None:
    assert is_authorized(make_settings(), user_id=123)


def test_is_authorized_checks_allowlist() -> None:
    settings = make_settings(allowed_telegram_user_ids=frozenset({123}))

    assert is_authorized(settings, user_id=123)
    assert not is_authorized(settings, user_id=999)


def test_is_video_document_accepts_video_mime_type() -> None:
    document = SimpleNamespace(mime_type="video/mp4", name="upload.bin")

    assert is_video_document(document)


def test_is_video_document_accepts_video_extension() -> None:
    document = SimpleNamespace(mime_type="application/octet-stream", name="clip.mov")

    assert is_video_document(document)


def test_get_media_attachment_accepts_telethon_video_message() -> None:
    message = FakeMessage(video=SimpleNamespace(size=1), file=video_file())

    assert get_media_attachment(message) is message


def test_get_media_attachment_accepts_video_file_message() -> None:
    message = FakeMessage(file=video_file(mime_type="application/octet-stream", name="clip.mkv"))

    assert get_media_attachment(message) is message


def test_get_attachment_suffix_defaults_to_mp4() -> None:
    assert get_attachment_suffix(FakeMessage(file=video_file(name="clip.txt"))) == ".mp4"


def test_is_audio_document_accepts_mp3_mime_type() -> None:
    assert is_audio_document(SimpleNamespace(mime_type="audio/mpeg", name="upload.bin"))


def test_is_audio_document_accepts_mp3_extension() -> None:
    assert is_audio_document(SimpleNamespace(mime_type="application/octet-stream", name="clip.mp3"))


def test_is_audio_document_rejects_other_audio() -> None:
    assert not is_audio_document(SimpleNamespace(mime_type="audio/wav", name="clip.wav"))
    assert not is_audio_document(SimpleNamespace(mime_type="audio/ogg", name="note.oga"))


def test_is_audio_message_accepts_voice_note() -> None:
    message = FakeMessage(voice=SimpleNamespace(size=1), file=voice_file())

    assert is_audio_message(message)
    assert get_media_attachment(message) is message


def test_get_media_attachment_accepts_mp3_file_message() -> None:
    message = FakeMessage(file=audio_file())

    assert get_media_attachment(message) is message


def test_get_audio_suffix_returns_mp3_for_mp3_file() -> None:
    assert get_audio_suffix(FakeMessage(file=audio_file())) == ".mp3"


def test_get_audio_suffix_returns_ogg_for_voice_note() -> None:
    assert get_audio_suffix(FakeMessage(voice=SimpleNamespace(size=1), file=voice_file())) == ".ogg"


def test_get_message_topic_id_uses_forum_topic_reply_id() -> None:
    message = group_message(reply_to=topic_reply(8))

    assert get_message_topic_id(message) == 8


@pytest.mark.asyncio
async def test_send_transcript_sends_short_text() -> None:
    message = FakeMessage()

    await send_transcript(message, "hello")

    assert message.text_replies == ["hello"]
    assert message.client.sent_files == []


@pytest.mark.asyncio
async def test_send_transcript_sends_long_text_as_document() -> None:
    message = group_message(id=99, chat_id=7)

    await send_transcript(message, "x" * 4000)

    assert message.text_replies == []
    assert len(message.client.sent_files) == 1
    sent = message.client.sent_files[0]
    assert sent["entity"] == "chat:7"
    assert sent["kwargs"]["caption"] == "Transcript"
    assert sent["kwargs"]["force_document"] is True
    assert sent["kwargs"]["reply_to"] == 99


@pytest.mark.asyncio
async def test_send_transcription_result_sends_transcription_then_refined_message() -> None:
    message = FakeMessage()

    await send_transcription_result(message, TranscriptionResult("raw transcript", "refined message"))

    assert message.text_replies == ["raw transcript", "refined message"]
    assert message.client.sent_files == []


@pytest.mark.asyncio
async def test_send_transcription_result_sends_long_parts_as_separate_documents() -> None:
    message = group_message(id=99, chat_id=7)

    await send_transcription_result(message, TranscriptionResult("x" * 4000, "y" * 4000))

    assert message.text_replies == []
    assert len(message.client.sent_files) == 2
    transcription, refined = message.client.sent_files
    assert getattr(transcription["file"], "name") == "transcription.txt"
    assert transcription["kwargs"]["caption"] == "Transcription"
    assert transcription["kwargs"]["reply_to"] == 99
    assert getattr(refined["file"], "name") == "refined_message.txt"
    assert refined["kwargs"]["caption"] == "Refined message"
    assert refined["kwargs"]["reply_to"] == 99


@pytest.mark.asyncio
async def test_handle_non_media_ignores_messages() -> None:
    message = group_message()

    await handle_non_media(message, make_state())

    assert message.text_replies == []


@pytest.mark.asyncio
async def test_handle_tempo_command_updates_runtime_tempo_in_group() -> None:
    message = group_message(id=123, reply_to=topic_reply(8))
    state = make_state(
        settings=make_settings(
            allowed_telegram_user_ids=frozenset({999}),
            allowed_telegram_topic_id=8,
        ),
        audio_tempo=1.0,
    )

    await handle_tempo_command(message, state, ["1.2"])

    assert state.audio_tempo == 1.2
    assert message.text_replies == ["Tempo set to 1.2x."]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        FakeMessage(),
        channel_message(),
    ],
)
async def test_handle_tempo_command_ignores_private_and_channel_chats(message: FakeMessage) -> None:
    state = make_state(audio_tempo=1.0)

    await handle_tempo_command(message, state, ["1.2"])

    assert state.audio_tempo == 1.0
    assert message.text_replies == []


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [[], ["fast"], ["1.0", "extra"], ["0.49"], ["2.01"]])
async def test_handle_tempo_command_ignores_invalid_values(args: list[str]) -> None:
    message = group_message()
    state = make_state(audio_tempo=1.0)

    await handle_tempo_command(message, state, args)

    assert state.audio_tempo == 1.0
    assert message.text_replies == []


@pytest.mark.asyncio
async def test_handle_tempo_command_ignores_other_group_topics() -> None:
    message = group_message(reply_to=topic_reply(9))
    state = make_state(settings=make_settings(allowed_telegram_topic_id=8), audio_tempo=1.0)

    await handle_tempo_command(message, state, ["1.2"])

    assert state.audio_tempo == 1.0
    assert message.text_replies == []


@pytest.mark.asyncio
async def test_handle_noise_command_ignores_no_arg_default_noise_command() -> None:
    message = group_message(chat_id=20, reply_to=topic_reply(8))
    state = make_state()

    await handle_noise_command(message, state, [])

    assert state.pending_noise_reductions == {}
    assert message.text_replies == []


@pytest.mark.asyncio
async def test_handle_noise_command_sets_extra_noise_for_group_thread() -> None:
    message = group_message(chat_id=20, reply_to=topic_reply(8))
    state = make_state()

    await handle_noise_command(message, state, ["extra"])

    assert state.pending_noise_reductions == {(20, 8): "extra"}
    assert message.text_replies == ["Extra noise reduction set for the next video."]


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [["fast"], ["extra", "now"]])
async def test_handle_noise_command_ignores_invalid_values(args: list[str]) -> None:
    message = group_message()
    state = make_state()

    await handle_noise_command(message, state, args)

    assert state.pending_noise_reductions == {}
    assert message.text_replies == []


@pytest.mark.asyncio
async def test_handle_noise_command_ignores_unauthorized_users() -> None:
    message = group_message(sender_id=999)
    state = make_state(settings=make_settings(allowed_telegram_user_ids=frozenset({123})))

    await handle_noise_command(message, state, [])

    assert state.pending_noise_reductions == {}
    assert message.text_replies == []


@pytest.mark.asyncio
async def test_handle_noise_command_ignores_other_group_topics() -> None:
    message = group_message(reply_to=topic_reply(9))
    state = make_state(settings=make_settings(allowed_telegram_topic_id=8))

    await handle_noise_command(message, state, [])

    assert state.pending_noise_reductions == {}
    assert message.text_replies == []


@pytest.mark.asyncio
async def test_handle_media_upload_rejects_unauthorized_user() -> None:
    message = group_message(file=video_file(), sender_id=999)
    state = make_state(settings=make_settings(allowed_telegram_user_ids=frozenset({123})))

    await handle_media_upload(message, state)

    assert "not enabled" in message.text_replies[0]


@pytest.mark.asyncio
async def test_handle_media_upload_ignores_non_video_document() -> None:
    message = FakeMessage(file=video_file(mime_type="text/plain", name="notes.txt"))
    state = make_state()

    await handle_media_upload(message, state)

    assert message.text_replies == []


@pytest.mark.asyncio
async def test_handle_media_upload_ignores_oversized_video() -> None:
    message = FakeMessage(file=video_file(size=11))
    state = make_state(settings=make_settings(max_video_mb=10 / 1024 / 1024))

    await handle_media_upload(message, state)

    assert message.text_replies == []


@pytest.mark.asyncio
async def test_handle_media_upload_allows_default_two_gib_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    size = mb_to_bytes(2048)
    message = FakeMessage(file=video_file(size=size))
    state = make_state(settings=make_settings())
    processed = False

    async def fake_process_media_message(*args: object) -> None:
        nonlocal processed
        processed = True
        assert args[1] is message

    monkeypatch.setattr(bot_module, "process_media_message", fake_process_media_message)

    await handle_media_upload(message, state)

    assert processed
    assert message.text_replies == []


@pytest.mark.asyncio
async def test_handle_media_upload_accepts_video_at_exact_configured_size(monkeypatch: pytest.MonkeyPatch) -> None:
    message = group_message(file=video_file(size=10), reply_to=topic_reply(8))
    state = make_state(settings=make_settings(max_video_mb=10 / 1024 / 1024, allowed_telegram_topic_id=8), audio_tempo=1.4)
    processed = False

    async def fake_process_media_message(*args: object) -> None:
        nonlocal processed
        processed = True
        assert args[0] is message
        assert args[1] is message
        assert args[6] == 1.4
        assert args[7] == "default"
        state.audio_tempo = 2.0

    monkeypatch.setattr(bot_module, "process_media_message", fake_process_media_message)

    await handle_media_upload(message, state)

    assert processed
    assert message.text_replies == []


@pytest.mark.asyncio
async def test_handle_media_upload_consumes_pending_noise_for_matching_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = group_message(file=video_file(), chat_id=20, reply_to=topic_reply(8))
    state = make_state()
    state.pending_noise_reductions[(20, 8)] = "extra"
    processed = False

    async def fake_process_media_message(*args: object) -> None:
        nonlocal processed
        processed = True
        assert args[7] == "extra"

    monkeypatch.setattr(bot_module, "process_media_message", fake_process_media_message)

    await handle_media_upload(message, state)

    assert processed
    assert state.pending_noise_reductions == {}


@pytest.mark.asyncio
async def test_handle_media_upload_uses_default_noise_without_pending_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = group_message(file=video_file(), chat_id=20, reply_to=topic_reply(8))
    state = make_state()
    processed = False

    async def fake_process_media_message(*args: object) -> None:
        nonlocal processed
        processed = True
        assert args[7] == "default"

    monkeypatch.setattr(bot_module, "process_media_message", fake_process_media_message)

    await handle_media_upload(message, state)

    assert processed


@pytest.mark.asyncio
async def test_handle_media_upload_does_not_consume_pending_extra_noise_for_other_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = group_message(file=video_file(), chat_id=20, reply_to=topic_reply(9))
    state = make_state()
    state.pending_noise_reductions[(20, 8)] = "extra"
    processed = False

    async def fake_process_media_message(*args: object) -> None:
        nonlocal processed
        processed = True
        assert args[7] == "default"

    monkeypatch.setattr(bot_module, "process_media_message", fake_process_media_message)

    await handle_media_upload(message, state)

    assert processed
    assert state.pending_noise_reductions == {(20, 8): "extra"}


@pytest.mark.asyncio
async def test_handle_media_upload_does_not_consume_pending_noise_for_known_oversized_video() -> None:
    message = group_message(file=video_file(size=11), chat_id=20, reply_to=topic_reply(8))
    state = make_state(settings=make_settings(max_video_mb=10 / 1024 / 1024))
    state.pending_noise_reductions[(20, 8)] = "extra"

    await handle_media_upload(message, state)

    assert state.pending_noise_reductions == {(20, 8): "extra"}
    assert message.text_replies == []


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_to", [topic_reply(9), None])
async def test_handle_media_upload_ignores_restricted_group_topics(
    monkeypatch: pytest.MonkeyPatch,
    reply_to: object | None,
) -> None:
    message = group_message(file=video_file(), reply_to=reply_to)
    state = make_state(settings=make_settings(allowed_telegram_topic_id=8))

    async def fail_process_media_message(*_: object) -> None:
        raise AssertionError("off-topic group videos should not be processed")

    monkeypatch.setattr(bot_module, "process_media_message", fail_process_media_message)

    await handle_media_upload(message, state)

    assert message.text_replies == []


@pytest.mark.asyncio
async def test_handle_media_upload_allows_private_chat_when_topic_restricted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = FakeMessage(file=video_file())
    state = make_state(settings=make_settings(allowed_telegram_topic_id=8))
    processed = False

    async def fake_process_media_message(*args: object) -> None:
        nonlocal processed
        processed = True
        assert args[0] is message
        assert args[1] is message

    monkeypatch.setattr(bot_module, "process_media_message", fake_process_media_message)

    await handle_media_upload(message, state)

    assert processed
    assert message.text_replies == []


@pytest.mark.asyncio
async def test_handle_new_message_routes_commands_and_video(monkeypatch: pytest.MonkeyPatch) -> None:
    state = make_state(audio_tempo=1.0)
    tempo = group_message(raw_text="/tempo@mybot 1.5")
    video = FakeMessage(file=video_file())
    processed = False

    async def fake_process_media_message(*_: object) -> None:
        nonlocal processed
        processed = True

    monkeypatch.setattr(bot_module, "process_media_message", fake_process_media_message)

    await handle_new_message(SimpleNamespace(message=tempo), state)
    await handle_new_message(SimpleNamespace(message=video), state)

    assert state.audio_tempo == 1.5
    assert processed


@pytest.mark.asyncio
async def test_process_media_message_reports_step_by_step_flow(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FakeTranscriber:
        async def transcribe_chunks_async_result(
            self,
            chunks: object,
            progress_callback: object = None,
        ) -> TranscriptionResult:
            chunk = list(chunks)[0]
            assert progress_callback is not None
            await progress_callback(
                "transcribing_chunk",
                {
                    "index": 1,
                    "total": 1,
                    "chunk_bytes": chunk.stat().st_size,
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
            return TranscriptionResult("هاي خام", "هاي مرتبة")

    def fake_extract_audio(
        video_path: Path,
        audio_path: Path,
        *,
        audio_tempo: float,
        noise_reduction_filter: str | None = None,
    ) -> Path:
        assert video_path.exists()
        assert audio_tempo == 1.4
        assert noise_reduction_filter == "anlmdn"
        audio_path.write_bytes(b"audio")
        return audio_path

    monkeypatch.setattr(bot_module, "extract_audio", fake_extract_audio)
    caplog.set_level("INFO", logger="telegram_transcript.bot")
    message = FakeMessage(file=video_file(size=5), media_bytes=b"video")
    status_ref: dict[str, object] = {"message": None}
    settings = make_settings()
    state = make_state(settings=settings, transcriber=FakeTranscriber())

    await process_media_message(message, message, settings, state, status_ref, "job1234", 1.4)

    assert message.downloads
    assert message.text_replies == ["Video received!", "هاي خام", "هاي مرتبة"]
    status = message.status_replies[0]
    assert status.edits == [
        "Downloading...",
        "Converting to MP3 at 1.4x with default noise reduction...",
        "Splitting audio...",
        "Transcribing...",
        "Refining transcript...",
        "Sending transcript...",
        "Transcript ready.",
    ]
    assert "job1234 step 1/6" in caplog.text
    assert "job1234 step 6/6" in caplog.text
    assert "هاي خام" not in caplog.text
    assert "هاي مرتبة" not in caplog.text


@pytest.mark.asyncio
async def test_process_media_message_applies_default_noise_reduction_without_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTranscriber:
        async def transcribe_chunks_async_result(
            self,
            chunks: object,
            progress_callback: object = None,
        ) -> TranscriptionResult:
            assert list(chunks)[0].name == "audio.mp3"
            return TranscriptionResult("done")

    def fake_extract_audio(
        video_path: Path,
        audio_path: Path,
        *,
        audio_tempo: float,
        noise_reduction_filter: str | None = None,
    ) -> Path:
        assert video_path.exists()
        assert audio_tempo == 1.0
        assert noise_reduction_filter == "anlmdn"
        audio_path.write_bytes(b"audio")
        return audio_path

    monkeypatch.setattr(bot_module, "extract_audio", fake_extract_audio)
    message = FakeMessage(file=video_file(size=5), media_bytes=b"video")
    status_ref: dict[str, object] = {"message": None}
    settings = make_settings()
    state = make_state(settings=settings, transcriber=FakeTranscriber())

    await process_media_message(
        message,
        message,
        settings,
        state,
        status_ref,
        "job1234",
        1.0,
    )

    assert message.status_replies[0].edits[1] == "Converting to MP3 with default noise reduction..."
    assert message.text_replies == ["Video received!", "done"]


@pytest.mark.asyncio
async def test_process_media_message_applies_extra_noise_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTranscriber:
        async def transcribe_chunks_async_result(
            self,
            chunks: object,
            progress_callback: object = None,
        ) -> TranscriptionResult:
            assert list(chunks)[0].name == "audio.mp3"
            return TranscriptionResult("done")

    def fake_extract_audio(
        video_path: Path,
        audio_path: Path,
        *,
        audio_tempo: float,
        noise_reduction_filter: str | None = None,
    ) -> Path:
        assert video_path.exists()
        assert audio_tempo == 1.0
        assert noise_reduction_filter == "highpass=f=80,afftdn=nr=15,loudnorm"
        audio_path.write_bytes(b"audio")
        return audio_path

    monkeypatch.setattr(bot_module, "extract_audio", fake_extract_audio)
    message = FakeMessage(file=video_file(size=5), media_bytes=b"video")
    status_ref: dict[str, object] = {"message": None}
    settings = make_settings()
    state = make_state(settings=settings, transcriber=FakeTranscriber())

    await process_media_message(
        message,
        message,
        settings,
        state,
        status_ref,
        "job1234",
        1.0,
        "extra",
    )

    assert message.status_replies[0].edits[1] == "Converting to MP3 with extra noise reduction..."
    assert message.text_replies == ["Video received!", "done"]


@pytest.mark.asyncio
async def test_process_media_message_rejects_downloaded_file_over_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_extract_audio(*_: object, **__: object) -> None:
        raise AssertionError("extract_audio should not run for an oversized downloaded video")

    monkeypatch.setattr(bot_module, "extract_audio", fail_extract_audio)
    message = FakeMessage(file=video_file(size=None), media_bytes=b"x" * 11)
    status_ref: dict[str, object] = {"message": None}
    settings = make_settings(max_video_mb=10 / 1024 / 1024)
    state = make_state(settings=settings, transcriber=object())

    await process_media_message(message, message, settings, state, status_ref, "job1234", 1.0)

    assert status_ref["message"] is not None
    assert message.text_replies == ["Video received!"]
    assert message.status_replies[0].edits == ["Downloading..."]


@pytest.mark.asyncio
async def test_process_media_message_skips_extraction_for_small_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTranscriber:
        async def transcribe_chunks_async_result(
            self,
            chunks: object,
            progress_callback: object = None,
        ) -> TranscriptionResult:
            assert list(chunks)[0].name == "source.mp3"
            return TranscriptionResult("صار")

    def fail_extract_audio(*_: object, **__: object) -> None:
        raise AssertionError("extract_audio should not run for already-small audio")

    monkeypatch.setattr(bot_module, "extract_audio", fail_extract_audio)
    message = FakeMessage(file=audio_file(size=5), media_bytes=b"audio")
    status_ref: dict[str, object] = {"message": None}
    settings = make_settings()
    state = make_state(settings=settings, transcriber=FakeTranscriber())

    await process_media_message(message, message, settings, state, status_ref, "job1234", 1.0, None, True)

    assert message.text_replies == ["Audio received!", "صار"]
    assert message.downloads and message.downloads[0].endswith("source.mp3")


@pytest.mark.asyncio
async def test_process_media_message_extracts_oversized_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted = False

    class FakeTranscriber:
        async def transcribe_chunks_async_result(
            self,
            chunks: object,
            progress_callback: object = None,
        ) -> TranscriptionResult:
            assert list(chunks)[0].name == "audio.mp3"
            return TranscriptionResult("done")

    def fake_extract_audio(
        source_path: Path,
        audio_path: Path,
        *,
        audio_tempo: float,
        noise_reduction_filter: str | None = None,
    ) -> Path:
        nonlocal extracted
        extracted = True
        assert source_path.exists()
        assert noise_reduction_filter == "anlmdn"
        audio_path.write_bytes(b"audio")
        return audio_path

    monkeypatch.setattr(bot_module, "extract_audio", fake_extract_audio)
    message = FakeMessage(file=audio_file(size=50), media_bytes=b"x" * 50)
    status_ref: dict[str, object] = {"message": None}
    settings = make_settings(max_openai_audio_mb=10 / 1024 / 1024)
    state = make_state(settings=settings, transcriber=FakeTranscriber())

    await process_media_message(message, message, settings, state, status_ref, "job1234", 1.0, None, True)

    assert extracted
    assert message.text_replies == ["Audio received!", "done"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        FakeMessage(file=audio_file()),
        FakeMessage(voice=SimpleNamespace(size=1), file=voice_file()),
    ],
)
async def test_handle_media_upload_routes_audio_with_is_audio_flag(
    monkeypatch: pytest.MonkeyPatch,
    message: FakeMessage,
) -> None:
    state = make_state()
    processed = False

    async def fake_process_media_message(*args: object) -> None:
        nonlocal processed
        processed = True
        assert args[8] is True

    monkeypatch.setattr(bot_module, "process_media_message", fake_process_media_message)

    await handle_media_upload(message, state)

    assert processed
