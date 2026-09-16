from __future__ import annotations

import asyncio
import warnings
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from telegram.constants import ChatType
from telegram.error import BadRequest, RetryAfter, TimedOut

from telegram_transcript import bot as bot_module
from telegram_transcript.bot import (
    get_attachment_suffix,
    get_media_attachment,
    get_media_attachment_basename,
    get_video_attachment,
    help_command,
    handle_model_command,
    handle_transcription_refinement_model_command,
    handle_translation_toggle_command,
    handle_translation_model_command,
    handle_translation_prompt_command,
    handle_non_video,
    handle_tempo_command,
    handle_video_upload,
    is_audio_document,
    is_authorized,
    is_video_document,
    process_media_message,
    process_video_message,
    send_transcript,
)
from telegram_transcript.config import ConfigError, Settings
from telegram_transcript.models import AudioChunk, TranscriptionResult, SubtitleCue
from telegram_transcript.scoped_state import ScopedStateStore
from telegram_transcript.transcriber import PERSIAN_TRANSLATION_NUMBER_WARNING


class FakeMessage:
    def __init__(
        self,
        *,
        video: object | None = None,
        audio: object | None = None,
        voice: object | None = None,
        document: object | None = None,
        chat_type: str = ChatType.PRIVATE,
        chat_id: int = 100,
        message_id: int = 42,
        message_thread_id: int | None = None,
        status: "FakeStatus | None" = None,
        from_user: SimpleNamespace | None = None,
    ) -> None:
        self.video = video
        self.audio = audio
        self.voice = voice
        self.document = document
        self.chat = SimpleNamespace(type=chat_type, id=chat_id)
        self.chat_id = chat_id
        self.message_id = message_id
        self.message_thread_id = message_thread_id
        self.text_replies: list[str] = []
        self.text_reply_kwargs: list[dict[str, object]] = []
        self.status_replies: list[FakeStatus] = []
        self.document_replies: list[object] = []
        self.document_reply_kwargs: list[dict[str, object]] = []
        self.status = status
        self.from_user = from_user

    async def reply_text(self, text: str, **kwargs: object) -> object:
        self.text_replies.append(text)
        self.text_reply_kwargs.append(kwargs)
        status = self.status or FakeStatus()
        self.status_replies.append(status)
        return status

    async def reply_document(self, *, document: object, caption: str, **kwargs: object) -> None:
        self.document_replies.append((document, caption))
        self.document_reply_kwargs.append(kwargs)


class FakeStatus:
    def __init__(self, edit_errors: list[BaseException] | None = None) -> None:
        self.edits: list[str] = []
        self.edit_errors = edit_errors or []
        self.delete_calls = 0

    async def edit_text(self, text: str) -> None:
        if self.edit_errors:
            raise self.edit_errors.pop(0)
        self.edits.append(text)

    async def delete(self) -> None:
        self.delete_calls += 1


class FakeMediaDownloader:
    def __init__(self, media_bytes: bytes = b"video") -> None:
        self.media_bytes = media_bytes
        self.downloads: list[tuple[int, int, Path]] = []

    async def download_message_media(self, chat_id: int, message_id: int, target_path: Path) -> Path:
        self.downloads.append((chat_id, message_id, target_path))
        target_path.write_bytes(self.media_bytes)
        return target_path


class FakeAdminBot:
    async def get_chat_member(self, chat_id: int, user_id: int) -> object:
        del chat_id, user_id
        return SimpleNamespace(status="administrator")


@pytest.mark.asyncio
async def test_optional_pass_commands_preserve_scopes_and_queued_settings(tmp_path: Path) -> None:
    settings = Settings(telegram_bot_token="test", deepgram_api_key="test", openrouter_api_key="test")
    defaults = bot_module.create_runtime_preferences_store(settings).defaults
    store = ScopedStateStore(tmp_path / "state.sqlite3", defaults=defaults)
    message = FakeMessage(video=SimpleNamespace(file_size=1))
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    context = SimpleNamespace(args=["qwen"], bot_data={
        "settings": settings, "scoped_state_store": store, "media_job_queue": asyncio.Queue(),
    })
    try:
        await handle_transcription_refinement_model_command(update, context)
        context.args = ["on"]
        await bot_module.handle_audio_correction_command(update, context)
        await handle_video_upload(update, context)
        queued = context.bot_data["media_job_queue"].get_nowait()
        assert queued.transcriber.transcription_refinement_model == "qwen/qwen3-30b-a3b-instruct-2507"
        assert "deepgram" in queued.transcriber.correctors_by_provider
        context.args = ["off"]
        await handle_transcription_refinement_model_command(update, context)
        await bot_module.handle_audio_correction_command(update, context)
        preferences = store.load_preferences("user:123")
        assert not preferences.transcription_refinement_enabled
        assert not preferences.audio_correction_enabled
        # The queued object is an immutable settings snapshot, not the live settings.
        assert queued.transcriber.transcription_refinement_model == preferences.transcription_refinement_model
        assert "deepgram" in queued.transcriber.correctors_by_provider
        assert store.load_preferences("user:456") == defaults
        context.args = ["on"]
        await handle_transcription_refinement_model_command(update, context)
        restored = store.load_preferences("user:123")
        assert restored.transcription_refinement_enabled
        assert restored.transcription_refinement_model == "qwen/qwen3-30b-a3b-instruct-2507"
        assert not restored.audio_correction_enabled
    finally:
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("handler", [handle_transcription_refinement_model_command, bot_module.handle_audio_correction_command])
async def test_optional_passes_require_group_administrator(tmp_path: Path, handler: object) -> None:
    class MemberBot:
        async def get_chat_member(self, chat_id: int, user_id: int) -> object:
            return SimpleNamespace(status="member")

    settings = Settings(telegram_bot_token="test", deepgram_api_key="test", openrouter_api_key="test")
    defaults = bot_module.create_runtime_preferences_store(settings).defaults
    store = ScopedStateStore(tmp_path / "state.sqlite3", defaults=defaults)
    message = FakeMessage(chat_type=ChatType.SUPERGROUP)
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    context = SimpleNamespace(args=["on"], bot=MemberBot(), bot_data={"settings": settings, "scoped_state_store": store})
    try:
        await handler(update, context)
        assert "Only chat administrators" in message.text_replies[-1]
        assert store.load_preferences("chat:100") == defaults
        context.bot = FakeAdminBot()
        await handler(update, context)
        changed = store.load_preferences("chat:100")
        assert changed != defaults
        assert store.load_preferences("user:123") == defaults
    finally:
        store.close()


def make_timed_out(cause_message: str = "network timeout") -> TimedOut:
    try:
        raise TimedOut from TimeoutError(cause_message)
    except TimedOut as exc:
        return exc


def test_create_transcriber_defaults_to_deepgram_without_extra_passes() -> None:
    transcriber = bot_module.create_transcriber(
        Settings(
            telegram_bot_token="token",
            deepgram_api_key="deepgram-key",
            openai_api_key="openai-key",
            openrouter_api_key="openrouter-key",
        )
    )

    assert transcriber.provider_name == "deepgram"
    assert transcriber.model == "nova-3"
    assert tuple(provider.provider_name for provider in transcriber.speech_to_text_providers) == (
        "deepgram",
    )
    assert transcriber.transcription_refinement_model is None
    assert transcriber.correctors_by_provider == {}


@pytest.mark.parametrize(
    ("selected", "expected"),
    [
        ('gemini', ('gemini',)),
        ('deepgram', ('deepgram',)),
        ('whisper', ('whisper',)),
        ('openai', ('openai',)),
    ],
)
def test_transcription_model_order_uses_selected_provider_first(
    selected: str,
    expected: tuple[str, ...],
) -> None:
    assert bot_module.get_transcription_model_order(selected) == expected


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        ("Whisper", "whisper"),
        (" openai/whisper-large-v3 ", "whisper"),
        ("openai", "openai"),
    ],
)
def test_parse_model_command_arg_accepts_whisper_aliases(argument: str, expected: str) -> None:
    assert bot_module.parse_model_command_arg(argument) == expected


@pytest.mark.parametrize(
    "translation_model",
    [
        "google/gemini-3.5-flash",
        "google/gemini-2.5-flash-lite",
        "qwen/qwen3-30b-a3b-instruct-2507",
    ],
)
def test_create_transcriber_uses_selected_translation_model(translation_model: str) -> None:
    transcriber = bot_module.create_transcriber(
        Settings(
            telegram_bot_token="token",
            deepgram_api_key="deepgram-key",
            openai_api_key="openai-key",
            openrouter_api_key="openrouter-key",
            refine=True,
        ),
        translation_model=translation_model,
    )

    assert transcriber.refinement_model == translation_model


@pytest.mark.parametrize(
    ("settings_enabled", "runtime_enabled", "expected_model"),
    [
        (False, True, "google/gemini-2.5-flash-lite"),
        (True, False, None),
    ],
)
def test_create_transcriber_runtime_translation_toggle_overrides_settings(
    settings_enabled: bool,
    runtime_enabled: bool,
    expected_model: str | None,
) -> None:
    transcriber = bot_module.create_transcriber(
        Settings(
            telegram_bot_token="token",
            deepgram_api_key="deepgram-key",
            openai_api_key="openai-key",
            openrouter_api_key="openrouter-key",
            refine=settings_enabled,
        ),
        translation_enabled=runtime_enabled,
    )

    assert transcriber.refinement_model == expected_model


def test_runtime_translation_default_comes_from_refine_setting(tmp_path: Path) -> None:
    settings = Settings(
        telegram_bot_token="token",
        refine=True,
        runtime_state_path=tmp_path / "runtime.json",
    )

    assert bot_module.create_runtime_preferences_store(settings).load().translation_enabled is True


def test_create_transcriber_uses_dedicated_transcription_refinement_model() -> None:
    transcriber = bot_module.create_transcriber(
        Settings(
            telegram_bot_token="token",
            deepgram_api_key="deepgram-key",
            openai_api_key="openai-key",
            openrouter_api_key="openrouter-key",
            openrouter_transcription_refinement_model="custom-arabic-refinement-model",
        ),
        transcription_refinement_enabled=True,
    )

    assert transcriber.transcription_refinement_model == "custom-arabic-refinement-model"
    assert transcriber.refinement_model is None


def test_create_transcriber_uses_runtime_transcription_refinement_model() -> None:
    transcriber = bot_module.create_transcriber(
        Settings(
            telegram_bot_token="token",
            deepgram_api_key="deepgram-key",
            openai_api_key="openai-key",
            openrouter_api_key="openrouter-key",
        ),
        transcription_refinement_model="google/gemini-3.5-flash",
        transcription_refinement_enabled=True,
    )

    assert transcriber.transcription_refinement_model == "google/gemini-3.5-flash"


def test_create_application_uses_selected_gemini_refiner_without_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = Settings(
        telegram_bot_token="123:token",
        openrouter_transcription_refinement_model="google/gemini-3.5-flash",
        runtime_state_path=tmp_path / "runtime.json",
        video_registry_path=tmp_path / "videos.sqlite3",
    )
    captured: dict[str, object] = {}
    fake_transcriber = SimpleNamespace(provider_name="gemini", model="gemini")

    def fake_create_transcriber(
        settings_arg: Settings,
        model_key: str,
        **kwargs: object,
    ) -> object:
        assert settings_arg is settings
        captured.update(kwargs)
        return fake_transcriber

    monkeypatch.setattr(bot_module, "create_transcriber", fake_create_transcriber)

    application = bot_module.create_application(settings)
    try:
        assert captured["transcription_refinement_model"] == "google/gemini-3.5-flash"
        assert application.bot_data["transcriber"] is fake_transcriber
        assert "effective_transcription_refinement_model" not in application.bot_data
    finally:
        application.bot_data["scoped_state_store"].close()


def test_create_application_configures_telegram_request_timeouts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = Settings(
        telegram_bot_token="123:token",
        openrouter_api_key="key",
        telegram_request_timeout_seconds=45.0,
        telegram_media_write_timeout_seconds=180.0,
        runtime_state_path=tmp_path / "runtime.json",
        video_registry_path=tmp_path / "videos.sqlite3",
        scoped_state_path=tmp_path / "state.sqlite3",
    )
    monkeypatch.setattr(
        bot_module,
        "create_transcriber",
        lambda *args, **kwargs: SimpleNamespace(provider_name="gemini", model="gemini"),
    )

    application = bot_module.create_application(settings)
    try:
        request = application.bot.request
        assert request.read_timeout == 45.0
        assert request._client.timeout.connect == 45.0
        assert request._client.timeout.write == 45.0
        assert request._client.timeout.pool == 45.0
        assert request._media_write_timeout == 180.0
    finally:
        application.bot_data["scoped_state_store"].close()


def test_create_application_automatically_deletes_legacy_registry(tmp_path: Path) -> None:
    registry = tmp_path / "videos.sqlite3"
    Path(f"{registry}-wal").write_bytes(b"legacy wal")
    registry.write_bytes(b"legacy registry")
    settings = Settings(
        telegram_bot_token="123:token",
        openrouter_api_key="key",
        deepgram_api_key="key",
        runtime_state_path=tmp_path / "runtime.json",
        video_registry_path=registry,
        scoped_state_path=tmp_path / "state.sqlite3",
    )

    application = bot_module.create_application(settings)
    try:
        assert not registry.exists()
        assert not Path(f"{registry}-wal").exists()
    finally:
        application.bot_data["scoped_state_store"].close()


def test_purge_configured_legacy_registry_preserves_shared_preferences(tmp_path: Path) -> None:
    database = tmp_path / "shared.sqlite3"
    connection = bot_module.sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE videos (transcript TEXT)")
        connection.execute("INSERT INTO videos VALUES ('private transcript')")
        connection.execute("CREATE TABLE retained_preferences (value TEXT)")
        connection.execute("INSERT INTO retained_preferences VALUES ('keep')")
        connection.commit()
    finally:
        connection.close()
    settings = Settings(
        telegram_bot_token="token",
        runtime_state_path=tmp_path / "runtime.json",
        video_registry_path=database,
        scoped_state_path=database,
    )

    bot_module.purge_configured_legacy_registry(settings)

    connection = bot_module.sqlite3.connect(database)
    try:
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'videos'"
        ).fetchone() is None
        assert connection.execute("SELECT value FROM retained_preferences").fetchone() == ("keep",)
    finally:
        connection.close()
    assert b"private transcript" not in database.read_bytes()


def test_purge_configured_legacy_registry_rejects_runtime_state_overlap(tmp_path: Path) -> None:
    shared_path = tmp_path / "shared-state"
    settings = Settings(
        telegram_bot_token="token",
        runtime_state_path=shared_path,
        video_registry_path=shared_path,
        scoped_state_path=tmp_path / "scoped.sqlite3",
    )

    with pytest.raises(ConfigError, match="must not overlap"):
        bot_module.purge_configured_legacy_registry(settings)


def test_purge_configured_legacy_registry_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def deny_delete(path: Path) -> bool:
        del path
        raise PermissionError("private path details")

    monkeypatch.setattr(bot_module, "delete_legacy_registry_files", deny_delete)
    settings = Settings(
        telegram_bot_token="token",
        runtime_state_path=tmp_path / "runtime.json",
        video_registry_path=tmp_path / "videos.sqlite3",
        scoped_state_path=tmp_path / "state.sqlite3",
    )

    with pytest.raises(ConfigError, match="Unable to purge") as error:
        bot_module.purge_configured_legacy_registry(settings)

    assert isinstance(error.value.__cause__, PermissionError)


@pytest.mark.asyncio
async def test_start_media_downloader_stores_started_downloader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        telegram_bot_token="token",
        deepgram_api_key="deepgram-key",
        openai_api_key="openai-key",
        openrouter_api_key="openrouter-key",
    )

    class Downloader:
        def __init__(self, settings_arg: Settings) -> None:
            assert settings_arg is settings
            self.started = False

        async def start(self) -> None:
            self.started = True

    monkeypatch.setattr(bot_module, "TelegramMediaDownloader", Downloader)
    application = SimpleNamespace(bot_data={"settings": settings})

    await bot_module.start_media_downloader(application)

    assert application.bot_data["media_downloader"].started is True


@pytest.mark.asyncio
async def test_media_worker_releases_job_references_and_sanitizes_failure_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sensitive_error = "private-provider-response-sentinel"

    async def fail_processing(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise bot_module.TranscriptionError(sensitive_error)

    monkeypatch.setattr(bot_module, "process_media_message", fail_processing)
    caplog.set_level("ERROR", logger="telegram_transcript.bot")
    queue: asyncio.Queue[bot_module.QueuedMediaJob] = asyncio.Queue()
    status = FakeStatus()
    status_ref = {"message": status}
    settings = Settings(telegram_bot_token="token")
    job = bot_module.QueuedMediaJob(
        message=FakeMessage(),
        attachment=SimpleNamespace(),
        settings=settings,
        context=SimpleNamespace(),
        status_ref=status_ref,
        job_id="anonymous-job",
        audio_tempo=1.0,
        transcriber=SimpleNamespace(),
        translation_enabled=False,
    )
    application = SimpleNamespace(bot_data={"media_job_queue": queue})
    worker = asyncio.create_task(bot_module.media_job_worker(application, 0))
    await queue.put(job)

    await asyncio.wait_for(queue.join(), timeout=1)
    worker.cancel()
    await asyncio.gather(worker, return_exceptions=True)

    assert status_ref["message"] is None
    assert status.edits == [f"Transcription failed: {sensitive_error}"]
    assert "error_type=TranscriptionError" in caplog.text
    assert sensitive_error not in caplog.text


@pytest.mark.asyncio
async def test_discard_queued_media_jobs_releases_pending_status_references() -> None:
    queue: asyncio.Queue[bot_module.QueuedMediaJob] = asyncio.Queue()
    status_refs: list[dict[str, object | None]] = []
    for index in range(2):
        status_ref: dict[str, object | None] = {"message": FakeStatus()}
        status_refs.append(status_ref)
        await queue.put(
            bot_module.QueuedMediaJob(
                message=FakeMessage(),
                attachment=SimpleNamespace(),
                settings=Settings(telegram_bot_token="token"),
                context=SimpleNamespace(),
                status_ref=status_ref,  # type: ignore[arg-type]
                job_id=f"job-{index}",
                audio_tempo=1.0,
                transcriber=SimpleNamespace(),
                translation_enabled=False,
            )
        )

    assert bot_module.discard_queued_media_jobs(queue) == 2
    await asyncio.wait_for(queue.join(), timeout=1)

    assert all(status_ref["message"] is None for status_ref in status_refs)
    assert queue.empty()


def test_unrelated_runtime_save_preserves_selected_gemini(
    tmp_path: Path,
) -> None:
    settings = Settings(
        telegram_bot_token="token",
        openrouter_api_key="key",
        runtime_state_path=tmp_path / "runtime.json",
    )
    context = SimpleNamespace(
        bot_data={
            "settings": settings,
            "transcription_refinement_model": "google/gemini-3.5-flash",
            "runtime_preferences_store": bot_module.create_runtime_preferences_store(settings),
        }
    )

    bot_module.persist_runtime_preferences(context, audio_tempo=1.2)

    persisted = context.bot_data["runtime_preferences_store"].load()
    assert persisted.audio_tempo == 1.2
    assert persisted.transcription_refinement_model == "google/gemini-3.5-flash"


def test_is_authorized_allows_everyone_without_allowlist() -> None:
    settings = Settings(telegram_bot_token="token", openrouter_api_key="key")

    assert is_authorized(settings, user_id=123)


def test_is_authorized_checks_allowlist() -> None:
    settings = Settings(
        telegram_bot_token="token",
        openrouter_api_key="key",
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


def test_is_audio_document_accepts_audio_mime_type() -> None:
    document = SimpleNamespace(mime_type="audio/mpeg", file_name="upload.bin")

    assert is_audio_document(document)


def test_get_media_attachment_accepts_voice() -> None:
    voice = SimpleNamespace(file_size=1, mime_type="audio/ogg")
    message = FakeMessage(voice=voice)

    assert get_media_attachment(message) is voice


def test_get_media_attachment_accepts_audio_document() -> None:
    document = SimpleNamespace(mime_type="application/octet-stream", file_name="clip.mp3")
    message = FakeMessage(document=document)

    assert get_media_attachment(message) is document


def test_get_video_attachment_prefers_video() -> None:
    video = SimpleNamespace(file_size=1)
    document = SimpleNamespace(mime_type="video/mp4", file_name="clip.mp4")
    message = SimpleNamespace(video=video, document=document)

    assert get_video_attachment(message) is video


def test_get_attachment_suffix_defaults_to_mp4() -> None:
    assert get_attachment_suffix(SimpleNamespace(file_name="clip.txt")) == ".mp4"


def test_get_media_attachment_basename_uses_original_stem() -> None:
    assert get_media_attachment_basename(SimpleNamespace(file_name="My Clip.mp4")) == "My Clip"


def test_get_media_attachment_basename_falls_back_without_file_name() -> None:
    assert get_media_attachment_basename(SimpleNamespace(mime_type="audio/ogg")) == "transcript"


def test_get_media_attachment_basename_falls_back_for_blank_stem() -> None:
    assert get_media_attachment_basename(SimpleNamespace(file_name="  .mp4")) == "transcript"


@pytest.mark.asyncio
async def test_help_command_lists_commands_and_schedules_deletion(monkeypatch: pytest.MonkeyPatch) -> None:
    message = FakeMessage(chat_type=ChatType.SUPERGROUP, message_id=123, message_thread_id=8)
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    scheduled: list[tuple[object, object, str]] = []

    async def skip_sleep(delay_seconds: float) -> None:
        assert delay_seconds == bot_module.HELP_MESSAGE_DELETE_DELAY_SECONDS

    monkeypatch.setattr(bot_module.asyncio, "sleep", skip_sleep)

    def create_task(coroutine: object, *, update: object, name: str) -> None:
        scheduled.append((coroutine, update, name))

    context = SimpleNamespace(application=SimpleNamespace(create_task=create_task))

    await help_command(update, context)

    reply = message.text_replies[0]
    assert "/help - Show this help message" in reply
    assert "/tempo <0.5-2.0>" in reply
    assert "/model [gemini|deepgram|whisper|openai]" in reply
    assert "/refiner [on|off|lite|qwen|gemini]" in reply
    assert "/translate - Toggle Persian translation" in reply
    assert "/tmodel [lite|qwen|gemini]" in reply
    assert "/translation [natural|literal]" in reply
    assert message.text_reply_kwargs == [
        {
            "reply_to_message_id": 123,
            "allow_sending_without_reply": True,
            "message_thread_id": 8,
        }
    ]
    assert scheduled[0][1:] == (update, "delete-help-message")

    await scheduled[0][0]
    assert message.status_replies[0].delete_calls == 1


@pytest.mark.asyncio
async def test_delete_message_after_delay_ignores_telegram_deletion_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def skip_sleep(delay_seconds: float) -> None:
        assert delay_seconds == bot_module.HELP_MESSAGE_DELETE_DELAY_SECONDS

    class UndeletableMessage:
        async def delete(self) -> None:
            raise BadRequest("message already deleted")

    monkeypatch.setattr(bot_module.asyncio, "sleep", skip_sleep)

    await bot_module.delete_message_after_delay(
        UndeletableMessage(), bot_module.HELP_MESSAGE_DELETE_DELAY_SECONDS
    )


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
async def test_send_transcript_formats_long_document_text(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeInputFile:
        def __init__(self, file: object, *, filename: str, read_file_handle: bool) -> None:
            assert read_file_handle is True
            self.filename = filename
            self.content = file.getvalue()

    monkeypatch.setattr(bot_module, "InputFile", FakeInputFile)
    message = FakeMessage()

    await send_transcript(message, "word " * 1000)

    document, _ = message.document_replies[0]
    assert document.filename == "transcript.txt"
    body = document.content.decode("utf-8")
    assert "\n" in body
    assert all(len(line) <= 100 for line in body.splitlines())


@pytest.mark.asyncio
async def test_document_delivery_closes_transient_output_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeInputFile:
        def __init__(self, file: object, *, filename: str, read_file_handle: bool) -> None:
            assert read_file_handle is True
            self.file = file
            self.filename = filename

    monkeypatch.setattr(bot_module, "InputFile", FakeInputFile)
    message = FakeMessage()

    await send_transcript(message, "x" * 4000)
    await bot_module.send_srt(message, "1\n00:00:00,000 --> 00:00:01,000\ntext\n")

    assert all(document.file.closed for document, _ in message.document_replies)


@pytest.mark.asyncio
async def test_retry_telegram_request_retries_timeouts_with_sanitized_logging(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempts = 0
    delays: list[float] = []

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise make_timed_out("private-request-details")
        return "delivered"

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(bot_module.asyncio, "sleep", record_sleep)
    caplog.set_level("WARNING", logger="telegram_transcript.bot")

    result = await bot_module.retry_telegram_request(
        operation,
        operation_name="send transcript",
        job_id="job1234",
    )

    assert result == "delivered"
    assert attempts == 3
    assert delays == [1.0, 2.0]
    assert "cause_type=TimeoutError" in caplog.text
    assert "private-request-details" not in caplog.text


@pytest.mark.asyncio
async def test_retry_telegram_request_raises_after_bounded_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise TimedOut

    async def skip_sleep(delay: float) -> None:
        assert delay in bot_module.TELEGRAM_RETRY_DELAYS_SECONDS

    monkeypatch.setattr(bot_module.asyncio, "sleep", skip_sleep)

    with pytest.raises(TimedOut):
        await bot_module.retry_telegram_request(operation, operation_name="send text")

    assert attempts == bot_module.TELEGRAM_REQUEST_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_retry_telegram_request_does_not_retry_other_errors() -> None:
    attempts = 0

    async def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise BadRequest("invalid request")

    with pytest.raises(BadRequest):
        await bot_module.retry_telegram_request(operation, operation_name="send text")

    assert attempts == 1


@pytest.mark.asyncio
async def test_document_delivery_recreates_buffer_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffers: list[object] = []
    payloads: list[bytes] = []

    class FakeInputFile:
        def __init__(self, file: object, *, filename: str, read_file_handle: bool) -> None:
            assert filename == "transcript.txt"
            assert read_file_handle is True
            buffers.append(file)
            payloads.append(file.getvalue())

    class TimeoutThenSuccessMessage(FakeMessage):
        async def reply_document(self, *, document: object, caption: str, **kwargs: object) -> None:
            del document, caption, kwargs
            if len(buffers) == 1:
                raise TimedOut

    async def skip_sleep(delay: float) -> None:
        assert delay == 1.0

    monkeypatch.setattr(bot_module, "InputFile", FakeInputFile)
    monkeypatch.setattr(bot_module.asyncio, "sleep", skip_sleep)

    await send_transcript(TimeoutThenSuccessMessage(), "x" * 4000)

    assert len(buffers) == 2
    assert buffers[0] is not buffers[1]
    assert payloads[0] == payloads[1]
    assert all(buffer.closed for buffer in buffers)


@pytest.mark.asyncio
async def test_status_edit_timeout_is_non_fatal_after_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = FakeStatus(edit_errors=[TimedOut(), TimedOut(), TimedOut()])

    async def skip_sleep(delay: float) -> None:
        assert delay in bot_module.TELEGRAM_RETRY_DELAYS_SECONDS

    monkeypatch.setattr(bot_module.asyncio, "sleep", skip_sleep)

    updated = await bot_module.edit_status_message(status, "Working...", job_id="job1234")

    assert updated is False
    assert status.edit_errors == []


@pytest.mark.asyncio
async def test_handle_non_video_ignores_group_messages() -> None:
    message = FakeMessage(chat_type=ChatType.GROUP)
    update = SimpleNamespace(effective_message=message)

    await handle_non_video(update, SimpleNamespace())

    assert message.text_replies == []


@pytest.mark.asyncio
async def test_privacy_command_describes_preference_only_storage() -> None:
    message = FakeMessage()
    update = SimpleNamespace(effective_message=message)

    await bot_module.handle_privacy_command(update, SimpleNamespace())

    assert len(message.text_replies) == 1
    privacy_text = message.text_replies[0]
    assert "does not retain media" in privacy_text
    assert "job metadata" in privacy_text
    assert "Anonymous operational metrics" in privacy_text
    assert "seven days" not in privacy_text


@pytest.mark.asyncio
async def test_forget_command_deletes_only_scoped_preferences(tmp_path: Path) -> None:
    settings = Settings(telegram_bot_token="token")
    defaults = bot_module.create_runtime_preferences_store(settings).defaults
    store = bot_module.ScopedStateStore(tmp_path / "state.sqlite3", defaults=defaults)
    store.save_preferences("user:123", defaults)
    message = FakeMessage()
    update = SimpleNamespace(
        effective_message=message,
        effective_user=SimpleNamespace(id=123),
    )
    context = SimpleNamespace(bot_data={"scoped_state_store": store})

    try:
        await bot_module.handle_forget_command(update, context)
        connection = bot_module.sqlite3.connect(store.path)
        try:
            assert connection.execute("SELECT COUNT(*) FROM scope_preferences").fetchone() == (0,)
        finally:
            connection.close()
    finally:
        store.close()

    assert message.text_replies == ["Your scoped preferences were deleted."]


@pytest.mark.asyncio
async def test_handle_non_video_ignores_private_messages() -> None:
    message = FakeMessage()
    update = SimpleNamespace(effective_message=message)

    await handle_non_video(update, SimpleNamespace())

    assert message.text_replies == []


@pytest.mark.asyncio
async def test_handle_tempo_command_updates_runtime_tempo_in_group(tmp_path: Path) -> None:
    message = FakeMessage(chat_type=ChatType.SUPERGROUP, message_id=123, message_thread_id=8)
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=999))
    context = SimpleNamespace(
        bot=FakeAdminBot(),
        args=["1.2"],
        bot_data={
            "audio_tempo": 1.0,
            "settings": Settings(
                telegram_bot_token="token",
                openrouter_api_key="key",
                allowed_telegram_user_ids=frozenset({123}),
                runtime_state_path=tmp_path / "runtime.json",
            ),
        },
    )
    context.bot_data["runtime_preferences_store"] = bot_module.create_runtime_preferences_store(
        context.bot_data["settings"]
    )

    await handle_tempo_command(update, context)

    assert context.bot_data["audio_tempo"] == 1.2
    assert context.bot_data["runtime_preferences_store"].load().audio_tempo == 1.2
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
async def test_handle_tempo_command_keeps_memory_unchanged_when_persistence_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    message = FakeMessage(chat_type=ChatType.GROUP)
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    settings = Settings(
        telegram_bot_token="token",
        runtime_state_path=tmp_path / "runtime.json",
    )
    store = bot_module.create_runtime_preferences_store(settings)

    def fail_save(preferences: object) -> None:
        raise bot_module.RuntimeStateError("Unable to save runtime settings.")

    monkeypatch.setattr(store, "save", fail_save)
    context = SimpleNamespace(
        bot=FakeAdminBot(),
        args=["1.2"],
        bot_data={
            "audio_tempo": 1.0,
            "settings": settings,
            "runtime_preferences_store": store,
        },
    )

    await handle_tempo_command(update, context)

    assert context.bot_data["audio_tempo"] == 1.0
    assert message.text_replies == ["Unable to save runtime settings."]


@pytest.mark.asyncio
async def test_handle_model_command_lists_current_and_available_models() -> None:
    message = FakeMessage(chat_type=ChatType.SUPERGROUP, message_id=123, message_thread_id=8)
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    context = SimpleNamespace(
        args=[],
        bot_data={
            "settings": Settings(
                telegram_bot_token="token",
                deepgram_api_key="deepgram-key",
                openai_api_key="openai-key",
                openrouter_api_key="openrouter-key",
            ),
            "transcription_model": "deepgram",
        },
    )

    await handle_model_command(update, context)

    assert "Current primary transcription model: Deepgram nova-3" in message.text_replies[0]
    assert "whisper: OpenAI: Whisper Large V3 (available)" in message.text_replies[0]
    assert "openai: Direct OpenAI whisper-1 (available)" in message.text_replies[0]
    assert "gemini: OpenRouter Gemini 3.5 Flash (available)" in message.text_replies[0]
    assert message.text_reply_kwargs == [
        {
            "reply_to_message_id": 123,
            "allow_sending_without_reply": True,
            "message_thread_id": 8,
        }
    ]


@pytest.mark.asyncio
async def test_handle_model_command_switches_runtime_transcriber(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    message = FakeMessage(chat_type=ChatType.SUPERGROUP)
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    settings = Settings(
        telegram_bot_token="token",
        deepgram_api_key="deepgram-key",
        openai_api_key="openai-key",
        openrouter_api_key="openrouter-key",
        runtime_state_path=tmp_path / "runtime.json",
    )
    context = SimpleNamespace(
        bot=FakeAdminBot(),
        args=["Whisper"],
        bot_data={
            "settings": settings,
            "transcription_refinement_model": "google/gemini-3.5-flash",
            "translation_model": "qwen/qwen3-30b-a3b-instruct-2507",
            "runtime_preferences_store": bot_module.create_runtime_preferences_store(settings),
        },
    )
    fake_transcriber = SimpleNamespace(provider_name="whisper", model="openai/whisper-large-v3")
    calls: list[tuple[Settings, str, str | None, str | None]] = []

    def fake_create_transcriber(
        settings_arg: Settings,
        model_key: str,
        translation_enabled: bool | None = None,
        transcription_refinement_enabled: bool = False,
        audio_correction_enabled: bool = False,
        translation_model: str | None = None,
        translation_prompt: str = "normal",
        transcription_refinement_model: str | None = None,
    ) -> object:
        calls.append(
            (settings_arg, model_key, translation_model, transcription_refinement_model)
        )
        return fake_transcriber

    monkeypatch.setattr(bot_module, "create_transcriber", fake_create_transcriber)

    await handle_model_command(update, context)

    assert calls == [
        (
            settings,
            "whisper",
            "qwen/qwen3-30b-a3b-instruct-2507",
            "google/gemini-3.5-flash",
        )
    ]
    assert context.bot_data["transcriber"] is fake_transcriber
    assert context.bot_data["transcription_model"] == "whisper"
    assert context.bot_data["runtime_preferences_store"].load().transcription_model == "whisper"
    assert message.text_replies == ["Primary transcription model set to OpenAI: Whisper Large V3."]


@pytest.mark.asyncio
async def test_handle_model_command_rejects_missing_provider_credentials() -> None:
    message = FakeMessage(chat_type=ChatType.SUPERGROUP)
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    context = SimpleNamespace(
        bot=FakeAdminBot(),
        args=["openai"],
        bot_data={"settings": Settings(telegram_bot_token="token", deepgram_api_key="deepgram-key")},
    )

    await handle_model_command(update, context)

    assert message.text_replies == ["OPENAI_API_KEY is required for direct OpenAI transcription."]
    assert "transcription_model" not in context.bot_data


@pytest.mark.asyncio
async def test_handle_model_command_requires_openrouter_for_whisper() -> None:
    message = FakeMessage(chat_type=ChatType.SUPERGROUP)
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    context = SimpleNamespace(
        bot=FakeAdminBot(),
        args=["whisper"],
        bot_data={"settings": Settings(telegram_bot_token="token", deepgram_api_key="deepgram-key")},
    )

    await handle_model_command(update, context)

    assert message.text_replies == ["OPENROUTER_API_KEY is required for OpenRouter Whisper transcription."]
    assert "transcription_model" not in context.bot_data


@pytest.mark.asyncio
async def test_handle_model_command_rejects_unknown_model() -> None:
    message = FakeMessage(chat_type=ChatType.SUPERGROUP)
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    context = SimpleNamespace(
        args=["unknown"],
        bot_data={"settings": Settings(telegram_bot_token="token", deepgram_api_key="deepgram-key")},
    )

    await handle_model_command(update, context)

    assert message.text_replies == [
        "Unknown transcription model. Available models: gemini, deepgram, whisper, openai."
    ]


@pytest.mark.asyncio
async def test_handle_transcription_refinement_model_command_lists_current_model() -> None:
    message = FakeMessage(message_id=123, message_thread_id=8)
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    context = SimpleNamespace(
        args=[],
        bot_data={
            "settings": Settings(
                telegram_bot_token="token",
                openrouter_api_key="openrouter-key",
            ),
            "transcription_refinement_model": "google/gemini-3.5-flash",
        },
    )

    await handle_transcription_refinement_model_command(update, context)

    reply = message.text_replies[0]
    assert "Current transcription refinement model: OpenRouter Gemini 3.5 Flash" in reply
    assert "lite: Gemini 2.5 Flash-Lite (budget default)" in reply
    assert "gemini: OpenRouter Gemini 3.5 Flash" in reply
    assert "Use /refiner on, off, lite, qwen, or gemini." in reply


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("argument", "expected_model", "expected_label"),
    [
        ("lite", "google/gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite (budget default)"),
        ("google/gemini-3.5-flash", "google/gemini-3.5-flash", "OpenRouter Gemini 3.5 Flash"),
    ],
)
async def test_handle_transcription_refinement_model_command_switches_and_persists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    argument: str,
    expected_model: str,
    expected_label: str,
) -> None:
    message = FakeMessage()
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    settings = Settings(
        telegram_bot_token="token",
        deepgram_api_key="deepgram-key",
        openrouter_api_key="openrouter-key",
        refine=True,
        runtime_state_path=tmp_path / "runtime.json",
    )
    context = SimpleNamespace(
        args=[argument],
        bot_data={
            "settings": settings,
            "transcription_model": "deepgram",
            "translation_model": "qwen/qwen3-30b-a3b-instruct-2507",
            "translation_prompt": "literal",
            "runtime_preferences_store": bot_module.create_runtime_preferences_store(settings),
        },
    )
    replacement_transcriber = SimpleNamespace(provider_name="deepgram", model="nova-3")
    calls: list[tuple[str, str | None, str, str | None]] = []

    def fake_create_transcriber(
        settings_arg: Settings,
        model_key: str,
        translation_enabled: bool | None = None,
        transcription_refinement_enabled: bool = False,
        audio_correction_enabled: bool = False,
        translation_model: str | None = None,
        translation_prompt: str = "normal",
        transcription_refinement_model: str | None = None,
    ) -> object:
        assert settings_arg is settings
        calls.append(
            (
                model_key,
                translation_model,
                translation_prompt,
                transcription_refinement_model,
            )
        )
        return replacement_transcriber

    monkeypatch.setattr(bot_module, "create_transcriber", fake_create_transcriber)

    await handle_transcription_refinement_model_command(update, context)

    assert calls == [
        ("deepgram", "qwen/qwen3-30b-a3b-instruct-2507", "literal", expected_model)
    ]
    assert context.bot_data["transcriber"] is replacement_transcriber
    assert context.bot_data["transcription_refinement_model"] == expected_model
    persisted = context.bot_data["runtime_preferences_store"].load()
    assert persisted.transcription_refinement_model == expected_model
    assert persisted.transcription_model == "deepgram"
    assert persisted.translation_model == "qwen/qwen3-30b-a3b-instruct-2507"
    assert persisted.translation_prompt == "literal"
    assert persisted.transcription_refinement_enabled is True
    assert "Iraqi refinement: on" in message.text_replies[0]
    assert expected_label in message.text_replies[0]


@pytest.mark.asyncio
async def test_handle_transcription_refinement_model_command_rejects_invalid_and_unauthorized() -> None:
    message = FakeMessage()
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    context = SimpleNamespace(
        args=["unknown"],
        bot_data={
            "settings": Settings(
                telegram_bot_token="token",
                openrouter_api_key="key",
            )
        },
    )

    await handle_transcription_refinement_model_command(update, context)

    assert message.text_replies == [
        "Use /refiner on, off, or a model: gemini, lite, qwen."
    ]

    message.text_replies.clear()
    context.bot_data["settings"] = Settings(
        telegram_bot_token="token",
        openrouter_api_key="key",
        allowed_telegram_user_ids=frozenset({999}),
    )
    context.args = ["gemini"]

    await handle_transcription_refinement_model_command(update, context)

    assert message.text_replies == ["Sorry, this bot is not enabled for your Telegram account."]


@pytest.mark.asyncio
async def test_handle_transcription_refinement_model_command_ignores_channels() -> None:
    message = FakeMessage(chat_type=ChatType.CHANNEL)
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    context = SimpleNamespace(args=["gemini"], bot_data={})

    await handle_transcription_refinement_model_command(update, context)

    assert message.text_replies == []


@pytest.mark.asyncio
async def test_handle_transcription_refinement_model_command_keeps_active_model_when_save_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    message = FakeMessage()
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    settings = Settings(
        telegram_bot_token="token",
        deepgram_api_key="deepgram-key",
        openai_api_key="openai-key",
        openrouter_api_key="openrouter-key",
        runtime_state_path=tmp_path / "runtime.json",
    )
    store = bot_module.create_runtime_preferences_store(settings)
    active_transcriber = SimpleNamespace(provider_name="gemini", model="gemini")
    context = SimpleNamespace(
        args=["gemini"],
        bot_data={
            "settings": settings,
            "transcriber": active_transcriber,
            "transcription_refinement_model": "google/gemini-2.5-flash-lite",
            "runtime_preferences_store": store,
        },
    )

    def fail_save(preferences: object) -> None:
        raise bot_module.RuntimeStateError("Unable to save runtime settings.")

    monkeypatch.setattr(store, "save", fail_save)

    await handle_transcription_refinement_model_command(update, context)

    assert context.bot_data["transcriber"] is active_transcriber
    assert context.bot_data["transcription_refinement_model"] == "google/gemini-2.5-flash-lite"
    assert message.text_replies == ["Unable to save runtime settings."]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial_enabled", "expected_enabled", "expected_reply"),
    [
        (False, True, "Translation enabled."),
        (True, False, "Translation disabled."),
    ],
)
async def test_handle_translation_toggle_command_switches_and_persists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    initial_enabled: bool,
    expected_enabled: bool,
    expected_reply: str,
) -> None:
    message = FakeMessage()
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    settings = Settings(
        telegram_bot_token="token",
        openrouter_api_key="openrouter-key",
        refine=initial_enabled,
        runtime_state_path=tmp_path / "runtime.json",
    )
    active_transcriber = SimpleNamespace(provider_name="gemini", model="gemini")
    replacement_transcriber = SimpleNamespace(provider_name="gemini", model="gemini")
    context = SimpleNamespace(
        args=[],
        bot_data={
            "settings": settings,
            "transcriber": active_transcriber,
            "translation_enabled": initial_enabled,
            "runtime_preferences_store": bot_module.create_runtime_preferences_store(settings),
        },
    )
    calls: list[bool | None] = []

    def fake_create_transcriber(
        settings_arg: Settings,
        model_key: str,
        translation_enabled: bool | None = None,
        transcription_refinement_enabled: bool = False,
        audio_correction_enabled: bool = False,
        translation_model: str | None = None,
        translation_prompt: str = "natural",
        transcription_refinement_model: str | None = None,
    ) -> object:
        assert settings_arg is settings
        calls.append(translation_enabled)
        return replacement_transcriber

    monkeypatch.setattr(bot_module, "create_transcriber", fake_create_transcriber)

    await handle_translation_toggle_command(update, context)

    assert calls == [expected_enabled]
    assert context.bot_data["transcriber"] is replacement_transcriber
    assert context.bot_data["translation_enabled"] is expected_enabled
    assert context.bot_data["runtime_preferences_store"].load().translation_enabled is expected_enabled
    assert message.text_replies == [expected_reply]


@pytest.mark.asyncio
async def test_handle_translation_toggle_command_keeps_active_state_when_save_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    message = FakeMessage()
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    settings = Settings(
        telegram_bot_token="token",
        openrouter_api_key="openrouter-key",
        runtime_state_path=tmp_path / "runtime.json",
    )
    store = bot_module.create_runtime_preferences_store(settings)
    active_transcriber = SimpleNamespace(provider_name="gemini", model="gemini")
    context = SimpleNamespace(
        args=[],
        bot_data={
            "settings": settings,
            "transcriber": active_transcriber,
            "translation_enabled": False,
            "runtime_preferences_store": store,
        },
    )

    monkeypatch.setattr(
        bot_module,
        "create_transcriber",
        lambda *args, **kwargs: SimpleNamespace(provider_name="gemini", model="gemini"),
    )

    def fail_save(preferences: object) -> None:
        raise bot_module.RuntimeStateError("Unable to save runtime settings.")

    monkeypatch.setattr(store, "save", fail_save)

    await handle_translation_toggle_command(update, context)

    assert context.bot_data["transcriber"] is active_transcriber
    assert context.bot_data["translation_enabled"] is False
    assert message.text_replies == ["Unable to save runtime settings."]


@pytest.mark.asyncio
async def test_handle_translation_toggle_command_validates_access_and_arguments() -> None:
    settings = Settings(
        telegram_bot_token="token",
        allowed_telegram_user_ids=frozenset({123}),
    )
    message = FakeMessage()
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=999))
    context = SimpleNamespace(args=[], bot_data={"settings": settings})

    await handle_translation_toggle_command(update, context)

    assert message.text_replies == ["Sorry, this bot is not enabled for your Telegram account."]

    update.effective_user.id = 123
    context.args = ["on"]
    await handle_translation_toggle_command(update, context)

    assert message.text_replies[-1] == "Usage: /translate"

    channel_message = FakeMessage(chat_type=ChatType.CHANNEL)
    await handle_translation_toggle_command(
        SimpleNamespace(effective_message=channel_message, effective_user=SimpleNamespace(id=123)),
        SimpleNamespace(args=[], bot_data={}),
    )
    assert channel_message.text_replies == []


@pytest.mark.asyncio
async def test_handle_translation_model_command_lists_current_models_and_refine_status() -> None:
    message = FakeMessage(chat_type=ChatType.SUPERGROUP, message_id=123, message_thread_id=8)
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    context = SimpleNamespace(
        args=[],
        bot_data={
            "settings": Settings(
                telegram_bot_token="token",
                openrouter_api_key="openrouter-key",
                refine=True,
            ),
            "translation_model": "google/gemini-3.5-flash",
        },
    )

    await handle_translation_model_command(update, context)

    reply = message.text_replies[0]
    assert "Current translation model: OpenRouter Gemini 3.5 Flash" in reply
    assert "Translation: enabled" in reply
    assert "lite: Gemini 2.5 Flash-Lite (budget default)" in reply
    assert "qwen: Qwen3 30B A3B Instruct (budget)" in reply
    assert message.text_reply_kwargs == [
        {
            "reply_to_message_id": 123,
            "allow_sending_without_reply": True,
            "message_thread_id": 8,
        }
    ]


@pytest.mark.asyncio
async def test_handle_translation_model_command_reports_when_refine_is_disabled() -> None:
    message = FakeMessage()
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    context = SimpleNamespace(
        args=[],
        bot_data={
            "settings": Settings(telegram_bot_token="token", openrouter_api_key="openrouter-key"),
        },
    )

    await handle_translation_model_command(update, context)

    assert "Translation: disabled" in message.text_replies[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("argument", "expected_model", "expected_label"),
    [
        ("gemini", "google/gemini-3.5-flash", "OpenRouter Gemini 3.5 Flash"),
        ("google/gemini-2.5-flash-lite", "google/gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite (budget default)"),
        ("qwen", "qwen/qwen3-30b-a3b-instruct-2507", "Qwen3 30B A3B Instruct (budget)"),
    ],
)
async def test_handle_translation_model_command_switches_model_and_preserves_transcription(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    argument: str,
    expected_model: str,
    expected_label: str,
) -> None:
    message = FakeMessage()
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    settings = Settings(
        telegram_bot_token="token",
        deepgram_api_key="deepgram-key",
        openrouter_api_key="openrouter-key",
        refine=True,
        runtime_state_path=tmp_path / "runtime.json",
    )
    context = SimpleNamespace(
        args=[argument],
        bot_data={
            "settings": settings,
            "transcription_model": "deepgram",
            "transcription_refinement_model": "google/gemini-3.5-flash",
            "runtime_preferences_store": bot_module.create_runtime_preferences_store(settings),
        },
    )
    fake_transcriber = SimpleNamespace(provider_name="deepgram", model="nova-3")
    calls: list[tuple[Settings, str, str | None, str | None]] = []

    def fake_create_transcriber(
        settings_arg: Settings,
        model_key: str,
        translation_enabled: bool | None = None,
        transcription_refinement_enabled: bool = False,
        audio_correction_enabled: bool = False,
        translation_model: str | None = None,
        translation_prompt: str = "normal",
        transcription_refinement_model: str | None = None,
    ) -> object:
        calls.append(
            (settings_arg, model_key, translation_model, transcription_refinement_model)
        )
        return fake_transcriber

    monkeypatch.setattr(bot_module, "create_transcriber", fake_create_transcriber)

    await handle_translation_model_command(update, context)

    assert calls == [
        (settings, "deepgram", expected_model, "google/gemini-3.5-flash")
    ]
    assert context.bot_data["transcriber"] is fake_transcriber
    assert context.bot_data["translation_model"] == expected_model
    assert context.bot_data["runtime_preferences_store"].load().translation_model == expected_model
    assert message.text_replies == [f"Translation model set to {expected_label}."]


@pytest.mark.asyncio
async def test_handle_translation_model_command_rejects_unknown_model() -> None:
    message = FakeMessage()
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    context = SimpleNamespace(
        args=["unknown"],
        bot_data={"settings": Settings(telegram_bot_token="token", openrouter_api_key="key")},
    )

    await handle_translation_model_command(update, context)

    assert message.text_replies == ["Unknown translation model. Available models: gemini, lite, qwen."]


@pytest.mark.asyncio
async def test_handle_translation_model_command_rejects_unauthorized_user() -> None:
    message = FakeMessage()
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=999))
    context = SimpleNamespace(
        args=["lite"],
        bot_data={
            "settings": Settings(
                telegram_bot_token="token",
                openrouter_api_key="key",
                allowed_telegram_user_ids=frozenset({123}),
            )
        },
    )

    await handle_translation_model_command(update, context)

    assert message.text_replies == ["Sorry, this bot is not enabled for your Telegram account."]
    assert "translation_model" not in context.bot_data


@pytest.mark.asyncio
async def test_handle_translation_model_command_ignores_channels() -> None:
    message = FakeMessage(chat_type=ChatType.CHANNEL)
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    context = SimpleNamespace(args=["lite"], bot_data={})

    await handle_translation_model_command(update, context)

    assert message.text_replies == []


@pytest.mark.asyncio
async def test_handle_translation_prompt_command_lists_and_switches_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    message = FakeMessage()
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    settings = Settings(
        telegram_bot_token="token",
        deepgram_api_key="deepgram-key",
        openrouter_api_key="openrouter-key",
        refine=True,
        runtime_state_path=tmp_path / "runtime.json",
    )
    store = bot_module.create_runtime_preferences_store(settings)
    context = SimpleNamespace(
        args=[],
        bot_data={
            "settings": settings,
            "translation_prompt": "normal",
            "runtime_preferences_store": store,
        },
    )

    await handle_translation_prompt_command(update, context)
    assert "Current translation prompt: literal" in message.text_replies[-1]

    fake_transcriber = SimpleNamespace(provider_name="gemini", model="gemini")
    calls: list[tuple[str, str, str, str | None]] = []

    def fake_create_transcriber(
        settings_arg: Settings,
        model_key: str,
        translation_enabled: bool | None = None,
        transcription_refinement_enabled: bool = False,
        audio_correction_enabled: bool = False,
        translation_model: str | None = None,
        translation_prompt: str = "normal",
        transcription_refinement_model: str | None = None,
    ) -> object:
        calls.append(
            (
                model_key,
                translation_model or "",
                translation_prompt,
                transcription_refinement_model,
            )
        )
        return fake_transcriber

    monkeypatch.setattr(bot_module, "create_transcriber", fake_create_transcriber)
    context.args = ["v2"]
    await handle_translation_prompt_command(update, context)

    assert calls == [
        ("deepgram", "google/gemini-2.5-flash-lite", "natural", "google/gemini-2.5-flash-lite")
    ]
    assert context.bot_data["translation_prompt"] == "natural"
    assert context.bot_data["transcriber"] is fake_transcriber
    assert store.load().translation_prompt == "natural"
    assert message.text_replies[-1] == "Translation prompt set to natural."


@pytest.mark.asyncio
async def test_handle_translation_prompt_command_rejects_invalid_and_unauthorized(tmp_path: Path) -> None:
    message = FakeMessage()
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=999))
    settings = Settings(
        telegram_bot_token="token",
        openrouter_api_key="key",
        allowed_telegram_user_ids=frozenset({123}),
        runtime_state_path=tmp_path / "runtime.json",
    )
    context = SimpleNamespace(
        args=["v2"],
        bot_data={
            "settings": settings,
            "runtime_preferences_store": bot_module.create_runtime_preferences_store(settings),
        },
    )

    await handle_translation_prompt_command(update, context)
    assert message.text_replies == ["Sorry, this bot is not enabled for your Telegram account."]
    assert not settings.runtime_state_path.exists()

    update.effective_user.id = 123
    context.args = ["unknown"]
    await handle_translation_prompt_command(update, context)
    assert message.text_replies[-1] == "Unknown translation prompt. Available prompts: natural, literal."
    assert not settings.runtime_state_path.exists()


@pytest.mark.asyncio
async def test_handle_video_upload_rejects_unauthorized_user() -> None:
    message = FakeMessage(video=SimpleNamespace(file_size=1), chat_type=ChatType.GROUP, message_id=123)
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=999))
    context = SimpleNamespace(
        bot_data={
            "settings": Settings(
                telegram_bot_token="token",
                openrouter_api_key="key",
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
    context = SimpleNamespace(bot_data={"settings": Settings(telegram_bot_token="token", openrouter_api_key="key")})

    await handle_video_upload(update, context)

    assert message.text_replies == []


@pytest.mark.asyncio
async def test_handle_video_upload_ignores_group_non_video_document() -> None:
    message = FakeMessage(
        document=SimpleNamespace(mime_type="text/plain", file_name="notes.txt"),
        chat_type=ChatType.GROUP,
    )
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    context = SimpleNamespace(bot_data={"settings": Settings(telegram_bot_token="token", openrouter_api_key="key")})

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
                openrouter_api_key="key",
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
        openrouter_api_key="key",
        deepgram_api_key="key",
        max_video_mb=10 / 1024 / 1024,
    )
    attachment = SimpleNamespace(file_size=10)
    message = FakeMessage(video=attachment, chat_type=ChatType.SUPERGROUP, message_id=123, message_thread_id=8)
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    context = SimpleNamespace(
        bot_data={
            "settings": settings,
            "media_job_queue": asyncio.Queue(),
            "audio_tempo": 1.4,
        }
    )

    await handle_video_upload(update, context)

    queued_job = context.bot_data["media_job_queue"].get_nowait()
    assert queued_job.message is message
    assert queued_job.attachment is attachment
    assert queued_job.audio_tempo == 1.4
    assert message.text_replies == ["Video queued for transcription..."]


@pytest.mark.asyncio
async def test_handle_video_upload_queues_job_when_acknowledgement_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TimeoutMessage(FakeMessage):
        reply_attempts = 0

        async def reply_text(self, text: str, **kwargs: object) -> object:
            del text, kwargs
            self.reply_attempts += 1
            raise TimedOut

    async def skip_sleep(delay: float) -> None:
        assert delay in bot_module.TELEGRAM_RETRY_DELAYS_SECONDS

    monkeypatch.setattr(bot_module.asyncio, "sleep", skip_sleep)
    attachment = SimpleNamespace(file_size=10)
    message = TimeoutMessage(video=attachment)
    update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=123))
    queue: asyncio.Queue[bot_module.QueuedMediaJob] = asyncio.Queue()
    context = SimpleNamespace(
        bot_data={
            "settings": Settings(telegram_bot_token="token", openrouter_api_key="key", deepgram_api_key="key"),
            "media_job_queue": queue,
        }
    )

    await handle_video_upload(update, context)

    queued_job = queue.get_nowait()
    assert queued_job.attachment is attachment
    assert queued_job.status_ref["message"] is None
    assert message.reply_attempts == bot_module.TELEGRAM_REQUEST_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_process_video_message_reports_step_by_step_flow(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def run_inline(func: object, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)

    class FakeAttachment:
        file_name = "clip.mp4"
        file_size = 5

    class FakeTranscriber:
        transcription_refiner = SimpleNamespace(model="arabic-model")
        async def transcribe_chunks_async(self, chunks: object, progress_callback: object = None) -> TranscriptionResult:
            chunk = list(chunks)[0]
            assert progress_callback is not None
            await progress_callback(
                "transcribing_chunk",
                {
                    "index": 1,
                    "total": 1,
                    "chunk_bytes": chunk.path.stat().st_size,
                    "model": "nova-3",
                },
            )
            await progress_callback(
                "provider_fallback",
                {
                    "index": 1,
                    "total": 1,
                    "failed_provider": "gemini",
                    "failed_model": "gemini-model",
                    "next_provider": "deepgram",
                    "next_model": "nova-3",
                },
            )
            await progress_callback("chunk_transcribed", {"index": 1, "total": 1, "raw_chars": 12})
            await progress_callback(
                "refining_transcription",
                {
                    "raw_chars": 12,
                    "raw_bytes": 24,
                    "model": "arabic-model",
                },
            )
            await progress_callback(
                "transcription_refinement_complete",
                {
                    "refined_srt_chars": 53,
                    "refined_transcript_chars": 10,
                    "model": "arabic-model",
                },
            )
            await progress_callback(
                "translating_subtitles",
                {
                    "index": 1,
                    "total": 1,
                    "raw_chars": 12,
                    "raw_bytes": 24,
                    "model": "gpt-5.4",
                },
            )
            await progress_callback(
                "translation_complete",
                {
                    "translated_srt_chars": 95,
                    "line_translated_transcript_chars": 17,
                },
            )
            return TranscriptionResult(
                raw_transcript="هاي خام",
                refined_transcript="هاي منقحة",
                cleaned_transcript="هاي منقحة",
                persian_transcript="این خام است",
                subtitle_cues=(SubtitleCue(0.0, 1.25, "هاي منقحة"),),
                translated_srt=(
                    "1\n"
                    "00:00:00,000 --> 00:00:01,250\n"
                    "هاي منقحة\n"
                    '<font color="green">این خام است</font>\n'
                ),
                line_translated_transcript="هاي منقحة\nاین خام است\n",
                translation_warnings=(PERSIAN_TRANSLATION_NUMBER_WARNING,),
            )

    def fake_extract_audio(video_path: Path, audio_path: Path, *, audio_tempo: float) -> Path:
        assert video_path.exists()
        assert audio_tempo == 1.4
        audio_path.write_bytes(b"audio")
        return audio_path

    def fake_split_audio_to_timed_chunks(
        audio_path: Path, chunks_dir: Path, *, time_scale: float = 1.0
    ) -> list[AudioChunk]:
        assert audio_path.name == "audio.flac"
        assert chunks_dir.name == "chunks"
        return [AudioChunk(path=audio_path)]

    monkeypatch.setattr(bot_module, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(bot_module, "split_audio_to_timed_chunks", fake_split_audio_to_timed_chunks)
    caplog.set_level("INFO", logger="telegram_transcript.bot")
    message = FakeMessage()
    status_ref: dict[str, object] = {"message": None}
    settings = Settings(telegram_bot_token="token", openrouter_api_key="key", refine=True)
    downloader = FakeMediaDownloader()
    context = SimpleNamespace(bot_data={"transcriber": FakeTranscriber(), "media_downloader": downloader})

    await process_video_message(message, FakeAttachment(), settings, context, status_ref, "job1234", 1.4)

    assert downloader.downloads and downloader.downloads[0][:2] == (100, 42)
    assert message.text_replies == [
        "Video received. Starting transcription...",
        "هاي منقحة",
        "این خام است",
        "Warning: " + PERSIAN_TRANSLATION_NUMBER_WARNING,
    ]
    assert "هاي خام" not in message.text_replies
    assert [caption for _, caption in message.document_replies] == ["SRT subtitles"]
    assert [document.filename for document, _ in message.document_replies] == ["clip.srt"]
    status = message.status_replies[0]
    assert status.edits == [
        "Step 1/7: downloading media...",
        "Step 2/7: extracting lossless FLAC audio at 1.4x...",
        "Step 3/7: preparing audio chunks...",
        "Step 4/7: transcribing chunk 1/1...",
        "Gemini could not produce valid SRT; retrying chunk 1/1 with Deepgram...",
        "Step 5/7: refining Iraqi Arabic transcription...",
        "Step 6/7: translating subtitles...",
        "Step 7/7: sending transcript...",
        "Transcript ready.",
    ]
    assert "job1234 step 1/7" in caplog.text
    assert "job1234 step 7/7" in caplog.text
    assert "این خام است" not in caplog.text


@pytest.mark.asyncio
async def test_process_video_message_throttles_rapid_translation_cue_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_inline(func: object, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)

    class FakeAttachment:
        file_name = "clip.mp4"
        file_size = 5

    class FakeTranscriber:
        async def transcribe_chunks_async(self, chunks: object, progress_callback: object = None) -> TranscriptionResult:
            assert progress_callback is not None
            await progress_callback(
                "translating_subtitles",
                {"index": 1, "total": 2, "raw_chars": 60, "raw_bytes": 60, "model": "gpt-5.4"},
            )
            await progress_callback(
                "translating_subtitles",
                {"index": 2, "total": 2, "raw_chars": 61, "raw_bytes": 61, "model": "gpt-5.4"},
            )
            await progress_callback(
                "translation_complete",
                {"translated_srt_chars": 95, "line_translated_transcript_chars": 17},
            )
            return TranscriptionResult(
                raw_transcript="هاي خام",
                subtitle_cues=(SubtitleCue(0.0, 1.25, "هاي خام"),),
                translated_srt=(
                    "1\n"
                    "00:00:00,000 --> 00:00:01,250\n"
                    "هاي خام\n"
                    '<font color="green">این خام است</font>\n'
                ),
                line_translated_transcript="هاي خام\nاین خام است\n",
            )

    def fake_extract_audio(video_path: Path, audio_path: Path, *, audio_tempo: float) -> Path:
        audio_path.write_bytes(b"audio")
        return audio_path

    def fake_split_audio_to_timed_chunks(
        audio_path: Path, chunks_dir: Path, *, time_scale: float = 1.0
    ) -> list[AudioChunk]:
        return [AudioChunk(path=audio_path)]

    monkeypatch.setattr(bot_module, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(bot_module, "split_audio_to_timed_chunks", fake_split_audio_to_timed_chunks)
    message = FakeMessage()
    status_ref: dict[str, object] = {"message": None}
    settings = Settings(telegram_bot_token="token", openrouter_api_key="key", refine=True)
    context = SimpleNamespace(
        bot_data={"transcriber": FakeTranscriber(), "media_downloader": FakeMediaDownloader()}
    )

    await process_video_message(message, FakeAttachment(), settings, context, status_ref, "job1234", 1.0)

    assert "Step 5/6: translating subtitle cue 1/2..." in message.status_replies[0].edits
    assert "Step 5/6: translating subtitle cue 2/2..." not in message.status_replies[0].edits


@pytest.mark.asyncio
async def test_process_video_message_reports_translation_progress_after_throttle_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_inline(func: object, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)
    current_time = [0.0]
    monkeypatch.setattr(bot_module.time, "monotonic", lambda: current_time[0])

    class FakeAttachment:
        file_name = "clip.mp4"
        file_size = 5

    class FakeTranscriber:
        async def transcribe_chunks_async(self, chunks: object, progress_callback: object = None) -> TranscriptionResult:
            assert progress_callback is not None
            current_time[0] = 0.0
            await progress_callback(
                "translating_subtitles",
                {"index": 1, "total": 3, "raw_chars": 60, "raw_bytes": 60, "model": "gpt-5.4"},
            )
            current_time[0] = bot_module.STATUS_PROGRESS_EDIT_INTERVAL_SECONDS - 1
            await progress_callback(
                "translating_subtitles",
                {"index": 2, "total": 3, "raw_chars": 61, "raw_bytes": 61, "model": "gpt-5.4"},
            )
            current_time[0] = bot_module.STATUS_PROGRESS_EDIT_INTERVAL_SECONDS
            await progress_callback(
                "translating_subtitles",
                {"index": 3, "total": 3, "raw_chars": 62, "raw_bytes": 62, "model": "gpt-5.4"},
            )
            return TranscriptionResult(raw_transcript="raw only")

    def fake_extract_audio(video_path: Path, audio_path: Path, *, audio_tempo: float) -> Path:
        audio_path.write_bytes(b"audio")
        return audio_path

    def fake_split_audio_to_timed_chunks(
        audio_path: Path, chunks_dir: Path, *, time_scale: float = 1.0
    ) -> list[AudioChunk]:
        return [AudioChunk(path=audio_path)]

    monkeypatch.setattr(bot_module, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(bot_module, "split_audio_to_timed_chunks", fake_split_audio_to_timed_chunks)
    message = FakeMessage()
    status_ref: dict[str, object] = {"message": None}
    settings = Settings(telegram_bot_token="token", openrouter_api_key="key", refine=True)
    context = SimpleNamespace(
        bot_data={"transcriber": FakeTranscriber(), "media_downloader": FakeMediaDownloader()}
    )

    await process_video_message(message, FakeAttachment(), settings, context, status_ref, "job1234", 1.0)

    edits = message.status_replies[0].edits
    assert "Step 5/6: translating subtitle cue 1/3..." in edits
    assert "Step 5/6: translating subtitle cue 2/3..." not in edits
    assert "Step 5/6: translating subtitle cue 3/3..." in edits


@pytest.mark.asyncio
async def test_process_video_message_skips_srt_when_timestamps_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_inline(func: object, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)

    class FakeAttachment:
        file_name = "clip.mp4"
        file_size = 5

    class FakeTranscriber:
        async def transcribe_chunks_async(self, chunks: object, progress_callback: object = None) -> TranscriptionResult:
            return TranscriptionResult(raw_transcript="raw only")

    def fake_extract_audio(video_path: Path, audio_path: Path, *, audio_tempo: float) -> Path:
        audio_path.write_bytes(b"audio")
        return audio_path

    def fake_split_audio_to_timed_chunks(
        audio_path: Path, chunks_dir: Path, *, time_scale: float = 1.0
    ) -> list[AudioChunk]:
        return [AudioChunk(path=audio_path)]

    monkeypatch.setattr(bot_module, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(bot_module, "split_audio_to_timed_chunks", fake_split_audio_to_timed_chunks)
    message = FakeMessage()
    status_ref: dict[str, object] = {"message": None}
    settings = Settings(telegram_bot_token="token", openrouter_api_key="key")
    context = SimpleNamespace(
        bot_data={"transcriber": FakeTranscriber(), "media_downloader": FakeMediaDownloader()}
    )

    await process_video_message(message, FakeAttachment(), settings, context, status_ref, "job1234", 1.0)

    assert message.text_replies == ["Video received. Starting transcription...", "raw only"]
    assert message.document_replies == []
    assert message.status_replies[0].edits[-1] == "Transcript ready. SRT unavailable for this provider/model."


@pytest.mark.asyncio
async def test_process_video_message_continues_when_status_edit_hits_retry_after(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def run_inline(func: object, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)

    class FakeAttachment:
        file_name = "clip.mp4"
        file_size = 5

    class FakeTranscriber:
        async def transcribe_chunks_async(self, chunks: object, progress_callback: object = None) -> TranscriptionResult:
            return TranscriptionResult(raw_transcript="raw only")

    def fake_extract_audio(video_path: Path, audio_path: Path, *, audio_tempo: float) -> Path:
        audio_path.write_bytes(b"audio")
        return audio_path

    def fake_split_audio_to_timed_chunks(
        audio_path: Path, chunks_dir: Path, *, time_scale: float = 1.0
    ) -> list[AudioChunk]:
        return [AudioChunk(path=audio_path)]

    monkeypatch.setattr(bot_module, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(bot_module, "split_audio_to_timed_chunks", fake_split_audio_to_timed_chunks)
    caplog.set_level("WARNING", logger="telegram_transcript.bot")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        retry_after = RetryAfter(timedelta(seconds=29))
    message = FakeMessage(status=FakeStatus(edit_errors=[retry_after]))
    status_ref: dict[str, object] = {"message": None}
    settings = Settings(telegram_bot_token="token", openrouter_api_key="key")
    context = SimpleNamespace(
        bot_data={"transcriber": FakeTranscriber(), "media_downloader": FakeMediaDownloader()}
    )

    await process_video_message(message, FakeAttachment(), settings, context, status_ref, "job1234", 1.0)

    assert message.text_replies == ["Video received. Starting transcription...", "raw only"]
    assert message.status_replies[0].edits[-1] == "Transcript ready. SRT unavailable for this provider/model."
    assert "skipped Telegram status edit after flood control" in caplog.text
    assert "retry_after=29" in caplog.text


@pytest.mark.asyncio
async def test_process_media_message_accepts_voice_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_inline(func: object, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)

    class FakeTranscriber:
        async def transcribe_chunks_async(self, chunks: object, progress_callback: object = None) -> TranscriptionResult:
            return TranscriptionResult(raw_transcript="voice transcript")

    def fake_extract_audio(source_path: Path, audio_path: Path, *, audio_tempo: float) -> Path:
        assert source_path.name == "source.ogg"
        audio_path.write_bytes(b"audio")
        return audio_path

    def fake_split_audio_to_timed_chunks(
        audio_path: Path, chunks_dir: Path, *, time_scale: float = 1.0
    ) -> list[AudioChunk]:
        return [AudioChunk(path=audio_path)]

    monkeypatch.setattr(bot_module, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(bot_module, "split_audio_to_timed_chunks", fake_split_audio_to_timed_chunks)
    voice = SimpleNamespace(file_size=5, mime_type="audio/ogg")
    message = FakeMessage(voice=voice)
    status_ref: dict[str, object] = {"message": None}
    settings = Settings(telegram_bot_token="token", openrouter_api_key="key")
    context = SimpleNamespace(
        bot_data={"transcriber": FakeTranscriber(), "media_downloader": FakeMediaDownloader(b"voice")}
    )

    await process_media_message(message, voice, settings, context, status_ref, "job1234", 1.0)

    assert message.text_replies == ["Voice note received. Starting transcription...", "voice transcript"]


@pytest.mark.asyncio
async def test_process_video_message_rejects_downloaded_file_over_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAttachment:
        file_name = "clip.mp4"
        file_size = None

    def fail_extract_audio(*_: object, **__: object) -> None:
        raise AssertionError("extract_audio should not run for an oversized downloaded video")

    monkeypatch.setattr(bot_module, "extract_audio", fail_extract_audio)
    message = FakeMessage()
    status_ref: dict[str, object] = {"message": None}
    settings = Settings(
        telegram_bot_token="token",
        openrouter_api_key="key",
        max_video_mb=10 / 1024 / 1024,
    )
    context = SimpleNamespace(
        bot_data={"transcriber": object(), "media_downloader": FakeMediaDownloader(b"x" * 11)}
    )

    await process_video_message(message, FakeAttachment(), settings, context, status_ref, "job1234", 1.0)

    assert status_ref["message"] is not None
    downloaded_path = context.bot_data["media_downloader"].downloads[0][2]
    assert not downloaded_path.exists()
    assert message.text_replies == ["Video received. Starting transcription..."]
    assert message.status_replies[0].edits == [
        "Step 1/5: downloading media...",
        "Media is larger than the configured upload limit.",
    ]


def install_privacy_pipeline_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_inline(func: object, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    def fake_extract_audio(video_path: Path, audio_path: Path, *, audio_tempo: float) -> Path:
        audio_path.write_bytes(b"audio")
        return audio_path

    def fake_split_audio_to_timed_chunks(
        audio_path: Path, chunks_dir: Path, *, time_scale: float = 1.0
    ) -> list[AudioChunk]:
        return [AudioChunk(path=audio_path)]

    monkeypatch.setattr(asyncio, "to_thread", run_inline)
    monkeypatch.setattr(bot_module, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(bot_module, "split_audio_to_timed_chunks", fake_split_audio_to_timed_chunks)


class PrivacyFakeTranscriber:
    async def transcribe_chunks_async(self, chunks: object, progress_callback: object = None) -> TranscriptionResult:
        return TranscriptionResult(raw_transcript="hello transcript")


@pytest.mark.asyncio
async def test_process_media_message_does_not_read_or_write_legacy_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_privacy_pipeline_fakes(monkeypatch)

    class BrokenRegistry:
        def find_duplicate(self, file_name: str | None, file_hash: str) -> None:
            raise AssertionError("legacy registry must not be read")

        def record_video(self, **kwargs: object) -> int:
            raise AssertionError("legacy registry must not be written")

    attachment = SimpleNamespace(file_name="clip.mp4", file_size=5, mime_type="video/mp4")
    message = FakeMessage(from_user=SimpleNamespace(id=1, username="alice", full_name="Alice A"))
    context = SimpleNamespace(
        bot_data={
            "transcriber": PrivacyFakeTranscriber(),
            "media_downloader": FakeMediaDownloader(b"first-video"),
            "video_registry": BrokenRegistry(),
        }
    )
    await process_media_message(
        message,
        attachment,
        Settings(telegram_bot_token="token", openrouter_api_key="key"),
        context,
        {"message": None},
        "job1234",
        1.0,
    )

    assert "hello transcript" in message.text_replies
    downloaded_path = context.bot_data["media_downloader"].downloads[0][2]
    assert not downloaded_path.exists()
