from __future__ import annotations

import asyncio
import logging
import shutil
import sqlite3
import tempfile
import time
import uuid
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path

from telegram import Audio, Document, InputFile, Message, Update, Video, Voice
from telegram.constants import ChatType
from telegram.error import RetryAfter, TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from telegram_transcript.config import ConfigError, Settings, load_settings, parse_audio_tempo
from telegram_transcript.ffmpeg import FfmpegError, ensure_ffmpeg_available, extract_audio, split_audio_to_timed_chunks
from telegram_transcript.models import TranscriptionResult
from telegram_transcript.runtime_state import RuntimePreferences, RuntimePreferencesStore, RuntimeStateError
from telegram_transcript.scoped_state import ScopedStateStore
from telegram_transcript.telegram_utils import (
    format_transcript_for_delivery,
    should_send_as_text,
    split_text_for_telegram,
)
from telegram_transcript.telegram_downloader import TelegramDownloadError, TelegramMediaDownloader
from telegram_transcript.transcriber import (
    CLAUDE_SONNET_TRANSLATION_MODEL,
    DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL,
    DEFAULT_GEMINI_TRANSCRIPTION_MODEL,
    DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
    DEFAULT_REFINEMENT_MODEL,
    DEFAULT_TRANSCRIPTION_REFINEMENT_MODEL,
    DEFAULT_TRANSLATION_PROMPT_KEY,
    DEFAULT_WHISPER_LARGE_V3_TRANSCRIPTION_MODEL,
    TRANSLATION_PROMPT_ALIASES,
    TRANSLATION_PROMPT_OPTIONS,
    DeepgramSpeechToTextProvider,
    GeminiAudioCorrectionProvider,
    GeminiSpeechToTextProvider,
    IraqiArabicTranscriptRefiner,
    LowConfidenceTranscriptCorrector,
    OpenAISpeechToTextProvider,
    OpenRouterWhisperSpeechToTextProvider,
    SpeechTranscriber,
    TranscriptCandidateResolver,
    TranscriptRefiner,
    TranscriptionError,
    canonicalize_translation_prompt_key,
    render_srt,
)

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".oga", ".ogg", ".opus", ".wav"}
AUDIO_SUFFIX_BY_MIME = {
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "audio/m4a": ".m4a",
    "audio/mp4": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
}
GROUP_CHAT_TYPES = {ChatType.GROUP, ChatType.SUPERGROUP}
DEFAULT_OUTPUT_BASENAME = "transcript"
DEFAULT_TRANSCRIPTION_MODEL_KEY = "gemini"
STATUS_PROGRESS_EDIT_INTERVAL_SECONDS = 30.0
HELP_MESSAGE_DELETE_DELAY_SECONDS = 20.0
MINIMUM_TEMPORARY_FREE_BYTES = 512 * 1024 * 1024
TEMPORARY_SPACE_MULTIPLIER = 3
HELP_MESSAGE = """Available commands:
/help - Show this help message (deleted after 20 seconds).
/start - Start the bot (currently no additional setup is required).
/tempo <0.5-2.0> - Set audio tempo for future media (groups only).
/model [gemini|deepgram|whisper|openai] - Show or select the primary transcription model.
/refiner [gpt|gemini] - Show or select the Iraqi transcription refinement model.
/translate - Toggle Persian translation for future media.
/tmodel [gemini|gpt|claude] - Show or select the translation model.
/translation [natural|literal] - Show or select the translation style.
/privacy - Show data handling and retention information.
/forget - Delete your scope's retained preferences and job metadata.

Send a video, audio file, or voice note to create a transcript."""


@dataclass(frozen=True)
class TranscriptionModelOption:
    key: str
    provider: str
    model: str
    label: str


@dataclass(frozen=True)
class TranslationModelOption:
    key: str
    model: str
    label: str


@dataclass(frozen=True)
class TranscriptionRefinementModelOption:
    key: str
    model: str
    label: str


@dataclass(frozen=True)
class QueuedMediaJob:
    message: Message
    attachment: Video | Audio | Voice | Document
    settings: Settings
    context: ContextTypes.DEFAULT_TYPE
    status_ref: dict[str, Message | None]
    job_id: str
    audio_tempo: float
    transcriber: SpeechTranscriber
    translation_enabled: bool


TRANSCRIPTION_MODEL_OPTIONS: dict[str, TranscriptionModelOption] = {
    "gemini": TranscriptionModelOption(
        key="gemini",
        provider="gemini",
        model=DEFAULT_GEMINI_TRANSCRIPTION_MODEL,
        label="OpenRouter Gemini 3.5 Flash",
    ),
    "deepgram": TranscriptionModelOption(
        key="deepgram",
        provider="deepgram",
        model=DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL,
        label=f"Deepgram {DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL}",
    ),
    "whisper": TranscriptionModelOption(
        key="whisper",
        provider="whisper",
        model=DEFAULT_WHISPER_LARGE_V3_TRANSCRIPTION_MODEL,
        label="OpenAI: Whisper Large V3",
    ),
    "openai": TranscriptionModelOption(
        key="openai",
        provider="openai",
        model=DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
        label="Direct OpenAI whisper-1",
    ),
}
TRANSCRIPTION_FALLBACK_ORDER = ("gemini", "deepgram", "whisper", "openai")
TRANSCRIPTION_MODEL_ALIASES = {
    option.key: option.key
    for option in TRANSCRIPTION_MODEL_OPTIONS.values()
} | {
    option.model: option.key
    for option in TRANSCRIPTION_MODEL_OPTIONS.values()
}

TRANSLATION_MODEL_OPTIONS: dict[str, TranslationModelOption] = {
    "gemini": TranslationModelOption(
        key="gemini",
        model=DEFAULT_GEMINI_TRANSCRIPTION_MODEL,
        label="OpenRouter Gemini 3.5 Flash",
    ),
    "gpt": TranslationModelOption(
        key="gpt",
        model=DEFAULT_REFINEMENT_MODEL,
        label="OpenRouter GPT-5.5",
    ),
    "claude": TranslationModelOption(
        key="claude",
        model=CLAUDE_SONNET_TRANSLATION_MODEL,
        label="OpenRouter Claude Sonnet 4.6",
    ),
}
TRANSLATION_MODEL_ALIASES = {
    option.key: option.key
    for option in TRANSLATION_MODEL_OPTIONS.values()
} | {
    option.model: option.key
    for option in TRANSLATION_MODEL_OPTIONS.values()
}

TRANSCRIPTION_REFINEMENT_MODEL_OPTIONS: dict[str, TranscriptionRefinementModelOption] = {
    "gpt": TranscriptionRefinementModelOption(
        key="gpt",
        model=DEFAULT_TRANSCRIPTION_REFINEMENT_MODEL,
        label="OpenRouter GPT-5.5",
    ),
    "gemini": TranscriptionRefinementModelOption(
        key="gemini",
        model=DEFAULT_GEMINI_TRANSCRIPTION_MODEL,
        label="OpenRouter Gemini 3.5 Flash",
    ),
}
TRANSCRIPTION_REFINEMENT_MODEL_ALIASES = {
    option.key: option.model
    for option in TRANSCRIPTION_REFINEMENT_MODEL_OPTIONS.values()
} | {
    option.model: option.model
    for option in TRANSCRIPTION_REFINEMENT_MODEL_OPTIONS.values()
}


def create_runtime_preferences_store(settings: Settings) -> RuntimePreferencesStore:
    defaults = RuntimePreferences(
        audio_tempo=settings.audio_tempo,
        transcription_model=DEFAULT_TRANSCRIPTION_MODEL_KEY,
        transcription_refinement_model=settings.openrouter_transcription_refinement_model,
        translation_enabled=settings.refine,
        translation_model=settings.openrouter_refine_model,
        translation_prompt=DEFAULT_TRANSLATION_PROMPT_KEY,
    )
    return RuntimePreferencesStore(
        settings.runtime_state_path,
        defaults=defaults,
        transcription_models=frozenset(TRANSCRIPTION_MODEL_OPTIONS),
        transcription_refinement_models=frozenset(
            {
                settings.openrouter_transcription_refinement_model,
                *(option.model for option in TRANSCRIPTION_REFINEMENT_MODEL_OPTIONS.values()),
            }
        ),
        translation_models=frozenset(
            {settings.openrouter_refine_model, *(option.model for option in TRANSLATION_MODEL_OPTIONS.values())}
        ),
        translation_prompts=frozenset(TRANSLATION_PROMPT_ALIASES),
    )


def create_application(settings: Settings | None = None) -> Application:
    settings = settings or load_settings()
    runtime_store = create_runtime_preferences_store(settings)
    runtime_preferences = runtime_store.load()
    translation_prompt = canonicalize_translation_prompt_key(runtime_preferences.translation_prompt)
    selected_refinement_model = runtime_preferences.transcription_refinement_model
    transcriber = create_transcriber(
        settings,
        runtime_preferences.transcription_model,
        translation_enabled=runtime_preferences.translation_enabled,
        transcription_refinement_model=selected_refinement_model,
        translation_model=runtime_preferences.translation_model,
        translation_prompt=translation_prompt,
    )

    app = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .post_init(start_media_downloader)
        .post_shutdown(stop_media_downloader)
        .build()
    )
    app.bot_data["settings"] = settings
    app.bot_data["transcriber"] = transcriber
    app.bot_data["transcription_model"] = runtime_preferences.transcription_model
    app.bot_data["transcription_refinement_model"] = selected_refinement_model
    app.bot_data["translation_enabled"] = runtime_preferences.translation_enabled
    app.bot_data["translation_model"] = runtime_preferences.translation_model
    app.bot_data["translation_prompt"] = translation_prompt
    app.bot_data["runtime_preferences_store"] = runtime_store
    app.bot_data["media_job_queue"] = asyncio.Queue(
        maxsize=max(10, settings.max_concurrent_jobs * 10)
    )
    app.bot_data["audio_tempo"] = runtime_preferences.audio_tempo
    app.bot_data["scoped_state_store"] = ScopedStateStore(
        settings.scoped_state_path,
        defaults=runtime_preferences,
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("tempo", handle_tempo_command))
    app.add_handler(CommandHandler("model", handle_model_command))
    app.add_handler(CommandHandler("refiner", handle_transcription_refinement_model_command))
    app.add_handler(CommandHandler("translate", handle_translation_toggle_command))
    app.add_handler(CommandHandler("tmodel", handle_translation_model_command))
    app.add_handler(CommandHandler("translation", handle_translation_prompt_command))
    app.add_handler(CommandHandler("privacy", handle_privacy_command))
    app.add_handler(CommandHandler("forget", handle_forget_command))
    app.add_handler(MessageHandler(media_message_filter(), handle_media_upload, block=False))
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_non_media))
    return app


def create_transcriber(
    settings: Settings,
    model_key: str = DEFAULT_TRANSCRIPTION_MODEL_KEY,
    translation_model: str | None = None,
    translation_prompt: str = DEFAULT_TRANSLATION_PROMPT_KEY,
    transcription_refinement_model: str | None = None,
    translation_enabled: bool | None = None,
) -> SpeechTranscriber:
    if not settings.openrouter_api_key:
        raise ConfigError(
            "OPENROUTER_API_KEY is required for Iraqi Arabic cleaning and Persian translation."
        )
    model_order = get_transcription_model_order(model_key)
    configured_model_order = tuple(
        ordered_model_key
        for ordered_model_key in model_order
        if is_model_option_configured(TRANSCRIPTION_MODEL_OPTIONS[ordered_model_key], settings)
    )
    if not configured_model_order:
        raise ConfigError("At least one transcription provider must be configured.")
    speech_to_text_providers = [
        create_speech_to_text_provider(settings, ordered_model_key)
        for ordered_model_key in configured_model_order
    ]
    canonical_prompt = canonicalize_translation_prompt_key(translation_prompt)
    effective_translation_enabled = settings.refine if translation_enabled is None else translation_enabled

    refiner = (
        TranscriptRefiner(
            api_key=settings.openrouter_api_key,
            model=translation_model or settings.openrouter_refine_model,
            system_prompt=TRANSLATION_PROMPT_OPTIONS[canonical_prompt],
            prompt_key=canonical_prompt,
        )
        if effective_translation_enabled
        else None
    )
    correctors_by_provider = {
        "deepgram": LowConfidenceTranscriptCorrector(
            secondary_provider=GeminiAudioCorrectionProvider(api_key=settings.openrouter_api_key),
            resolver=TranscriptCandidateResolver(
                api_key=settings.openrouter_api_key,
                model=translation_model or settings.openrouter_refine_model,
            ),
        )
    }
    selected_refinement_model = (
        transcription_refinement_model or settings.openrouter_transcription_refinement_model
    )
    return SpeechTranscriber(
        speech_to_text_provider=speech_to_text_providers[0],
        fallback_speech_to_text_providers=speech_to_text_providers[1:],
        transcription_refiner=IraqiArabicTranscriptRefiner(
            api_key=settings.openrouter_api_key,
            model=selected_refinement_model,
        ),
        refiner=refiner,
        correctors_by_provider=correctors_by_provider,
    )


def create_runtime_transcriber(
    context: ContextTypes.DEFAULT_TYPE | Application,
    settings: Settings,
    *,
    model_key: str | None = None,
    translation_enabled: bool | None = None,
    translation_model: str | None = None,
    translation_prompt: str | None = None,
    transcription_refinement_model: str | None = None,
) -> SpeechTranscriber:
    selected_refinement_model = (
        transcription_refinement_model
        or get_runtime_transcription_refinement_model(context, settings)
    )
    return create_transcriber(
        settings,
        model_key or get_runtime_model_key(context),
        translation_enabled=(
            get_runtime_translation_enabled(context, settings)
            if translation_enabled is None
            else translation_enabled
        ),
        translation_model=translation_model or get_runtime_translation_model(context, settings),
        translation_prompt=translation_prompt or get_runtime_translation_prompt(context),
        transcription_refinement_model=selected_refinement_model,
    )


def get_transcription_model_order(model_key: str) -> tuple[str, ...]:
    if model_key not in TRANSCRIPTION_MODEL_OPTIONS:
        raise ConfigError(f"Unknown transcription model: {model_key}")
    return (model_key, *(key for key in TRANSCRIPTION_FALLBACK_ORDER if key != model_key))


def create_speech_to_text_provider(settings: Settings, model_key: str) -> object:
    option = TRANSCRIPTION_MODEL_OPTIONS.get(model_key)
    if option is None:
        raise ConfigError(f"Unknown transcription model: {model_key}")

    if option.provider == "deepgram":
        if not settings.deepgram_api_key:
            raise ConfigError("DEEPGRAM_API_KEY is required for Deepgram transcription.")
        return DeepgramSpeechToTextProvider(
            api_key=settings.deepgram_api_key,
            model=settings.deepgram_transcribe_model,
            language=settings.deepgram_language,
            keyterms=settings.deepgram_keyterms,
        )
    if option.provider == "openai":
        if not settings.openai_api_key:
            raise ConfigError("OPENAI_API_KEY is required for direct OpenAI transcription.")
        return OpenAISpeechToTextProvider(api_key=settings.openai_api_key, model=option.model)
    if option.provider == "whisper":
        if not settings.openrouter_api_key:
            raise ConfigError("OPENROUTER_API_KEY is required for OpenRouter Whisper transcription.")
        return OpenRouterWhisperSpeechToTextProvider(
            api_key=settings.openrouter_api_key,
            model=option.model,
        )
    if option.provider == "gemini":
        if not settings.openrouter_api_key:
            raise ConfigError("OPENROUTER_API_KEY is required for OpenRouter transcription.")
        return GeminiSpeechToTextProvider(api_key=settings.openrouter_api_key, model=option.model)

    raise ConfigError(f"Unsupported transcription provider: {option.provider}")


async def start_media_downloader(application: Application) -> None:
    settings: Settings = application.bot_data["settings"]
    downloader = TelegramMediaDownloader(settings)
    await downloader.start()
    application.bot_data["media_downloader"] = downloader
    scoped_store = application.bot_data.get("scoped_state_store")
    if isinstance(scoped_store, ScopedStateStore):
        interrupted = await asyncio.to_thread(scoped_store.fail_incomplete_jobs)
        await asyncio.to_thread(scoped_store.purge_expired_jobs)
        if interrupted:
            logger.warning("marked %d interrupted jobs as failed during startup", interrupted)
    queue = application.bot_data.get("media_job_queue")
    if isinstance(queue, asyncio.Queue):
        application.bot_data["media_job_workers"] = [
            asyncio.create_task(media_job_worker(application, worker_index), name=f"media-job-{worker_index}")
            for worker_index in range(settings.max_concurrent_jobs)
        ]


async def stop_media_downloader(application: Application) -> None:
    workers = application.bot_data.get("media_job_workers", [])
    for worker in workers:
        worker.cancel()
    if workers:
        await asyncio.gather(*workers, return_exceptions=True)
    downloader = application.bot_data.get("media_downloader")
    if isinstance(downloader, TelegramMediaDownloader):
        await downloader.close()
    scoped_store = application.bot_data.get("scoped_state_store")
    if isinstance(scoped_store, ScopedStateStore):
        await asyncio.to_thread(scoped_store.fail_incomplete_jobs)
        scoped_store.close()


async def media_job_worker(application: Application, worker_index: int) -> None:
    queue = application.bot_data["media_job_queue"]
    scoped_store = application.bot_data.get("scoped_state_store")
    while True:
        job: QueuedMediaJob = await queue.get()
        try:
            logger.info("job %s started by worker %d", job.job_id, worker_index)
            if isinstance(scoped_store, ScopedStateStore):
                await asyncio.to_thread(scoped_store.set_job_status, job.job_id, "running")
            await process_media_message(
                job.message,
                job.attachment,
                job.settings,
                job.context,
                job.status_ref,
                job.job_id,
                job.audio_tempo,
                transcriber_override=job.transcriber,
                translation_enabled_override=job.translation_enabled,
            )
            if isinstance(scoped_store, ScopedStateStore):
                await asyncio.to_thread(scoped_store.set_job_status, job.job_id, "completed")
        except asyncio.CancelledError:
            if isinstance(scoped_store, ScopedStateStore):
                await asyncio.to_thread(scoped_store.set_job_status, job.job_id, "failed")
            raise
        except (FfmpegError, TelegramDownloadError, TranscriptionError) as exc:
            logger.exception("job %s media transcription failed", job.job_id)
            status = job.status_ref["message"]
            if status is not None:
                await edit_status_message(status, f"Transcription failed: {exc}", job_id=job.job_id)
            if isinstance(scoped_store, ScopedStateStore):
                await asyncio.to_thread(scoped_store.set_job_status, job.job_id, "failed")
        except Exception:
            logger.exception("job %s unexpected media transcription failure", job.job_id)
            status = job.status_ref["message"]
            if status is not None:
                await edit_status_message(
                    status,
                    "Transcription failed because of an unexpected error.",
                    job_id=job.job_id,
                )
            if isinstance(scoped_store, ScopedStateStore):
                await asyncio.to_thread(scoped_store.set_job_status, job.job_id, "failed")
        finally:
            queue.task_done()


def media_message_filter() -> filters.BaseFilter:
    return filters.VIDEO | filters.AUDIO | filters.VOICE | filters.Document.ALL


def video_message_filter() -> filters.BaseFilter:
    return media_message_filter()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    return


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or get_chat_type(message) == ChatType.CHANNEL:
        return

    help_message = await reply_to_source(message, HELP_MESSAGE)
    context.application.create_task(
        delete_message_after_delay(help_message, HELP_MESSAGE_DELETE_DELAY_SECONDS),
        update=update,
        name="delete-help-message",
    )


async def delete_message_after_delay(message: Message, delay_seconds: float) -> None:
    await asyncio.sleep(delay_seconds)
    try:
        await message.delete()
    except TelegramError as exc:
        logger.warning("Unable to delete help message: %s", exc)


async def handle_non_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    return


async def handle_non_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_non_media(update, context)


async def handle_privacy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or get_chat_type(message) == ChatType.CHANNEL:
        return
    await reply_to_source(
        message,
        "Media is processed temporarily and sent to configured transcription and translation providers. "
        "The bot does not retain media, transcripts, or sender names after delivery. Scoped job metadata "
        "and hashed Telegram file identifiers are retained for seven days for idempotency. Use /forget "
        "to delete your scope metadata sooner.",
    )


async def handle_forget_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or get_chat_type(message) == ChatType.CHANNEL:
        return
    user_id = get_update_user_id(update)
    if not await can_change_scope_preferences(context, message, user_id):
        await reply_to_source(message, "Only chat administrators can delete group metadata.")
        return
    store = context.bot_data.get("scoped_state_store")
    if not isinstance(store, ScopedStateStore):
        await reply_to_source(message, "Scoped metadata storage is unavailable.")
        return
    scope_key = get_preference_scope(message, user_id)
    await asyncio.to_thread(store.delete_scope, scope_key)
    await reply_to_source(message, "Your scoped preferences and retained job metadata were deleted.")


async def handle_tempo_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or get_chat_type(message) not in GROUP_CHAT_TYPES:
        return

    audio_tempo = parse_tempo_command_args(getattr(context, "args", None))
    if audio_tempo is None:
        return
    user_id = get_update_user_id(update)
    if not await can_change_scope_preferences(context, message, user_id):
        await reply_to_source(message, "Only chat administrators can change group transcription settings.")
        return

    try:
        preferences = get_scoped_preferences(context, message, user_id)
        save_scoped_preferences(context, message, user_id, replace(preferences, audio_tempo=audio_tempo))
    except RuntimeStateError as exc:
        await reply_to_source(message, str(exc))
        return
    await reply_to_source(message, f"Tempo set to {audio_tempo:g}x.")


async def handle_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or get_chat_type(message) == ChatType.CHANNEL:
        return

    settings: Settings = context.bot_data["settings"]
    user_id = get_update_user_id(update)
    if not is_authorized(settings, user_id):
        await reply_to_source(message, "Sorry, this bot is not enabled for your Telegram account.")
        return
    preferences = get_scoped_preferences(context, message, user_id)

    args = getattr(context, "args", None)
    if not isinstance(args, list) or not args:
        await reply_to_source(message, format_model_preferences_message(preferences, settings))
        return
    if len(args) != 1:
        await reply_to_source(message, format_unknown_model_message())
        return

    model_key = parse_model_command_arg(args[0])
    if model_key is None:
        await reply_to_source(message, format_unknown_model_message())
        return
    if not await can_change_scope_preferences(context, message, user_id):
        await reply_to_source(message, "Only chat administrators can change group transcription settings.")
        return

    selected_option = TRANSCRIPTION_MODEL_OPTIONS[model_key]
    if not is_model_option_configured(selected_option, settings):
        await reply_to_source(message, missing_model_credential_message(selected_option))
        return

    try:
        replacement_transcriber = create_transcriber(
            settings,
            model_key,
            translation_enabled=preferences.translation_enabled,
            transcription_refinement_model=preferences.transcription_refinement_model,
            translation_model=preferences.translation_model,
            translation_prompt=preferences.translation_prompt,
        )
    except ConfigError as exc:
        await reply_to_source(message, str(exc))
        return

    try:
        save_scoped_preferences(
            context,
            message,
            user_id,
            replace(preferences, transcription_model=model_key),
        )
        activate_legacy_runtime_transcriber(context, replacement_transcriber)
    except RuntimeStateError as exc:
        await reply_to_source(message, str(exc))
        return
    await reply_to_source(message, f"Primary transcription model set to {TRANSCRIPTION_MODEL_OPTIONS[model_key].label}.")


async def handle_transcription_refinement_model_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    message = update.effective_message
    if message is None or get_chat_type(message) == ChatType.CHANNEL:
        return

    settings: Settings = context.bot_data["settings"]
    user_id = get_update_user_id(update)
    if not is_authorized(settings, user_id):
        await reply_to_source(message, "Sorry, this bot is not enabled for your Telegram account.")
        return
    preferences = get_scoped_preferences(context, message, user_id)

    args = getattr(context, "args", None)
    if not isinstance(args, list) or not args:
        await reply_to_source(
            message,
            format_transcription_refinement_preferences_message(preferences),
        )
        return
    if len(args) != 1:
        await reply_to_source(message, format_unknown_transcription_refinement_model_message())
        return

    refinement_model = parse_transcription_refinement_model_command_arg(args[0])
    if refinement_model is None:
        await reply_to_source(message, format_unknown_transcription_refinement_model_message())
        return
    if not await can_change_scope_preferences(context, message, user_id):
        await reply_to_source(message, "Only chat administrators can change group transcription settings.")
        return
    try:
        replacement_transcriber = create_transcriber(
            settings,
            preferences.transcription_model,
            translation_enabled=preferences.translation_enabled,
            transcription_refinement_model=refinement_model,
            translation_model=preferences.translation_model,
            translation_prompt=preferences.translation_prompt,
        )
        save_scoped_preferences(
            context,
            message,
            user_id,
            replace(preferences, transcription_refinement_model=refinement_model),
        )
        activate_legacy_runtime_transcriber(context, replacement_transcriber)
    except (ConfigError, RuntimeStateError) as exc:
        await reply_to_source(message, str(exc))
        return

    option = next(
        option
        for option in TRANSCRIPTION_REFINEMENT_MODEL_OPTIONS.values()
        if option.model == refinement_model
    )
    await reply_to_source(message, f"Transcription refinement model set to {option.label}.")


async def handle_translation_toggle_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or get_chat_type(message) == ChatType.CHANNEL:
        return

    settings: Settings = context.bot_data["settings"]
    user_id = get_update_user_id(update)
    if not is_authorized(settings, user_id):
        await reply_to_source(message, "Sorry, this bot is not enabled for your Telegram account.")
        return
    preferences = get_scoped_preferences(context, message, user_id)

    args = getattr(context, "args", None)
    if not isinstance(args, list) or args:
        await reply_to_source(message, "Usage: /translate")
        return
    if not await can_change_scope_preferences(context, message, user_id):
        await reply_to_source(message, "Only chat administrators can change group transcription settings.")
        return

    translation_enabled = not preferences.translation_enabled
    try:
        replacement_transcriber = create_transcriber(
            settings,
            preferences.transcription_model,
            translation_enabled=translation_enabled,
            transcription_refinement_model=preferences.transcription_refinement_model,
            translation_model=preferences.translation_model,
            translation_prompt=preferences.translation_prompt,
        )
        save_scoped_preferences(
            context,
            message,
            user_id,
            replace(preferences, translation_enabled=translation_enabled),
        )
        activate_legacy_runtime_transcriber(context, replacement_transcriber)
    except (ConfigError, RuntimeStateError) as exc:
        await reply_to_source(message, str(exc))
        return

    state = "enabled" if translation_enabled else "disabled"
    await reply_to_source(message, f"Translation {state}.")


async def handle_translation_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or get_chat_type(message) == ChatType.CHANNEL:
        return

    settings: Settings = context.bot_data["settings"]
    user_id = get_update_user_id(update)
    if not is_authorized(settings, user_id):
        await reply_to_source(message, "Sorry, this bot is not enabled for your Telegram account.")
        return
    preferences = get_scoped_preferences(context, message, user_id)

    args = getattr(context, "args", None)
    if not isinstance(args, list) or not args:
        await reply_to_source(message, format_translation_preferences_message(preferences))
        return
    if len(args) != 1:
        await reply_to_source(message, format_unknown_translation_model_message())
        return

    model_key = parse_translation_model_command_arg(args[0])
    if model_key is None:
        await reply_to_source(message, format_unknown_translation_model_message())
        return
    if not await can_change_scope_preferences(context, message, user_id):
        await reply_to_source(message, "Only chat administrators can change group transcription settings.")
        return

    option = TRANSLATION_MODEL_OPTIONS[model_key]
    try:
        replacement_transcriber = create_transcriber(
            settings,
            preferences.transcription_model,
            translation_enabled=preferences.translation_enabled,
            transcription_refinement_model=preferences.transcription_refinement_model,
            translation_model=option.model,
            translation_prompt=preferences.translation_prompt,
        )
    except ConfigError as exc:
        await reply_to_source(message, str(exc))
        return

    try:
        save_scoped_preferences(
            context,
            message,
            user_id,
            replace(preferences, translation_model=option.model),
        )
        activate_legacy_runtime_transcriber(context, replacement_transcriber)
    except RuntimeStateError as exc:
        await reply_to_source(message, str(exc))
        return
    await reply_to_source(message, f"Translation model set to {option.label}.")


async def handle_translation_prompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or get_chat_type(message) == ChatType.CHANNEL:
        return

    settings: Settings = context.bot_data["settings"]
    user_id = get_update_user_id(update)
    if not is_authorized(settings, user_id):
        await reply_to_source(message, "Sorry, this bot is not enabled for your Telegram account.")
        return
    preferences = get_scoped_preferences(context, message, user_id)

    args = getattr(context, "args", None)
    if not isinstance(args, list) or not args:
        await reply_to_source(message, format_translation_prompt_preferences_message(preferences))
        return
    if len(args) != 1:
        await reply_to_source(message, format_unknown_translation_prompt_message())
        return

    prompt_key = TRANSLATION_PROMPT_ALIASES.get(args[0].strip().lower())
    if prompt_key is None:
        await reply_to_source(message, format_unknown_translation_prompt_message())
        return
    if not await can_change_scope_preferences(context, message, user_id):
        await reply_to_source(message, "Only chat administrators can change group transcription settings.")
        return
    try:
        replacement_transcriber = create_transcriber(
            settings,
            preferences.transcription_model,
            translation_enabled=preferences.translation_enabled,
            transcription_refinement_model=preferences.transcription_refinement_model,
            translation_model=preferences.translation_model,
            translation_prompt=prompt_key,
        )
        save_scoped_preferences(
            context,
            message,
            user_id,
            replace(preferences, translation_prompt=prompt_key),
        )
        activate_legacy_runtime_transcriber(context, replacement_transcriber)
    except (ConfigError, RuntimeStateError) as exc:
        await reply_to_source(message, str(exc))
        return

    await reply_to_source(message, f"Translation prompt set to {prompt_key}.")


async def handle_media_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    if get_chat_type(message) == ChatType.CHANNEL:
        return

    settings: Settings = context.bot_data["settings"]
    user_id = get_update_user_id(update)
    if not is_authorized(settings, user_id):
        await reply_to_source(message, "Sorry, this bot is not enabled for your Telegram account.")
        return

    attachment = get_media_attachment(message)
    if attachment is None:
        return

    file_size = get_attachment_file_size(attachment)
    if file_size is not None and file_size > settings.max_video_bytes:
        return

    preferences = get_scoped_preferences(context, message, user_id)
    audio_tempo = preferences.audio_tempo
    try:
        transcriber = create_transcriber(
            settings,
            preferences.transcription_model,
            translation_enabled=preferences.translation_enabled,
            transcription_refinement_model=preferences.transcription_refinement_model,
            translation_model=preferences.translation_model,
            translation_prompt=preferences.translation_prompt,
        )
    except ConfigError as exc:
        await reply_to_source(message, f"Transcription configuration is unavailable: {exc}")
        return
    provider_name, model_name = transcriber.provider_name, transcriber.model
    transcription_refinement_model = preferences.transcription_refinement_model
    translation_enabled = preferences.translation_enabled
    translation_model = preferences.translation_model
    job_id = uuid.uuid4().hex[:8]
    scoped_store = context.bot_data.get("scoped_state_store")
    if isinstance(scoped_store, ScopedStateStore):
        chat_id = get_message_chat_id(message)
        message_id = getattr(message, "message_id", None)
        if chat_id is not None and isinstance(message_id, int):
            inserted = await asyncio.to_thread(
                scoped_store.record_job,
                job_id=job_id,
                scope_key=get_preference_scope(message, user_id),
                chat_id=chat_id,
                message_id=message_id,
                file_unique_id=getattr(attachment, "file_unique_id", None),
            )
            if not inserted:
                await reply_to_source(message, "This Telegram message is already queued or was processed recently.")
                return
            await asyncio.to_thread(scoped_store.purge_expired_jobs)
    logger.info(
        "job %s queued: suffix=%s telegram_file_size=%s audio_tempo=%g stt_provider=%s transcribe_model=%s transcription_refinement_model=%s translation_enabled=%s translation_model=%s",
        job_id,
        get_media_attachment_suffix(attachment),
        file_size,
        audio_tempo,
        provider_name,
        model_name,
        transcription_refinement_model,
        translation_enabled,
        translation_model,
    )

    status = await reply_to_source(
        message,
        f"{get_media_kind_label(message, attachment)} queued for transcription...",
    )
    status_ref: dict[str, Message | None] = {"message": status}
    queue = context.bot_data.get("media_job_queue")
    if not isinstance(queue, asyncio.Queue):
        raise RuntimeError("Media job queue is not initialized.")
    job = QueuedMediaJob(
        message=message,
        attachment=attachment,
        settings=settings,
        context=context,
        status_ref=status_ref,
        job_id=job_id,
        audio_tempo=audio_tempo,
        transcriber=transcriber,
        translation_enabled=translation_enabled,
    )
    try:
        queue.put_nowait(job)
    except asyncio.QueueFull:
        if isinstance(scoped_store, ScopedStateStore):
            await asyncio.to_thread(scoped_store.set_job_status, job_id, "failed")
        await edit_status_message(
            status,
            "The transcription queue is temporarily full. Please resend this media later.",
            job_id=job_id,
        )


async def handle_video_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_media_upload(update, context)


async def process_media_message(
    message: Message,
    attachment: Video | Audio | Voice | Document,
    settings: Settings,
    context: ContextTypes.DEFAULT_TYPE,
    status_ref: dict[str, Message | None],
    job_id: str,
    audio_tempo: float,
    *,
    transcriber_override: SpeechTranscriber | None = None,
    translation_enabled_override: bool | None = None,
) -> None:
    transcriber: SpeechTranscriber = transcriber_override or context.bot_data["transcriber"]
    translation_enabled = (
        get_runtime_translation_enabled(context, settings)
        if translation_enabled_override is None
        else translation_enabled_override
    )
    base_name = get_media_attachment_basename(attachment)
    job_started = time.monotonic()
    total_steps = 7 if translation_enabled else 6
    sending_step = 7 if translation_enabled else 6

    with tempfile.TemporaryDirectory(prefix="telegram-transcript-") as tmp:
        work_dir = Path(tmp)
        source_path = work_dir / f"source{get_media_attachment_suffix(attachment)}"
        audio_path = work_dir / "audio.flac"

        status = status_ref.get("message")
        if status is None:
            status = await reply_to_source(
                message,
                f"{get_media_kind_label(message, attachment)} received. Starting transcription...",
            )
        status_ref["message"] = status

        ensure_temporary_space(get_attachment_file_size(attachment))

        step_started = time.monotonic()
        await edit_status_message(status, f"Step 1/{total_steps}: downloading media...", job_id=job_id)
        logger.info(
            "job %s step 1/%d downloading media: suffix=%s telegram_file_size=%s",
            job_id,
            total_steps,
            get_media_attachment_suffix(attachment),
            get_attachment_file_size(attachment),
        )
        await download_message_media(context, message, source_path)
        source_bytes = source_path.stat().st_size
        logger.info(
            "job %s step 1/%d downloaded media: source_bytes=%d duration_ms=%d",
            job_id,
            total_steps,
            source_bytes,
            elapsed_ms(step_started),
        )
        if source_bytes > settings.max_video_bytes:
            logger.info(
                "job %s rejected after download: source_bytes=%d max_video_bytes=%d",
                job_id,
                source_bytes,
                settings.max_video_bytes,
            )
            await edit_status_message(status, "Media is larger than the configured upload limit.", job_id=job_id)
            return
        ensure_temporary_space(source_bytes)

        await edit_status_message(
            status,
            f"Step 2/{total_steps}: extracting lossless FLAC audio at {audio_tempo:g}x...",
            job_id=job_id,
        )
        step_started = time.monotonic()
        logger.info(
            "job %s step 2/%d extracting FLAC audio: source_bytes=%d audio_tempo=%g",
            job_id,
            total_steps,
            source_bytes,
            audio_tempo,
        )
        await asyncio.to_thread(
            extract_audio,
            source_path,
            audio_path,
            audio_tempo=audio_tempo,
        )
        audio_bytes = audio_path.stat().st_size
        logger.info(
            "job %s step 2/%d extracted FLAC audio: audio_bytes=%d duration_ms=%d",
            job_id,
            total_steps,
            audio_bytes,
            elapsed_ms(step_started),
        )

        await edit_status_message(status, f"Step 3/{total_steps}: preparing audio chunks...", job_id=job_id)
        step_started = time.monotonic()
        logger.info(
            "job %s step 3/%d preparing bounded transcription chunks: audio_bytes=%d",
            job_id,
            total_steps,
            audio_bytes,
        )
        chunks = await asyncio.to_thread(
            split_audio_to_timed_chunks,
            audio_path,
            work_dir / "chunks",
            time_scale=audio_tempo,
        )
        chunk_sizes = [chunk.path.stat().st_size for chunk in chunks]
        logger.info(
            "job %s step 3/%d prepared chunks: chunk_count=%d total_chunk_bytes=%d min_chunk_bytes=%d max_chunk_bytes=%d duration_ms=%d",
            job_id,
            total_steps,
            len(chunks),
            sum(chunk_sizes),
            min(chunk_sizes),
            max(chunk_sizes),
            elapsed_ms(step_started),
        )
        last_translation_status_attempt_at: float | None = None

        async def report_progress(event: str, data: object) -> None:
            nonlocal last_translation_status_attempt_at
            progress_data = data if isinstance(data, dict) else {}
            if event == "transcribing_chunk":
                index = progress_data.get("index")
                total = progress_data.get("total")
                await edit_status_message(
                    status,
                    f"Step 4/{total_steps}: transcribing chunk {index}/{total}...",
                    job_id=job_id,
                )
                logger.info(
                    "job %s step 4/%d transcribing chunk %s/%s: chunk_bytes=%s provider=%s model=%s",
                    job_id,
                    total_steps,
                    index,
                    total,
                    progress_data.get("chunk_bytes"),
                    progress_data.get("provider"),
                    progress_data.get("model"),
                )
            elif event == "chunk_transcribed":
                logger.info(
                    "job %s step 4/%d transcribed chunk %s/%s: raw_chars=%s",
                    job_id,
                    total_steps,
                    progress_data.get("index"),
                    progress_data.get("total"),
                    progress_data.get("raw_chars"),
                )
            elif event == "provider_fallback":
                index = progress_data.get("index")
                total = progress_data.get("total")
                failed_provider = format_provider_name(progress_data.get("failed_provider"))
                next_provider = format_provider_name(progress_data.get("next_provider"))
                await edit_status_message(
                    status,
                    f"{failed_provider} could not produce valid SRT; "
                    f"retrying chunk {index}/{total} with {next_provider}...",
                    job_id=job_id,
                )
                logger.warning(
                    "job %s step 4/%d provider fallback for chunk %s/%s: failed_provider=%s failed_model=%s next_provider=%s next_model=%s",
                    job_id,
                    total_steps,
                    index,
                    total,
                    progress_data.get("failed_provider"),
                    progress_data.get("failed_model"),
                    progress_data.get("next_provider"),
                    progress_data.get("next_model"),
                )
            elif event == "refining_transcription":
                await edit_status_message(
                    status,
                    f"Step 5/{total_steps}: refining Iraqi Arabic transcription...",
                    job_id=job_id,
                )
                logger.info(
                    "job %s step 5/%d refining Iraqi Arabic SRT: srt_chars=%s srt_bytes=%s model=%s",
                    job_id,
                    total_steps,
                    progress_data.get("raw_chars"),
                    progress_data.get("raw_bytes"),
                    progress_data.get("model"),
                )
            elif event == "transcription_refinement_complete":
                logger.info(
                    "job %s step 5/%d refined Iraqi Arabic SRT: refined_srt_chars=%s refined_transcript_chars=%s model=%s",
                    job_id,
                    total_steps,
                    progress_data.get("refined_srt_chars"),
                    progress_data.get("refined_transcript_chars"),
                    progress_data.get("model"),
                )
            elif event == "transcription_refinement_failed":
                await edit_status_message(
                    status,
                    f"Step 5/{total_steps}: refinement failed; using original subtitles...",
                    job_id=job_id,
                )
                logger.warning(
                    "job %s step 5/%d Iraqi Arabic refinement failed; original subtitles will be used: model=%s",
                    job_id,
                    total_steps,
                    progress_data.get("model"),
                )
            elif event == "translating_subtitles":
                index = progress_data.get("index")
                total = progress_data.get("total")
                if isinstance(total, int) and total > 1:
                    status_text = f"Step 6/{total_steps}: translating subtitle cue {index}/{total}..."
                else:
                    status_text = f"Step 6/{total_steps}: translating subtitles..."
                now = time.monotonic()
                if (
                    last_translation_status_attempt_at is None
                    or now - last_translation_status_attempt_at >= STATUS_PROGRESS_EDIT_INTERVAL_SECONDS
                ):
                    last_translation_status_attempt_at = now
                    await edit_status_message(status, status_text, job_id=job_id)
                logger.info(
                    "job %s step 6/%d translating subtitle cue %s/%s: srt_chars=%s srt_bytes=%s model=%s",
                    job_id,
                    total_steps,
                    index,
                    total,
                    progress_data.get("raw_chars"),
                    progress_data.get("raw_bytes"),
                    progress_data.get("model"),
                )
            elif event == "translation_complete":
                logger.info(
                    "job %s step 6/%d translated SRT and derived transcript: translated_srt_chars=%s line_translated_transcript_chars=%s",
                    job_id,
                    total_steps,
                    progress_data.get("translated_srt_chars"),
                    progress_data.get("line_translated_transcript_chars"),
                )
            elif event == "translation_failed":
                await edit_status_message(
                    status,
                    f"Step 6/{total_steps}: translation failed; preparing Arabic subtitles...",
                    job_id=job_id,
                )
                logger.warning(
                    "job %s step 6/%d translation failed; Arabic subtitles will be delivered: model=%s",
                    job_id,
                    total_steps,
                    progress_data.get("model"),
                )

        transcription_result = normalize_transcription_result(
            await transcriber.transcribe_chunks_async(chunks, progress_callback=report_progress)
        )

    raw_transcript = transcription_result.raw_transcript.strip() or "No speech was detected."
    cleaned_transcript = (
        transcription_result.cleaned_transcript or transcription_result.refined_transcript or ""
    ).strip()
    final_transcript = raw_transcript
    translated_srt = (
        transcription_result.translated_srt.strip() if transcription_result.translated_srt is not None else None
    )
    line_translated_transcript = (
        transcription_result.line_translated_transcript.strip()
        if transcription_result.line_translated_transcript is not None
        else None
    )
    persian_transcript = (
        transcription_result.persian_transcript.strip()
        if transcription_result.persian_transcript is not None
        else None
    )
    srt = translated_srt or render_srt(transcription_result.subtitle_cues)
    await edit_status_message(
        status,
        f"Step {sending_step}/{total_steps}: sending transcript...",
        job_id=job_id,
    )
    logger.info(
        "job %s step %d/%d sending transcript: final_chars=%d translated_srt_chars=%s line_translated_transcript_chars=%s srt_cues=%d delivery=%s",
        job_id,
        sending_step,
        total_steps,
        len(final_transcript),
        len(translated_srt) if translated_srt is not None else None,
        len(line_translated_transcript) if line_translated_transcript is not None else None,
        len(transcription_result.subtitle_cues),
        "text" if should_send_as_text(final_transcript) else "document",
    )
    await send_transcript(
        message,
        final_transcript,
        filename=f"{base_name}.transcription.txt",
        caption="Transcription",
    )
    if cleaned_transcript and cleaned_transcript != raw_transcript:
        await send_transcript(
            message,
            cleaned_transcript,
            filename=f"{base_name}.iraqi-clean.txt",
            caption="Cleaned Iraqi reading version",
        )
    if srt:
        await send_srt(message, srt, filename=f"{base_name}.srt")
    if persian_transcript:
        await send_transcript(
            message,
            persian_transcript,
            filename=f"{base_name}.fa.txt",
            caption="Persian translation",
        )
    if transcription_result.warnings:
        await edit_status_message(
            status,
            "Transcript ready. " + " ".join(transcription_result.warnings),
            job_id=job_id,
        )
    elif srt:
        await edit_status_message(status, "Transcript ready.", job_id=job_id)
    else:
        await edit_status_message(status, "Transcript ready. SRT unavailable for this provider/model.", job_id=job_id)
    logger.info(
        "job %s completed: duration_ms=%d final_chars=%d translated_srt_chars=%s line_translated_transcript_chars=%s srt_cues=%d",
        job_id,
        elapsed_ms(job_started),
        len(final_transcript),
        len(translated_srt) if translated_srt is not None else None,
        len(line_translated_transcript) if line_translated_transcript is not None else None,
        len(transcription_result.subtitle_cues),
    )


async def process_video_message(
    message: Message,
    attachment: Video | Document,
    settings: Settings,
    context: ContextTypes.DEFAULT_TYPE,
    status_ref: dict[str, Message | None],
    job_id: str,
    audio_tempo: float,
) -> None:
    await process_media_message(message, attachment, settings, context, status_ref, job_id, audio_tempo)


async def send_transcript(
    message: Message,
    transcript: str,
    *,
    filename: str = "transcript.txt",
    caption: str = "Transcript",
) -> None:
    formatted_transcript = format_transcript_for_delivery(transcript)
    if should_send_as_text(formatted_transcript):
        for chunk in split_text_for_telegram(formatted_transcript):
            await reply_to_source(message, chunk)
        return

    transcript_file = BytesIO(formatted_transcript.encode("utf-8"))
    transcript_file.name = filename
    transcript_file.seek(0)
    await message.reply_document(
        document=InputFile(transcript_file, filename=filename),
        caption=caption,
        **source_reply_kwargs(message),
    )


async def send_srt(message: Message, srt: str, *, filename: str = "transcript.srt") -> None:
    srt_file = BytesIO(srt.encode("utf-8"))
    srt_file.name = filename
    srt_file.seek(0)
    await message.reply_document(
        document=InputFile(srt_file, filename=filename),
        caption="SRT subtitles",
        **source_reply_kwargs(message),
    )


def normalize_transcription_result(result: object) -> TranscriptionResult:
    if isinstance(result, TranscriptionResult):
        return result
    if isinstance(result, str):
        return TranscriptionResult(raw_transcript=result)
    raise TranscriptionError("Transcriber returned an unsupported result.")


def format_provider_name(value: object) -> str:
    if value == "openai":
        return "OpenAI"
    if isinstance(value, str) and value:
        return value.title()
    return "Provider"


async def reply_to_source(message: Message, text: str) -> Message:
    return await message.reply_text(text, **source_reply_kwargs(message))


async def edit_status_message(status: Message, text: str, *, job_id: str) -> bool:
    try:
        await status.edit_text(text)
    except RetryAfter as exc:
        logger.warning(
            "job %s skipped Telegram status edit after flood control: retry_after=%s text=%r",
            job_id,
            format_retry_after(get_retry_after_value(exc)),
            text,
        )
        return False
    return True


def get_retry_after_value(exc: RetryAfter) -> object:
    retry_after = getattr(exc, "_retry_after", None)
    if retry_after is not None:
        return retry_after
    return exc.retry_after


def format_retry_after(retry_after: object) -> str:
    total_seconds = getattr(retry_after, "total_seconds", None)
    if callable(total_seconds):
        return f"{total_seconds():g}"
    return str(retry_after)


def source_reply_kwargs(message: Message) -> dict[str, int | bool]:
    kwargs: dict[str, int | bool] = {}
    message_id = getattr(message, "message_id", None)
    if message_id is not None:
        kwargs["reply_to_message_id"] = message_id
        kwargs["allow_sending_without_reply"] = True

    message_thread_id = getattr(message, "message_thread_id", None)
    if message_thread_id is not None:
        kwargs["message_thread_id"] = message_thread_id
    return kwargs


def get_chat_type(message: Message) -> str | None:
    chat = getattr(message, "chat", None)
    return getattr(chat, "type", None)


def parse_tempo_command_args(args: object) -> float | None:
    if not isinstance(args, list) or len(args) != 1:
        return None
    try:
        return parse_audio_tempo(args[0])
    except ConfigError:
        return None


def get_update_user_id(update: Update) -> int | None:
    user = getattr(update, "effective_user", None)
    user_id = getattr(user, "id", None)
    return user_id if isinstance(user_id, int) else None


def get_preference_scope(message: Message, user_id: int | None) -> str:
    chat_type = get_chat_type(message)
    chat_id = get_message_chat_id(message)
    if chat_type in GROUP_CHAT_TYPES and chat_id is not None:
        return f"chat:{chat_id}"
    if user_id is not None:
        return f"user:{user_id}"
    if chat_id is not None:
        return f"chat:{chat_id}"
    raise RuntimeStateError("Unable to determine a preference scope for this message.")


def get_scoped_preferences(
    context: ContextTypes.DEFAULT_TYPE,
    message: Message,
    user_id: int | None,
) -> RuntimePreferences:
    store = context.bot_data.get("scoped_state_store")
    if isinstance(store, ScopedStateStore):
        return store.load_preferences(get_preference_scope(message, user_id))
    settings: Settings = context.bot_data["settings"]
    return RuntimePreferences(
        audio_tempo=get_runtime_audio_tempo(context, settings),
        transcription_model=get_runtime_model_key(context),
        transcription_refinement_model=get_runtime_transcription_refinement_model(context, settings),
        translation_enabled=get_runtime_translation_enabled(context, settings),
        translation_model=get_runtime_translation_model(context, settings),
        translation_prompt=get_runtime_translation_prompt(context),
    )


def save_scoped_preferences(
    context: ContextTypes.DEFAULT_TYPE,
    message: Message,
    user_id: int | None,
    preferences: RuntimePreferences,
) -> None:
    store = context.bot_data.get("scoped_state_store")
    if isinstance(store, ScopedStateStore):
        try:
            store.save_preferences(get_preference_scope(message, user_id), preferences)
        except (OSError, sqlite3.Error) as exc:
            raise RuntimeStateError(f"Unable to save scoped settings: {exc}") from exc
        return
    persist_runtime_preferences(
        context,
        audio_tempo=preferences.audio_tempo,
        transcription_model=preferences.transcription_model,
        transcription_refinement_model=preferences.transcription_refinement_model,
        translation_enabled=preferences.translation_enabled,
        translation_model=preferences.translation_model,
        translation_prompt=preferences.translation_prompt,
    )
    context.bot_data.update(
        {
            "audio_tempo": preferences.audio_tempo,
            "transcription_model": preferences.transcription_model,
            "transcription_refinement_model": preferences.transcription_refinement_model,
            "translation_enabled": preferences.translation_enabled,
            "translation_model": preferences.translation_model,
            "translation_prompt": preferences.translation_prompt,
        }
    )


def activate_legacy_runtime_transcriber(
    context: ContextTypes.DEFAULT_TYPE,
    transcriber: SpeechTranscriber,
) -> None:
    if not isinstance(context.bot_data.get("scoped_state_store"), ScopedStateStore):
        context.bot_data["transcriber"] = transcriber


async def can_change_scope_preferences(
    context: ContextTypes.DEFAULT_TYPE,
    message: Message,
    user_id: int | None,
) -> bool:
    if get_chat_type(message) not in GROUP_CHAT_TYPES:
        return True
    if user_id is None:
        return False
    bot = getattr(context, "bot", None)
    get_chat_member = getattr(bot, "get_chat_member", None)
    chat_id = get_message_chat_id(message)
    if not callable(get_chat_member) or chat_id is None:
        return False
    try:
        member = await get_chat_member(chat_id, user_id)
    except TelegramError:
        logger.warning("Unable to verify group administrator for chat %s", chat_id)
        return False
    status = getattr(member, "status", None)
    return status in {"administrator", "creator", "owner"}


def get_runtime_audio_tempo(context: ContextTypes.DEFAULT_TYPE, settings: Settings) -> float:
    candidate = context.bot_data.get("audio_tempo", settings.audio_tempo)
    try:
        return parse_audio_tempo(str(candidate))
    except ConfigError:
        return settings.audio_tempo


def parse_model_command_arg(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    return TRANSCRIPTION_MODEL_ALIASES.get(raw.strip().lower())


def parse_translation_model_command_arg(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    return TRANSLATION_MODEL_ALIASES.get(raw.strip().lower())


def parse_transcription_refinement_model_command_arg(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    return TRANSCRIPTION_REFINEMENT_MODEL_ALIASES.get(raw.strip().lower())


def get_runtime_model_key(context: ContextTypes.DEFAULT_TYPE) -> str:
    candidate = context.bot_data.get("transcription_model", DEFAULT_TRANSCRIPTION_MODEL_KEY)
    if isinstance(candidate, str) and candidate in TRANSCRIPTION_MODEL_OPTIONS:
        return candidate
    return DEFAULT_TRANSCRIPTION_MODEL_KEY


def get_runtime_translation_model(context: ContextTypes.DEFAULT_TYPE, settings: Settings) -> str:
    candidate = context.bot_data.get("translation_model", settings.openrouter_refine_model)
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return settings.openrouter_refine_model


def get_runtime_translation_enabled(
    context: ContextTypes.DEFAULT_TYPE | Application,
    settings: Settings,
) -> bool:
    candidate = context.bot_data.get("translation_enabled", settings.refine)
    return candidate if isinstance(candidate, bool) else settings.refine


def get_runtime_transcription_refinement_model(
    context: ContextTypes.DEFAULT_TYPE,
    settings: Settings,
) -> str:
    candidate = context.bot_data.get(
        "transcription_refinement_model",
        settings.openrouter_transcription_refinement_model,
    )
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return settings.openrouter_transcription_refinement_model


def get_runtime_translation_prompt(context: ContextTypes.DEFAULT_TYPE) -> str:
    candidate = context.bot_data.get("translation_prompt", DEFAULT_TRANSLATION_PROMPT_KEY)
    if isinstance(candidate, str):
        canonical = TRANSLATION_PROMPT_ALIASES.get(candidate)
        if canonical is not None:
            return canonical
    return DEFAULT_TRANSLATION_PROMPT_KEY


def persist_runtime_preferences(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    audio_tempo: float | None = None,
    transcription_model: str | None = None,
    transcription_refinement_model: str | None = None,
    translation_enabled: bool | None = None,
    translation_model: str | None = None,
    translation_prompt: str | None = None,
) -> None:
    settings: Settings = context.bot_data["settings"]
    store = context.bot_data.get("runtime_preferences_store")
    if not isinstance(store, RuntimePreferencesStore):
        raise RuntimeStateError("Runtime preferences store is unavailable.")
    store.save(
        RuntimePreferences(
            audio_tempo=audio_tempo if audio_tempo is not None else get_runtime_audio_tempo(context, settings),
            transcription_model=transcription_model or get_runtime_model_key(context),
            transcription_refinement_model=(
                transcription_refinement_model
                or get_runtime_transcription_refinement_model(context, settings)
            ),
            translation_enabled=(
                get_runtime_translation_enabled(context, settings)
                if translation_enabled is None
                else translation_enabled
            ),
            translation_model=translation_model or get_runtime_translation_model(context, settings),
            translation_prompt=translation_prompt or get_runtime_translation_prompt(context),
        )
    )


def get_runtime_transcriber_info(context: ContextTypes.DEFAULT_TYPE, settings: Settings) -> tuple[str, str]:
    transcriber = context.bot_data.get("transcriber")
    provider_name = getattr(transcriber, "provider_name", None)
    model = getattr(transcriber, "model", None)
    if isinstance(provider_name, str) and isinstance(model, str):
        return provider_name, model
    option = TRANSCRIPTION_MODEL_OPTIONS[DEFAULT_TRANSCRIPTION_MODEL_KEY]
    return option.provider, option.model


def format_model_settings_message(context: ContextTypes.DEFAULT_TYPE, settings: Settings) -> str:
    current_model_key = get_runtime_model_key(context)
    lines = [
        f"Current primary transcription model: {TRANSCRIPTION_MODEL_OPTIONS[current_model_key].label}",
        "",
        "Available models:",
    ]
    for option in TRANSCRIPTION_MODEL_OPTIONS.values():
        availability = "available" if is_model_option_configured(option, settings) else "missing credentials"
        lines.append(f"- {option.key}: {option.label} ({availability})")
    lines.append("")
    lines.append("Use /model gemini, /model deepgram, /model whisper, or /model openai.")
    return "\n".join(lines)


def format_model_preferences_message(preferences: RuntimePreferences, settings: Settings) -> str:
    lines = [
        f"Current primary transcription model: {TRANSCRIPTION_MODEL_OPTIONS[preferences.transcription_model].label}",
        "",
        "Available models:",
    ]
    for option in TRANSCRIPTION_MODEL_OPTIONS.values():
        availability = "available" if is_model_option_configured(option, settings) else "missing credentials"
        lines.append(f"- {option.key}: {option.label} ({availability})")
    lines.extend(("", "Use /model gemini, /model deepgram, /model whisper, or /model openai."))
    return "\n".join(lines)


def format_unknown_model_message() -> str:
    options = ", ".join(option.key for option in TRANSCRIPTION_MODEL_OPTIONS.values())
    return f"Unknown transcription model. Available models: {options}."


def format_transcription_refinement_model_settings_message(
    context: ContextTypes.DEFAULT_TYPE,
    settings: Settings,
) -> str:
    current_model = get_runtime_transcription_refinement_model(context, settings)
    current_label = next(
        (
            option.label
            for option in TRANSCRIPTION_REFINEMENT_MODEL_OPTIONS.values()
            if option.model == current_model
        ),
        current_model,
    )
    lines = [f"Current transcription refinement model: {current_label}", "", "Available models:"]
    for option in TRANSCRIPTION_REFINEMENT_MODEL_OPTIONS.values():
        lines.append(f"- {option.key}: {option.label}")
    lines.append("")
    lines.append("Use /refiner gpt or /refiner gemini.")
    return "\n".join(lines)


def format_transcription_refinement_preferences_message(preferences: RuntimePreferences) -> str:
    current_label = next(
        (
            option.label
            for option in TRANSCRIPTION_REFINEMENT_MODEL_OPTIONS.values()
            if option.model == preferences.transcription_refinement_model
        ),
        preferences.transcription_refinement_model,
    )
    lines = [f"Current transcription refinement model: {current_label}", "", "Available models:"]
    lines.extend(f"- {option.key}: {option.label}" for option in TRANSCRIPTION_REFINEMENT_MODEL_OPTIONS.values())
    lines.extend(("", "Use /refiner gpt or /refiner gemini."))
    return "\n".join(lines)


def format_unknown_transcription_refinement_model_message() -> str:
    options = ", ".join(option.key for option in TRANSCRIPTION_REFINEMENT_MODEL_OPTIONS.values())
    return f"Unknown transcription refinement model. Available models: {options}."


def format_translation_model_settings_message(context: ContextTypes.DEFAULT_TYPE, settings: Settings) -> str:
    current_model = get_runtime_translation_model(context, settings)
    current_label = next(
        (option.label for option in TRANSLATION_MODEL_OPTIONS.values() if option.model == current_model),
        current_model,
    )
    lines = [
        f"Current translation model: {current_label}",
        f"Translation: {'enabled' if get_runtime_translation_enabled(context, settings) else 'disabled'}",
        "",
        "Available models:",
    ]
    for option in TRANSLATION_MODEL_OPTIONS.values():
        lines.append(f"- {option.key}: {option.label}")
    lines.append("")
    lines.append("Use /tmodel gemini, /tmodel gpt, or /tmodel claude.")
    return "\n".join(lines)


def format_translation_preferences_message(preferences: RuntimePreferences) -> str:
    current_label = next(
        (option.label for option in TRANSLATION_MODEL_OPTIONS.values() if option.model == preferences.translation_model),
        preferences.translation_model,
    )
    lines = [
        f"Current translation model: {current_label}",
        f"Translation: {'enabled' if preferences.translation_enabled else 'disabled'}",
        "",
        "Available models:",
    ]
    lines.extend(f"- {option.key}: {option.label}" for option in TRANSLATION_MODEL_OPTIONS.values())
    lines.extend(("", "Use /tmodel gemini, /tmodel gpt, or /tmodel claude."))
    return "\n".join(lines)


def format_unknown_translation_model_message() -> str:
    options = ", ".join(option.key for option in TRANSLATION_MODEL_OPTIONS.values())
    return f"Unknown translation model. Available models: {options}."


def format_translation_prompt_settings_message(context: ContextTypes.DEFAULT_TYPE) -> str:
    current_prompt = get_runtime_translation_prompt(context)
    return (
        f"Current translation prompt: {current_prompt}\n\n"
        "Available prompts:\n- natural\n- literal\n\n"
        "Use /translation natural or /translation literal. Legacy aliases v2 and normal remain accepted."
    )


def format_translation_prompt_preferences_message(preferences: RuntimePreferences) -> str:
    return (
        f"Current translation prompt: {preferences.translation_prompt}\n\n"
        "Available prompts:\n- natural\n- literal\n\n"
        "Use /translation natural or /translation literal. Legacy aliases v2 and normal remain accepted."
    )


def format_unknown_translation_prompt_message() -> str:
    return "Unknown translation prompt. Available prompts: natural, literal."


def is_model_option_configured(option: TranscriptionModelOption, settings: Settings) -> bool:
    if option.provider == "deepgram":
        return bool(settings.deepgram_api_key)
    if option.provider == "openai":
        return bool(settings.openai_api_key)
    if option.provider in {"gemini", "whisper"}:
        return bool(settings.openrouter_api_key)
    return False


def missing_model_credential_message(option: TranscriptionModelOption) -> str:
    if option.provider == "deepgram":
        return "DEEPGRAM_API_KEY is required for Deepgram transcription."
    if option.provider == "openai":
        return "OPENAI_API_KEY is required for direct OpenAI transcription."
    if option.provider == "whisper":
        return "OPENROUTER_API_KEY is required for OpenRouter Whisper transcription."
    return "OPENROUTER_API_KEY is required for OpenRouter transcription."


def is_authorized(settings: Settings, user_id: int | None) -> bool:
    if not settings.allowed_telegram_user_ids:
        return True
    return user_id in settings.allowed_telegram_user_ids


async def download_message_media(context: ContextTypes.DEFAULT_TYPE, message: Message, target_path: Path) -> Path:
    downloader = context.bot_data.get("media_downloader")
    download_media = getattr(downloader, "download_message_media", None)
    if not callable(download_media):
        raise TelegramDownloadError("Telegram media downloader is not available.")

    chat_id = get_message_chat_id(message)
    message_id = getattr(message, "message_id", None)
    if chat_id is None or message_id is None:
        raise TelegramDownloadError("Telegram message did not include chat and message identifiers.")
    return await download_media(chat_id, message_id, target_path)


def get_message_chat_id(message: Message) -> int | None:
    chat_id = getattr(message, "chat_id", None)
    if isinstance(chat_id, int):
        return chat_id
    chat = getattr(message, "chat", None)
    candidate = getattr(chat, "id", None)
    return candidate if isinstance(candidate, int) else None


def get_media_attachment(message: Message) -> Video | Audio | Voice | Document | None:
    video = getattr(message, "video", None)
    if video is not None:
        return video
    audio = getattr(message, "audio", None)
    if audio is not None:
        return audio
    voice = getattr(message, "voice", None)
    if voice is not None:
        return voice
    document = getattr(message, "document", None)
    if document is not None and (is_video_document(document) or is_audio_document(document)):
        return document
    return None


def get_video_attachment(message: Message) -> Video | Document | None:
    video = getattr(message, "video", None)
    if video is not None:
        return video
    document = getattr(message, "document", None)
    if document is not None and is_video_document(document):
        return document
    return None


def is_audio_document(document: Document) -> bool:
    mime_type = getattr(document, "mime_type", None) or ""
    if mime_type.startswith("audio/"):
        return True
    file_name = getattr(document, "file_name", None) or ""
    return Path(file_name).suffix.lower() in AUDIO_EXTENSIONS


def is_video_document(document: Document) -> bool:
    mime_type = getattr(document, "mime_type", None) or ""
    if mime_type.startswith("video/"):
        return True
    file_name = getattr(document, "file_name", None) or ""
    return Path(file_name).suffix.lower() in VIDEO_EXTENSIONS


def get_media_attachment_suffix(attachment: Video | Audio | Voice | Document) -> str:
    if is_audio_attachment(attachment):
        return get_audio_attachment_suffix(attachment)
    return get_attachment_suffix(attachment)


def get_media_attachment_basename(attachment: Video | Audio | Voice | Document) -> str:
    # Name delivered outputs after the original upload; voice notes and some audio
    # arrive without a file_name, so fall back to a stable default base.
    file_name = getattr(attachment, "file_name", None)
    if isinstance(file_name, str):
        stem = Path(file_name).stem.strip()
        if stem:
            return stem
    return DEFAULT_OUTPUT_BASENAME


def get_audio_attachment_suffix(attachment: Audio | Voice | Document) -> str:
    file_name = getattr(attachment, "file_name", None)
    suffix = Path(file_name).suffix.lower() if file_name else ""
    if suffix in AUDIO_EXTENSIONS:
        return suffix

    mime_type = getattr(attachment, "mime_type", None)
    if isinstance(mime_type, str):
        return AUDIO_SUFFIX_BY_MIME.get(mime_type, ".ogg")
    return ".ogg"


def get_attachment_suffix(attachment: Video | Document) -> str:
    file_name = getattr(attachment, "file_name", None)
    suffix = Path(file_name).suffix.lower() if file_name else ""
    return suffix if suffix in VIDEO_EXTENSIONS else ".mp4"


def is_audio_attachment(attachment: Video | Audio | Voice | Document) -> bool:
    mime_type = getattr(attachment, "mime_type", "") or ""
    if isinstance(mime_type, str) and mime_type.startswith("audio/"):
        return True
    file_name = getattr(attachment, "file_name", None)
    if isinstance(file_name, str) and Path(file_name).suffix.lower() in AUDIO_EXTENSIONS:
        return True
    return isinstance(attachment, (Audio, Voice))


def get_media_kind_label(message: Message, attachment: Video | Audio | Voice | Document) -> str:
    if getattr(message, "voice", None) is attachment:
        return "Voice note"
    if is_audio_attachment(attachment):
        return "Audio"
    return "Video"


def get_attachment_file_size(attachment: Video | Audio | Voice | Document) -> int | None:
    file_size = getattr(attachment, "file_size", None)
    return file_size if isinstance(file_size, int) else None


def ensure_temporary_space(media_bytes: int | None) -> None:
    """Reject work that cannot fit its source, extracted audio, and chunk copies."""
    required_bytes = max(
        MINIMUM_TEMPORARY_FREE_BYTES,
        (media_bytes or 0) * TEMPORARY_SPACE_MULTIPLIER,
    )
    free_bytes = shutil.disk_usage(tempfile.gettempdir()).free
    if free_bytes < required_bytes:
        required_gib = required_bytes / (1024**3)
        free_gib = free_bytes / (1024**3)
        raise FfmpegError(
            f"Insufficient temporary disk space: {free_gib:.1f} GiB free, "
            f"at least {required_gib:.1f} GiB required."
        )


def elapsed_ms(started_at: float) -> int:
    return int((time.monotonic() - started_at) * 1000)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        settings = load_settings()
        ensure_ffmpeg_available()
        create_application(settings).run_polling(allowed_updates=Update.ALL_TYPES)
    except (ConfigError, FfmpegError, TelegramDownloadError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
