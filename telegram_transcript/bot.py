from __future__ import annotations

import asyncio
import logging
import tempfile
import time
import uuid
from io import BytesIO
from pathlib import Path

from telegram import Document, InputFile, Message, Update, Video
from telegram.constants import ChatType
from telegram.error import BadRequest
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from telegram_transcript.config import ConfigError, Settings, load_settings, mb_to_bytes, parse_audio_tempo
from telegram_transcript.ffmpeg import FfmpegError, ensure_ffmpeg_available, extract_audio, split_audio_to_chunks
from telegram_transcript.telegram_utils import should_send_as_text, split_text_for_telegram
from telegram_transcript.transcriber import OpenAITranscriber, TranscriptionError

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
GROUP_CHAT_TYPES = {ChatType.GROUP, ChatType.SUPERGROUP}
HOSTED_TELEGRAM_DOWNLOAD_LIMIT_MB = 20.0
HOSTED_TELEGRAM_DOWNLOAD_LIMIT_BYTES = mb_to_bytes(HOSTED_TELEGRAM_DOWNLOAD_LIMIT_MB)
PENDING_NOISE_REDUCTION_KEY = "pending_noise_reduction_by_thread"
DEFAULT_NOISE_REDUCTION_MODE = "default"
EXTRA_NOISE_REDUCTION_MODE = "extra"
NOISE_REDUCTION_FILTERS = {
    DEFAULT_NOISE_REDUCTION_MODE: "anlmdn",
    EXTRA_NOISE_REDUCTION_MODE: "highpass=f=80,afftdn=nr=15,loudnorm",
}


def create_application(settings: Settings | None = None) -> Application:
    settings = settings or load_settings()
    transcriber = OpenAITranscriber(
        api_key=settings.openai_api_key,
        model=settings.openai_transcribe_model,
        refinement_model=settings.openai_refine_model,
        refine=settings.refine,
    )

    builder = Application.builder().token(settings.telegram_bot_token)
    if settings.telegram_api_base_url is not None:
        builder = builder.base_url(settings.telegram_api_base_url)
    if settings.telegram_api_base_file_url is not None:
        builder = builder.base_file_url(settings.telegram_api_base_file_url)
    if settings.telegram_local_mode:
        builder = builder.local_mode(True)

    app = builder.build()
    app.bot_data["settings"] = settings
    app.bot_data["transcriber"] = transcriber
    app.bot_data["job_semaphore"] = asyncio.Semaphore(settings.max_concurrent_jobs)
    app.bot_data["audio_tempo"] = settings.audio_tempo

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("tempo", handle_tempo_command))
    app.add_handler(CommandHandler("noise", handle_noise_command))
    app.add_handler(MessageHandler(video_message_filter(), handle_video_upload))
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_non_video))
    return app


def video_message_filter() -> filters.BaseFilter:
    document_video_filter = getattr(filters.Document, "VIDEO", filters.Document.ALL)
    return filters.VIDEO | document_video_filter


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    return


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    return


async def handle_non_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    return


async def handle_tempo_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or get_chat_type(message) not in GROUP_CHAT_TYPES:
        return

    settings: Settings = context.bot_data["settings"]
    if not is_allowed_group_topic(settings, message):
        return

    audio_tempo = parse_tempo_command_args(getattr(context, "args", None))
    if audio_tempo is None:
        return

    context.bot_data["audio_tempo"] = audio_tempo
    await reply_to_source(message, f"Tempo set to {audio_tempo:g}x.")


async def handle_noise_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or get_chat_type(message) == ChatType.CHANNEL:
        return

    settings: Settings = context.bot_data["settings"]
    if not is_allowed_group_topic(settings, message):
        return

    user_id = update.effective_user.id if update.effective_user else None
    if not is_authorized(settings, user_id):
        return

    noise_reduction_mode = parse_noise_command_args(getattr(context, "args", None))
    if noise_reduction_mode is None:
        return

    if not set_pending_noise_reduction_mode(context, message, noise_reduction_mode):
        return
    await reply_to_source(message, noise_reduction_confirmation_text(noise_reduction_mode))


async def handle_video_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    if get_chat_type(message) == ChatType.CHANNEL:
        return

    settings: Settings = context.bot_data["settings"]
    if not is_allowed_group_topic(settings, message):
        return

    user_id = update.effective_user.id if update.effective_user else None
    if not is_authorized(settings, user_id):
        await reply_to_source(message, "Sorry, this bot is not enabled for your Telegram account.")
        return

    attachment = get_video_attachment(message)
    if attachment is None:
        return

    file_size = getattr(attachment, "file_size", None)
    if file_size is not None and file_size > settings.max_video_bytes:
        return
    if exceeds_hosted_telegram_download_limit(file_size, settings):
        await reply_to_source(message, hosted_telegram_download_limit_message())
        return

    audio_tempo = get_runtime_audio_tempo(context, settings)
    noise_reduction_mode = consume_pending_noise_reduction_mode(context, message)
    job_id = uuid.uuid4().hex[:8]
    logger.info(
        "job %s queued: suffix=%s telegram_file_size=%s audio_tempo=%g noise_reduction=%s transcribe_model=%s refine=%s refine_model=%s",
        job_id,
        get_attachment_suffix(attachment),
        file_size,
        audio_tempo,
        noise_reduction_mode or "none",
        settings.openai_transcribe_model,
        settings.refine,
        settings.openai_refine_model,
    )

    semaphore: asyncio.Semaphore = context.bot_data["job_semaphore"]
    status_ref: dict[str, Message | None] = {"message": None}
    async with semaphore:
        try:
            logger.info("job %s started", job_id)
            await process_video_message(
                message,
                attachment,
                settings,
                context,
                status_ref,
                job_id,
                audio_tempo,
                noise_reduction_mode,
            )
        except (FfmpegError, TranscriptionError) as exc:
            logger.exception("job %s video transcription failed", job_id)
            status = status_ref["message"]
            if status is not None:
                await status.edit_text(f"Transcription failed: {exc}")
        except Exception:
            logger.exception("job %s unexpected video transcription failure", job_id)
            status = status_ref["message"]
            if status is not None:
                await status.edit_text("Transcription failed because of an unexpected error.")


async def process_video_message(
    message: Message,
    attachment: Video | Document,
    settings: Settings,
    context: ContextTypes.DEFAULT_TYPE,
    status_ref: dict[str, Message | None],
    job_id: str,
    audio_tempo: float,
    noise_reduction_mode: str | None = None,
) -> None:
    transcriber: OpenAITranscriber = context.bot_data["transcriber"]
    job_started = time.monotonic()
    noise_reduction_filter = get_noise_reduction_filter(noise_reduction_mode)

    with tempfile.TemporaryDirectory(prefix="telegram-transcript-") as tmp:
        work_dir = Path(tmp)
        video_path = work_dir / f"video{get_attachment_suffix(attachment)}"
        audio_path = work_dir / "audio.mp3"

        step_started = time.monotonic()
        logger.info(
            "job %s step 1/6 downloading video: suffix=%s telegram_file_size=%s",
            job_id,
            get_attachment_suffix(attachment),
            getattr(attachment, "file_size", None),
        )
        try:
            telegram_file = await attachment.get_file()
            await telegram_file.download_to_drive(custom_path=str(video_path))
        except BadRequest as exc:
            if is_file_too_big_error(exc) and get_hosted_telegram_download_limit_bytes(settings) is not None:
                logger.info(
                    "job %s rejected by Telegram getFile: telegram_file_size=%s hosted_limit_bytes=%d",
                    job_id,
                    getattr(attachment, "file_size", None),
                    HOSTED_TELEGRAM_DOWNLOAD_LIMIT_BYTES,
                )
                await reply_to_source(message, hosted_telegram_download_limit_message())
                return
            raise
        video_bytes = video_path.stat().st_size
        logger.info(
            "job %s step 1/6 downloaded video: video_bytes=%d duration_ms=%d",
            job_id,
            video_bytes,
            elapsed_ms(step_started),
        )
        if video_bytes > settings.max_video_bytes:
            logger.info(
                "job %s rejected after download: video_bytes=%d max_video_bytes=%d",
                job_id,
                video_bytes,
                settings.max_video_bytes,
            )
            return

        status = await reply_to_source(message, "Video received. Starting transcription...")
        status_ref["message"] = status

        await status.edit_text(extracting_audio_status_text(audio_tempo, noise_reduction_mode))
        step_started = time.monotonic()
        logger.info(
            "job %s step 2/6 extracting MP3 audio: video_bytes=%d audio_tempo=%g noise_reduction=%s",
            job_id,
            video_bytes,
            audio_tempo,
            noise_reduction_mode or "none",
        )
        await asyncio.to_thread(
            extract_audio,
            video_path,
            audio_path,
            audio_tempo=audio_tempo,
            noise_reduction_filter=noise_reduction_filter,
        )
        audio_bytes = audio_path.stat().st_size
        logger.info(
            "job %s step 2/6 extracted MP3 audio: audio_bytes=%d duration_ms=%d",
            job_id,
            audio_bytes,
            elapsed_ms(step_started),
        )

        await status.edit_text("Step 3/6: preparing audio chunks...")
        step_started = time.monotonic()
        logger.info(
            "job %s step 3/6 preparing chunks: audio_bytes=%d max_chunk_bytes=%d",
            job_id,
            audio_bytes,
            settings.max_openai_audio_bytes,
        )
        if audio_bytes <= settings.max_openai_audio_bytes:
            chunks = [audio_path]
        else:
            chunks = await asyncio.to_thread(
                split_audio_to_chunks,
                audio_path,
                work_dir / "chunks",
                settings.max_openai_audio_bytes,
            )
        chunk_sizes = [chunk.stat().st_size for chunk in chunks]
        logger.info(
            "job %s step 3/6 prepared chunks: chunk_count=%d total_chunk_bytes=%d min_chunk_bytes=%d max_chunk_bytes=%d duration_ms=%d",
            job_id,
            len(chunks),
            sum(chunk_sizes),
            min(chunk_sizes),
            max(chunk_sizes),
            elapsed_ms(step_started),
        )

        async def report_progress(event: str, data: object) -> None:
            progress_data = data if isinstance(data, dict) else {}
            if event == "transcribing_chunk":
                index = progress_data.get("index")
                total = progress_data.get("total")
                await status.edit_text(f"Step 4/6: transcribing chunk {index}/{total}...")
                logger.info(
                    "job %s step 4/6 transcribing chunk %s/%s: chunk_bytes=%s model=%s",
                    job_id,
                    index,
                    total,
                    progress_data.get("chunk_bytes"),
                    progress_data.get("model"),
                )
            elif event == "chunk_transcribed":
                logger.info(
                    "job %s step 4/6 transcribed chunk %s/%s: raw_chars=%s",
                    job_id,
                    progress_data.get("index"),
                    progress_data.get("total"),
                    progress_data.get("raw_chars"),
                )
            elif event == "refining_transcript":
                await status.edit_text("Step 5/6: refining transcript...")
                logger.info(
                    "job %s step 5/6 refining transcript: raw_chars=%s model=%s",
                    job_id,
                    progress_data.get("raw_chars"),
                    progress_data.get("model"),
                )
            elif event == "refinement_complete":
                logger.info(
                    "job %s step 5/6 refined transcript: cleaned_chars=%s",
                    job_id,
                    progress_data.get("cleaned_chars"),
                )

        transcript = await transcriber.transcribe_chunks_async(chunks, progress_callback=report_progress)

    transcript = transcript.strip() or "No speech was detected."
    await status.edit_text("Step 6/6: sending transcript...")
    logger.info(
        "job %s step 6/6 sending transcript: output_chars=%d delivery=%s",
        job_id,
        len(transcript),
        "text" if should_send_as_text(transcript) else "document",
    )
    await send_transcript(message, transcript)
    await status.edit_text("Transcript ready.")
    logger.info(
        "job %s completed: duration_ms=%d output_chars=%d",
        job_id,
        elapsed_ms(job_started),
        len(transcript),
    )


async def send_transcript(message: Message, transcript: str) -> None:
    if should_send_as_text(transcript):
        for chunk in split_text_for_telegram(transcript):
            await reply_to_source(message, chunk)
        return

    transcript_file = BytesIO(transcript.encode("utf-8"))
    transcript_file.name = "transcript.txt"
    transcript_file.seek(0)
    await message.reply_document(
        document=InputFile(transcript_file, filename="transcript.txt"),
        caption="Transcript",
        **source_reply_kwargs(message),
    )


async def reply_to_source(message: Message, text: str) -> Message:
    return await message.reply_text(text, **source_reply_kwargs(message))


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


def parse_noise_command_args(args: object) -> str | None:
    if not isinstance(args, list):
        return None
    if len(args) == 0:
        return DEFAULT_NOISE_REDUCTION_MODE
    if len(args) == 1 and isinstance(args[0], str) and args[0].casefold() == EXTRA_NOISE_REDUCTION_MODE:
        return EXTRA_NOISE_REDUCTION_MODE
    return None


def set_pending_noise_reduction_mode(
    context: ContextTypes.DEFAULT_TYPE,
    message: Message,
    noise_reduction_mode: str,
) -> bool:
    pending = get_pending_noise_reductions(context, create=True)
    if pending is None:
        return False
    pending[get_noise_reduction_thread_key(message)] = noise_reduction_mode
    return True


def consume_pending_noise_reduction_mode(context: ContextTypes.DEFAULT_TYPE, message: Message) -> str | None:
    pending = get_pending_noise_reductions(context, create=False)
    if pending is None:
        return None
    noise_reduction_mode = pending.pop(get_noise_reduction_thread_key(message), None)
    return noise_reduction_mode if noise_reduction_mode in NOISE_REDUCTION_FILTERS else None


def get_pending_noise_reductions(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    create: bool,
) -> dict[int | None, str] | None:
    chat_data = getattr(context, "chat_data", None)
    if chat_data is None:
        return None

    pending = chat_data.get(PENDING_NOISE_REDUCTION_KEY)
    if pending is None:
        if not create:
            return None
        pending = {}
        chat_data[PENDING_NOISE_REDUCTION_KEY] = pending
    if not isinstance(pending, dict):
        if not create:
            return None
        pending = {}
        chat_data[PENDING_NOISE_REDUCTION_KEY] = pending
    return pending


def get_noise_reduction_thread_key(message: Message) -> int | None:
    return getattr(message, "message_thread_id", None)


def get_noise_reduction_filter(noise_reduction_mode: str | None) -> str | None:
    if noise_reduction_mode is None:
        return None
    return NOISE_REDUCTION_FILTERS.get(noise_reduction_mode)


def describe_noise_reduction(noise_reduction_mode: str) -> str:
    if noise_reduction_mode == EXTRA_NOISE_REDUCTION_MODE:
        return "extra noise reduction"
    return "default noise reduction"


def noise_reduction_confirmation_text(noise_reduction_mode: str) -> str:
    label = describe_noise_reduction(noise_reduction_mode).capitalize()
    return f"{label} set for the next video."


def extracting_audio_status_text(audio_tempo: float, noise_reduction_mode: str | None) -> str:
    suffix = ""
    if noise_reduction_mode is not None:
        suffix = f" with {describe_noise_reduction(noise_reduction_mode)}"
    return f"Step 2/6: extracting MP3 audio at {audio_tempo:g}x{suffix}..."


def get_runtime_audio_tempo(context: ContextTypes.DEFAULT_TYPE, settings: Settings) -> float:
    candidate = context.bot_data.get("audio_tempo", settings.audio_tempo)
    try:
        return parse_audio_tempo(str(candidate))
    except ConfigError:
        return settings.audio_tempo


def is_authorized(settings: Settings, user_id: int | None) -> bool:
    if not settings.allowed_telegram_user_ids:
        return True
    return user_id in settings.allowed_telegram_user_ids


def is_allowed_group_topic(settings: Settings, message: Message) -> bool:
    if settings.allowed_telegram_topic_id is None:
        return True
    if get_chat_type(message) not in GROUP_CHAT_TYPES:
        return True
    return getattr(message, "message_thread_id", None) == settings.allowed_telegram_topic_id


def get_hosted_telegram_download_limit_bytes(settings: Settings) -> int | None:
    if settings.telegram_local_mode:
        return None
    return HOSTED_TELEGRAM_DOWNLOAD_LIMIT_BYTES


def exceeds_hosted_telegram_download_limit(file_size: int | None, settings: Settings) -> bool:
    limit = get_hosted_telegram_download_limit_bytes(settings)
    return file_size is not None and limit is not None and file_size > limit


def hosted_telegram_download_limit_message() -> str:
    limit = f"{HOSTED_TELEGRAM_DOWNLOAD_LIMIT_MB:g} MB"
    return (
        f"Telegram can only let bots download files up to {limit} on the hosted Bot API. "
        "Send it as a compressed video, make the file smaller, or run a local Telegram Bot API server for larger files."
    )


def is_file_too_big_error(exc: BadRequest) -> bool:
    return "file is too big" in str(exc).lower()


def get_video_attachment(message: Message) -> Video | Document | None:
    if message.video is not None:
        return message.video
    document = message.document
    if document is not None and is_video_document(document):
        return document
    return None


def is_video_document(document: Document) -> bool:
    mime_type = document.mime_type or ""
    if mime_type.startswith("video/"):
        return True
    file_name = document.file_name or ""
    return Path(file_name).suffix.lower() in VIDEO_EXTENSIONS


def get_attachment_suffix(attachment: Video | Document) -> str:
    file_name = getattr(attachment, "file_name", None)
    suffix = Path(file_name).suffix.lower() if file_name else ""
    return suffix if suffix in VIDEO_EXTENSIONS else ".mp4"


def elapsed_ms(started_at: float) -> int:
    return int((time.monotonic() - started_at) * 1000)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        settings = load_settings()
        ensure_ffmpeg_available()
        create_application(settings).run_polling(allowed_updates=Update.ALL_TYPES)
    except (ConfigError, FfmpegError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
