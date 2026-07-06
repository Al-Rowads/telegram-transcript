from __future__ import annotations

import asyncio
import logging
import tempfile
import time
import uuid
from io import BytesIO
from pathlib import Path

from telegram import Audio, Document, InputFile, Message, Update, Video, Voice
from telegram.constants import ChatType
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from telegram_transcript.config import ConfigError, Settings, load_settings, parse_audio_tempo
from telegram_transcript.ffmpeg import FfmpegError, ensure_ffmpeg_available, extract_audio, split_audio_to_timed_chunks
from telegram_transcript.models import TranscriptionResult
from telegram_transcript.telegram_utils import should_send_as_text, split_text_for_telegram
from telegram_transcript.telegram_downloader import TelegramDownloadError, TelegramMediaDownloader
from telegram_transcript.transcriber import (
    DeepgramSpeechToTextProvider,
    SpeechTranscriber,
    TranscriptRefiner,
    TranscriptionError,
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


def create_application(settings: Settings | None = None) -> Application:
    settings = settings or load_settings()
    transcriber = create_transcriber(settings)

    app = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .post_init(start_media_downloader)
        .post_shutdown(stop_media_downloader)
        .build()
    )
    app.bot_data["settings"] = settings
    app.bot_data["transcriber"] = transcriber
    app.bot_data["job_semaphore"] = asyncio.Semaphore(settings.max_concurrent_jobs)
    app.bot_data["audio_tempo"] = settings.audio_tempo

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("tempo", handle_tempo_command))
    app.add_handler(MessageHandler(media_message_filter(), handle_media_upload))
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_non_media))
    return app


def create_transcriber(settings: Settings) -> SpeechTranscriber:
    speech_to_text_provider = DeepgramSpeechToTextProvider(
        api_key=settings.deepgram_api_key,
        model=settings.deepgram_transcribe_model,
        language=settings.deepgram_language,
    )

    refiner = (
        TranscriptRefiner(
            api_key=settings.openai_api_key,
            model=settings.openai_refine_model,
        )
        if settings.refine
        else None
    )
    return SpeechTranscriber(speech_to_text_provider=speech_to_text_provider, refiner=refiner)


async def start_media_downloader(application: Application) -> None:
    settings: Settings = application.bot_data["settings"]
    downloader = TelegramMediaDownloader(settings)
    await downloader.start()
    application.bot_data["media_downloader"] = downloader


async def stop_media_downloader(application: Application) -> None:
    downloader = application.bot_data.get("media_downloader")
    if isinstance(downloader, TelegramMediaDownloader):
        await downloader.close()


def media_message_filter() -> filters.BaseFilter:
    return filters.VIDEO | filters.AUDIO | filters.VOICE | filters.Document.ALL


def video_message_filter() -> filters.BaseFilter:
    return media_message_filter()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    return


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    return


async def handle_non_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    return


async def handle_non_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_non_media(update, context)


async def handle_tempo_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or get_chat_type(message) not in GROUP_CHAT_TYPES:
        return

    audio_tempo = parse_tempo_command_args(getattr(context, "args", None))
    if audio_tempo is None:
        return

    context.bot_data["audio_tempo"] = audio_tempo
    await reply_to_source(message, f"Tempo set to {audio_tempo:g}x.")


async def handle_media_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    if get_chat_type(message) == ChatType.CHANNEL:
        return

    settings: Settings = context.bot_data["settings"]
    user_id = update.effective_user.id if update.effective_user else None
    if not is_authorized(settings, user_id):
        await reply_to_source(message, "Sorry, this bot is not enabled for your Telegram account.")
        return

    attachment = get_media_attachment(message)
    if attachment is None:
        return

    file_size = get_attachment_file_size(attachment)
    if file_size is not None and file_size > settings.max_video_bytes:
        return

    audio_tempo = get_runtime_audio_tempo(context, settings)
    job_id = uuid.uuid4().hex[:8]
    logger.info(
        "job %s queued: suffix=%s telegram_file_size=%s audio_tempo=%g stt_provider=%s transcribe_model=%s refine=%s refine_model=%s",
        job_id,
        get_media_attachment_suffix(attachment),
        file_size,
        audio_tempo,
        "deepgram",
        get_transcribe_model_name(settings),
        settings.refine,
        settings.openai_refine_model,
    )

    semaphore: asyncio.Semaphore = context.bot_data["job_semaphore"]
    status_ref: dict[str, Message | None] = {"message": None}
    async with semaphore:
        try:
            logger.info("job %s started", job_id)
            await process_media_message(message, attachment, settings, context, status_ref, job_id, audio_tempo)
        except (FfmpegError, TelegramDownloadError, TranscriptionError) as exc:
            logger.exception("job %s media transcription failed", job_id)
            status = status_ref["message"]
            if status is not None:
                await status.edit_text(f"Transcription failed: {exc}")
        except Exception:
            logger.exception("job %s unexpected media transcription failure", job_id)
            status = status_ref["message"]
            if status is not None:
                await status.edit_text("Transcription failed because of an unexpected error.")


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
) -> None:
    transcriber: SpeechTranscriber = context.bot_data["transcriber"]
    job_started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="telegram-transcript-") as tmp:
        work_dir = Path(tmp)
        source_path = work_dir / f"source{get_media_attachment_suffix(attachment)}"
        audio_path = work_dir / "audio.mp3"

        status = await reply_to_source(
            message,
            f"{get_media_kind_label(message, attachment)} received. Starting transcription...",
        )
        status_ref["message"] = status

        step_started = time.monotonic()
        await status.edit_text("Step 1/6: downloading media...")
        logger.info(
            "job %s step 1/6 downloading media: suffix=%s telegram_file_size=%s",
            job_id,
            get_media_attachment_suffix(attachment),
            get_attachment_file_size(attachment),
        )
        await download_message_media(context, message, source_path)
        source_bytes = source_path.stat().st_size
        logger.info(
            "job %s step 1/6 downloaded media: source_bytes=%d duration_ms=%d",
            job_id,
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
            await status.edit_text("Media is larger than the configured upload limit.")
            return

        await status.edit_text(f"Step 2/6: extracting MP3 audio at {audio_tempo:g}x...")
        step_started = time.monotonic()
        logger.info(
            "job %s step 2/6 extracting MP3 audio: source_bytes=%d audio_tempo=%g",
            job_id,
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
            "job %s step 2/6 extracted MP3 audio: audio_bytes=%d duration_ms=%d",
            job_id,
            audio_bytes,
            elapsed_ms(step_started),
        )

        await status.edit_text("Step 3/6: preparing audio chunks...")
        step_started = time.monotonic()
        logger.info(
            "job %s step 3/6 preparing 1300-second chunks: audio_bytes=%d",
            job_id,
            audio_bytes,
        )
        chunks = await asyncio.to_thread(
            split_audio_to_timed_chunks,
            audio_path,
            work_dir / "chunks",
        )
        chunk_sizes = [chunk.path.stat().st_size for chunk in chunks]
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
                    "job %s step 4/6 transcribing chunk %s/%s: chunk_bytes=%s provider=%s model=%s",
                    job_id,
                    index,
                    total,
                    progress_data.get("chunk_bytes"),
                    progress_data.get("provider"),
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

        transcription_result = normalize_transcription_result(
            await transcriber.transcribe_chunks_async(chunks, progress_callback=report_progress)
        )

    raw_transcript = transcription_result.raw_transcript.strip() or "No speech was detected."
    refined_transcript = (
        transcription_result.refined_transcript.strip()
        if transcription_result.refined_transcript is not None
        else None
    )
    srt = render_srt(transcription_result.subtitle_cues)
    await status.edit_text("Step 6/6: sending transcript...")
    logger.info(
        "job %s step 6/6 sending transcript: raw_chars=%d refined_chars=%s srt_cues=%d raw_delivery=%s",
        job_id,
        len(raw_transcript),
        len(refined_transcript) if refined_transcript is not None else None,
        len(transcription_result.subtitle_cues),
        "text" if should_send_as_text(raw_transcript) else "document",
    )
    await send_transcript(message, raw_transcript, caption="Transcription")
    if refined_transcript:
        await send_transcript(
            message,
            refined_transcript,
            filename="refined_transcript.txt",
            caption="Refined transcript",
        )
    if srt:
        await send_srt(message, srt)
        await status.edit_text("Transcript ready.")
    else:
        await status.edit_text("Transcript ready. SRT unavailable for this provider/model.")
    logger.info(
        "job %s completed: duration_ms=%d raw_chars=%d refined_chars=%s srt_cues=%d",
        job_id,
        elapsed_ms(job_started),
        len(raw_transcript),
        len(refined_transcript) if refined_transcript is not None else None,
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
    if should_send_as_text(transcript):
        for chunk in split_text_for_telegram(transcript):
            await reply_to_source(message, chunk)
        return

    transcript_file = BytesIO(transcript.encode("utf-8"))
    transcript_file.name = filename
    transcript_file.seek(0)
    await message.reply_document(
        document=InputFile(transcript_file, filename=filename),
        caption=caption,
        **source_reply_kwargs(message),
    )


async def send_srt(message: Message, srt: str) -> None:
    srt_file = BytesIO(srt.encode("utf-8"))
    srt_file.name = "transcript.srt"
    srt_file.seek(0)
    await message.reply_document(
        document=InputFile(srt_file, filename="transcript.srt"),
        caption="SRT subtitles",
        **source_reply_kwargs(message),
    )


def normalize_transcription_result(result: object) -> TranscriptionResult:
    if isinstance(result, TranscriptionResult):
        return result
    if isinstance(result, str):
        return TranscriptionResult(raw_transcript=result)
    raise TranscriptionError("Transcriber returned an unsupported result.")


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


def get_runtime_audio_tempo(context: ContextTypes.DEFAULT_TYPE, settings: Settings) -> float:
    candidate = context.bot_data.get("audio_tempo", settings.audio_tempo)
    try:
        return parse_audio_tempo(str(candidate))
    except ConfigError:
        return settings.audio_tempo


def get_transcribe_model_name(settings: Settings) -> str:
    return settings.deepgram_transcribe_model


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
