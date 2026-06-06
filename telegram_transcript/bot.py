from __future__ import annotations

import asyncio
import logging
import tempfile
from io import BytesIO
from pathlib import Path

from telegram import Document, InputFile, Message, Update, Video
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from telegram_transcript.config import ConfigError, Settings, load_settings
from telegram_transcript.ffmpeg import FfmpegError, ensure_ffmpeg_available, prepare_audio_chunks_async
from telegram_transcript.telegram_utils import should_send_as_text, split_text_for_telegram
from telegram_transcript.transcriber import OpenAITranscriber, TranscriptionError

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def create_application(settings: Settings | None = None) -> Application:
    settings = settings or load_settings()
    transcriber = OpenAITranscriber(
        api_key=settings.openai_api_key,
        model=settings.openai_transcribe_model,
        refinement_model=settings.openai_refine_model,
    )

    app = Application.builder().token(settings.telegram_bot_token).build()
    app.bot_data["settings"] = settings
    app.bot_data["transcriber"] = transcriber
    app.bot_data["job_semaphore"] = asyncio.Semaphore(settings.max_concurrent_jobs)

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(video_message_filter(), handle_video_upload))
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_non_video))
    return app


def video_message_filter() -> filters.BaseFilter:
    document_video_filter = getattr(filters.Document, "VIDEO", filters.Document.ALL)
    return filters.VIDEO | document_video_filter


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await help_command(update, context)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    settings: Settings = context.bot_data["settings"]
    await message.reply_text(
        "Send me a video file and I will reply with its transcript.\n\n"
        f"Maximum configured video size: {settings.max_video_mb:g} MB."
    )


async def handle_non_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    await message.reply_text("Please send a video file to transcribe.")


async def handle_video_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return

    settings: Settings = context.bot_data["settings"]
    user_id = update.effective_user.id if update.effective_user else None
    if not is_authorized(settings, user_id):
        await message.reply_text("Sorry, this bot is not enabled for your Telegram account.")
        return

    attachment = get_video_attachment(message)
    if attachment is None:
        await message.reply_text("Please send a video file to transcribe.")
        return

    file_size = getattr(attachment, "file_size", None)
    if file_size and file_size > settings.max_video_bytes:
        await message.reply_text(f"This video is larger than the configured {settings.max_video_mb:g} MB limit.")
        return

    semaphore: asyncio.Semaphore = context.bot_data["job_semaphore"]
    status = await message.reply_text("Video received. Waiting for an available transcription slot...")

    async with semaphore:
        try:
            await process_video_message(message, attachment, settings, context, status)
        except (FfmpegError, TranscriptionError) as exc:
            logger.exception("Video transcription failed")
            await status.edit_text(f"Transcription failed: {exc}")
        except Exception:
            logger.exception("Unexpected video transcription failure")
            await status.edit_text("Transcription failed because of an unexpected error.")


async def process_video_message(
    message: Message,
    attachment: Video | Document,
    settings: Settings,
    context: ContextTypes.DEFAULT_TYPE,
    status: Message,
) -> None:
    transcriber: OpenAITranscriber = context.bot_data["transcriber"]

    with tempfile.TemporaryDirectory(prefix="telegram-transcript-") as tmp:
        work_dir = Path(tmp)
        video_path = work_dir / f"video{get_attachment_suffix(attachment)}"

        await status.edit_text("Downloading video...")
        telegram_file = await attachment.get_file()
        await telegram_file.download_to_drive(custom_path=str(video_path))

        await status.edit_text("Extracting audio...")
        chunks = await prepare_audio_chunks_async(
            video_path,
            work_dir,
            settings.max_openai_audio_bytes,
        )

        chunk_note = f" ({len(chunks)} chunks)" if len(chunks) > 1 else ""
        await status.edit_text(f"Transcribing audio{chunk_note}...")
        transcript = await transcriber.transcribe_chunks_async(chunks)

    transcript = transcript.strip() or "No speech was detected."
    await status.edit_text("Transcript ready.")
    await send_transcript(message, transcript)


async def send_transcript(message: Message, transcript: str) -> None:
    if should_send_as_text(transcript):
        for chunk in split_text_for_telegram(transcript):
            await message.reply_text(chunk)
        return

    transcript_file = BytesIO(transcript.encode("utf-8"))
    transcript_file.name = "transcript.txt"
    transcript_file.seek(0)
    await message.reply_document(
        document=InputFile(transcript_file, filename="transcript.txt"),
        caption="Transcript",
    )


def is_authorized(settings: Settings, user_id: int | None) -> bool:
    if not settings.allowed_telegram_user_ids:
        return True
    return user_id in settings.allowed_telegram_user_ids


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
