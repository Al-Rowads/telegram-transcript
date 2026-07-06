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
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from telegram_transcript.config import ConfigError, Settings, load_settings, parse_audio_tempo
from telegram_transcript.ffmpeg import FfmpegError, ensure_ffmpeg_available, extract_audio, split_audio_to_timed_chunks
from telegram_transcript.models import AudioChunk, TranscriptionResult
from telegram_transcript.telegram_utils import should_send_as_text, split_text_for_telegram
from telegram_transcript.transcriber import (
    DeepgramSpeechToTextProvider,
    OpenAISpeechToTextProvider,
    SpeechTranscriber,
    TranscriptRefiner,
    TranscriptionError,
    render_srt,
)

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
GROUP_CHAT_TYPES = {ChatType.GROUP, ChatType.SUPERGROUP}


def create_application(settings: Settings | None = None) -> Application:
    settings = settings or load_settings()
    transcriber = create_transcriber(settings)

    app = Application.builder().token(settings.telegram_bot_token).build()
    app.bot_data["settings"] = settings
    app.bot_data["transcriber"] = transcriber
    app.bot_data["job_semaphore"] = asyncio.Semaphore(settings.max_concurrent_jobs)
    app.bot_data["audio_tempo"] = settings.audio_tempo

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("tempo", handle_tempo_command))
    app.add_handler(MessageHandler(video_message_filter(), handle_video_upload))
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_non_video))
    return app


def create_transcriber(settings: Settings) -> SpeechTranscriber:
    if settings.speech_to_text_provider == "deepgram":
        speech_to_text_provider = DeepgramSpeechToTextProvider(
            api_key=settings.deepgram_api_key,
            model=settings.deepgram_transcribe_model,
            language=settings.deepgram_language,
        )
    elif settings.speech_to_text_provider == "openai":
        speech_to_text_provider = OpenAISpeechToTextProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_transcribe_model,
        )
    else:
        raise ConfigError(f"Unsupported speech-to-text provider: {settings.speech_to_text_provider}")

    refiner = (
        TranscriptRefiner(
            api_key=settings.openai_api_key,
            model=settings.openai_refine_model,
        )
        if settings.refine
        else None
    )
    return SpeechTranscriber(speech_to_text_provider=speech_to_text_provider, refiner=refiner)


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

    audio_tempo = parse_tempo_command_args(getattr(context, "args", None))
    if audio_tempo is None:
        return

    context.bot_data["audio_tempo"] = audio_tempo
    await reply_to_source(message, f"Tempo set to {audio_tempo:g}x.")


async def handle_video_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

    attachment = get_video_attachment(message)
    if attachment is None:
        return

    file_size = getattr(attachment, "file_size", None)
    if file_size is not None and file_size > settings.max_video_bytes:
        return

    audio_tempo = get_runtime_audio_tempo(context, settings)
    job_id = uuid.uuid4().hex[:8]
    logger.info(
        "job %s queued: suffix=%s telegram_file_size=%s audio_tempo=%g stt_provider=%s transcribe_model=%s refine=%s refine_model=%s",
        job_id,
        get_attachment_suffix(attachment),
        file_size,
        audio_tempo,
        settings.speech_to_text_provider,
        get_transcribe_model_name(settings),
        settings.refine,
        settings.openai_refine_model,
    )

    semaphore: asyncio.Semaphore = context.bot_data["job_semaphore"]
    status_ref: dict[str, Message | None] = {"message": None}
    async with semaphore:
        try:
            logger.info("job %s started", job_id)
            await process_video_message(message, attachment, settings, context, status_ref, job_id, audio_tempo)
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
) -> None:
    transcriber: SpeechTranscriber = context.bot_data["transcriber"]
    job_started = time.monotonic()

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
        telegram_file = await attachment.get_file()
        await telegram_file.download_to_drive(custom_path=str(video_path))
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

        await status.edit_text(f"Step 2/6: extracting MP3 audio at {audio_tempo:g}x...")
        step_started = time.monotonic()
        logger.info(
            "job %s step 2/6 extracting MP3 audio: video_bytes=%d audio_tempo=%g",
            job_id,
            video_bytes,
            audio_tempo,
        )
        await asyncio.to_thread(
            extract_audio,
            video_path,
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
            "job %s step 3/6 preparing chunks: audio_bytes=%d max_chunk_bytes=%d",
            job_id,
            audio_bytes,
            settings.max_audio_bytes,
        )
        if audio_bytes <= settings.max_audio_bytes:
            chunks = [AudioChunk(path=audio_path)]
        else:
            chunks = await asyncio.to_thread(
                split_audio_to_timed_chunks,
                audio_path,
                work_dir / "chunks",
                settings.max_audio_bytes,
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
    if settings.speech_to_text_provider == "deepgram":
        return settings.deepgram_transcribe_model
    return settings.openai_transcribe_model


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
