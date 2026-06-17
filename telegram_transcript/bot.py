from __future__ import annotations

import asyncio
import logging
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

from telethon import TelegramClient, events
from telethon.sessions import MemorySession
from telethon.tl.custom.message import Message
from telethon.tl.types import DocumentAttributeFilename

from telegram_transcript.config import ConfigError, Settings, load_settings, parse_audio_tempo
from telegram_transcript.ffmpeg import FfmpegError, ensure_ffmpeg_available, extract_audio, split_audio_to_chunks
from telegram_transcript.telegram_utils import should_send_as_text, split_text_for_telegram
from telegram_transcript.transcriber import OpenAITranscriber, TranscriptionError

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
DEFAULT_NOISE_REDUCTION_MODE = "default"
EXTRA_NOISE_REDUCTION_MODE = "extra"
NOISE_REDUCTION_FILTERS = {
    DEFAULT_NOISE_REDUCTION_MODE: "anlmdn",
    EXTRA_NOISE_REDUCTION_MODE: "highpass=f=80,afftdn=nr=15,loudnorm",
}

TopicKey = tuple[int | None, int | None]


@dataclass
class BotState:
    settings: Settings
    transcriber: OpenAITranscriber
    job_semaphore: asyncio.Semaphore
    audio_tempo: float
    pending_noise_reductions: dict[TopicKey, str] = field(default_factory=dict)


def create_bot_state(settings: Settings | None = None) -> BotState:
    settings = settings or load_settings()
    transcriber = OpenAITranscriber(
        api_key=settings.openai_api_key,
        model=settings.openai_transcribe_model,
        refinement_model=settings.openai_refine_model,
        refine=settings.refine,
    )
    return BotState(
        settings=settings,
        transcriber=transcriber,
        job_semaphore=asyncio.Semaphore(settings.max_concurrent_jobs),
        audio_tempo=settings.audio_tempo,
    )


def create_client(settings: Settings) -> TelegramClient:
    return TelegramClient(MemorySession(), settings.telegram_api_id, settings.telegram_api_hash)


def register_handlers(client: TelegramClient, state: BotState) -> None:
    async def handle_event(event: events.NewMessage.Event) -> None:
        await handle_new_message(event, state)

    client.add_event_handler(handle_event, events.NewMessage(incoming=True))


async def run_bot(settings: Settings | None = None) -> None:
    settings = settings or load_settings()
    state = create_bot_state(settings)
    client = create_client(settings)
    register_handlers(client, state)
    await client.start(bot_token=settings.telegram_bot_token)
    logger.info("telegram bot started with Telethon")
    try:
        await client.run_until_disconnected()
    finally:
        await client.disconnect()


async def handle_new_message(event: events.NewMessage.Event, state: BotState) -> None:
    message = getattr(event, "message", None)
    if message is None:
        return

    command = parse_command(get_message_text(message))
    if command is not None:
        command_name, args = command
        if command_name == "tempo":
            await handle_tempo_command(message, state, args)
            return
        if command_name == "noise":
            await handle_noise_command(message, state, args)
            return

    if get_video_attachment(message) is not None:
        await handle_video_upload(message, state)
        return

    await handle_non_video(message, state)


async def handle_non_video(message: Message, state: BotState) -> None:
    return


async def handle_tempo_command(message: Message, state: BotState, args: list[str] | None = None) -> None:
    if not is_group_chat(message) or is_broadcast_channel(message):
        return

    if not is_allowed_group_topic(state.settings, message):
        return

    audio_tempo = parse_tempo_command_args(args)
    if audio_tempo is None:
        return

    state.audio_tempo = audio_tempo
    await reply_to_source(message, f"Tempo set to {audio_tempo:g}x.")


async def handle_noise_command(message: Message, state: BotState, args: list[str] | None = None) -> None:
    if is_broadcast_channel(message):
        return

    if not is_allowed_group_topic(state.settings, message):
        return

    if not is_authorized(state.settings, get_sender_id(message)):
        return

    noise_reduction_mode = parse_noise_command_args(args)
    if noise_reduction_mode is None:
        return

    set_pending_noise_reduction_mode(state, message, noise_reduction_mode)
    await reply_to_source(message, noise_reduction_confirmation_text(noise_reduction_mode))


async def handle_video_upload(message: Message, state: BotState) -> None:
    if is_broadcast_channel(message):
        return

    settings = state.settings
    if not is_allowed_group_topic(settings, message):
        return

    if not is_authorized(settings, get_sender_id(message)):
        await reply_to_source(message, "Sorry, this bot is not enabled for your Telegram account.")
        return

    attachment = get_video_attachment(message)
    if attachment is None:
        return

    file_size = get_attachment_file_size(attachment)
    if file_size is not None and file_size > settings.max_video_bytes:
        return

    audio_tempo = get_runtime_audio_tempo(state, settings)
    noise_reduction_mode = consume_pending_noise_reduction_mode(state, message)
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

    status_ref: dict[str, Any | None] = {"message": None}
    async with state.job_semaphore:
        try:
            logger.info("job %s started", job_id)
            await process_video_message(
                message,
                attachment,
                settings,
                state,
                status_ref,
                job_id,
                audio_tempo,
                noise_reduction_mode,
            )
        except (FfmpegError, TranscriptionError) as exc:
            logger.exception("job %s video transcription failed", job_id)
            status = status_ref["message"]
            if status is not None:
                await edit_status(status, f"Transcription failed: {exc}")
        except Exception:
            logger.exception("job %s unexpected video transcription failure", job_id)
            status = status_ref["message"]
            if status is not None:
                await edit_status(status, "Transcription failed because of an unexpected error.")


async def process_video_message(
    message: Message,
    attachment: Message,
    settings: Settings,
    state: BotState,
    status_ref: dict[str, Any | None],
    job_id: str,
    audio_tempo: float,
    noise_reduction_mode: str | None = None,
) -> None:
    transcriber = state.transcriber
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
            get_attachment_file_size(attachment),
        )
        await download_attachment(attachment, video_path)
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

        await edit_status(status, extracting_audio_status_text(audio_tempo, noise_reduction_mode))
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

        await edit_status(status, "Step 3/6: preparing audio chunks...")
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
                await edit_status(status, f"Step 4/6: transcribing chunk {index}/{total}...")
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
                await edit_status(status, "Step 5/6: refining transcript...")
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
    await edit_status(status, "Step 6/6: sending transcript...")
    logger.info(
        "job %s step 6/6 sending transcript: output_chars=%d delivery=%s",
        job_id,
        len(transcript),
        "text" if should_send_as_text(transcript) else "document",
    )
    await send_transcript(message, transcript)
    await edit_status(status, "Transcript ready.")
    logger.info(
        "job %s completed: duration_ms=%d output_chars=%d",
        job_id,
        elapsed_ms(job_started),
        len(transcript),
    )


async def download_attachment(attachment: Message, video_path: Path) -> None:
    await attachment.download_media(file=str(video_path))
    if not video_path.exists():
        raise RuntimeError("Telegram media download did not produce a file.")


async def send_transcript(message: Message, transcript: str) -> None:
    if should_send_as_text(transcript):
        for chunk in split_text_for_telegram(transcript):
            await reply_to_source(message, chunk)
        return

    transcript_file = BytesIO(transcript.encode("utf-8"))
    transcript_file.name = "transcript.txt"
    transcript_file.seek(0)
    client = get_message_client(message)
    if client is None:
        raise RuntimeError("Cannot send transcript document without a Telethon client.")
    await client.send_file(
        await get_message_entity(message),
        transcript_file,
        caption="Transcript",
        force_document=True,
        reply_to=get_message_id(message),
        attributes=[DocumentAttributeFilename("transcript.txt")],
    )


async def reply_to_source(message: Message, text: str) -> Any:
    return await message.reply(text)


async def edit_status(status: Any, text: str) -> None:
    edit = getattr(status, "edit", None)
    if edit is None:
        edit = getattr(status, "edit_text", None)
    if edit is None:
        raise RuntimeError("Status message cannot be edited.")
    await edit(text)


async def get_message_entity(message: Message) -> Any:
    get_input_chat = getattr(message, "get_input_chat", None)
    if get_input_chat is not None:
        return await get_input_chat()
    return getattr(message, "chat_id", None)


def get_message_client(message: Message) -> Any | None:
    return getattr(message, "client", None)


def get_message_id(message: Message) -> int | None:
    return getattr(message, "id", getattr(message, "message_id", None))


def get_message_text(message: Message) -> str:
    text = getattr(message, "raw_text", None)
    if text is None:
        text = getattr(message, "text", None)
    return text if isinstance(text, str) else ""


def parse_command(text: str) -> tuple[str, list[str]] | None:
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    command_token, *args = stripped.split()
    command = command_token[1:].split("@", 1)[0].casefold()
    return (command, args) if command else None


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


def set_pending_noise_reduction_mode(state: BotState, message: Message, noise_reduction_mode: str) -> None:
    state.pending_noise_reductions[get_noise_reduction_thread_key(message)] = noise_reduction_mode


def consume_pending_noise_reduction_mode(state: BotState, message: Message) -> str | None:
    noise_reduction_mode = state.pending_noise_reductions.pop(get_noise_reduction_thread_key(message), None)
    return noise_reduction_mode if noise_reduction_mode in NOISE_REDUCTION_FILTERS else None


def get_noise_reduction_thread_key(message: Message) -> TopicKey:
    return (get_chat_id(message), get_message_topic_id(message))


def get_message_topic_id(message: Message) -> int | None:
    reply_to = getattr(message, "reply_to", None)
    if reply_to is None:
        return None

    top_id = getattr(reply_to, "reply_to_top_id", None)
    reply_msg_id = getattr(reply_to, "reply_to_msg_id", None)
    if getattr(reply_to, "forum_topic", False):
        return top_id or reply_msg_id
    return top_id


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


def get_runtime_audio_tempo(state: BotState, settings: Settings) -> float:
    try:
        return parse_audio_tempo(str(state.audio_tempo))
    except ConfigError:
        return settings.audio_tempo


def is_authorized(settings: Settings, user_id: int | None) -> bool:
    if not settings.allowed_telegram_user_ids:
        return True
    return user_id in settings.allowed_telegram_user_ids


def is_allowed_group_topic(settings: Settings, message: Message) -> bool:
    if settings.allowed_telegram_topic_id is None:
        return True
    if not is_group_chat(message):
        return True
    return get_message_topic_id(message) == settings.allowed_telegram_topic_id


def is_group_chat(message: Message) -> bool:
    return bool(getattr(message, "is_group", False))


def is_broadcast_channel(message: Message) -> bool:
    return bool(getattr(message, "is_channel", False)) and not is_group_chat(message)


def get_sender_id(message: Message) -> int | None:
    return getattr(message, "sender_id", None)


def get_chat_id(message: Message) -> int | None:
    return getattr(message, "chat_id", None)


def get_video_attachment(message: Message) -> Message | None:
    return message if is_video_message(message) else None


def is_video_message(message: Message) -> bool:
    if getattr(message, "video", None) is not None:
        return True
    file = getattr(message, "file", None)
    return file is not None and is_video_document(file)


def is_video_document(document: Any) -> bool:
    mime_type = get_file_mime_type(document)
    if mime_type.startswith("video/"):
        return True
    file_name = get_file_name(document)
    return Path(file_name).suffix.lower() in VIDEO_EXTENSIONS


def get_attachment_suffix(attachment: Message) -> str:
    file_name = get_file_name(getattr(attachment, "file", None)) or get_file_name(attachment)
    suffix = Path(file_name).suffix.lower() if file_name else ""
    return suffix if suffix in VIDEO_EXTENSIONS else ".mp4"


def get_attachment_file_size(attachment: Message) -> int | None:
    file = getattr(attachment, "file", None)
    size = getattr(file, "size", None) if file is not None else None
    if size is None:
        video = getattr(attachment, "video", None)
        size = getattr(video, "size", None) if video is not None else None
    if size is None:
        size = getattr(attachment, "file_size", None)
    return size if isinstance(size, int) else None


def get_file_name(file: Any) -> str:
    if file is None:
        return ""
    for attr in ("name", "file_name"):
        value = getattr(file, attr, None)
        if isinstance(value, str):
            return value
    return ""


def get_file_mime_type(file: Any) -> str:
    value = getattr(file, "mime_type", None)
    return value if isinstance(value, str) else ""


def elapsed_ms(started_at: float) -> int:
    return int((time.monotonic() - started_at) * 1000)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        settings = load_settings()
        ensure_ffmpeg_available()
        asyncio.run(run_bot(settings))
    except (ConfigError, FfmpegError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
