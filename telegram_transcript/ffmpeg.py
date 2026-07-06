from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

from telegram_transcript.models import AudioChunk


DEFAULT_AUDIO_BITRATE_KBPS = 64
DEFAULT_AUDIO_TEMPO = 1.0
DEEPGRAM_CHUNK_SECONDS = 1300


class FfmpegError(RuntimeError):
    """Raised when ffmpeg is unavailable or fails."""


def ensure_ffmpeg_available(executable: str = "ffmpeg", probe_executable: str = "ffprobe") -> None:
    if shutil.which(executable) is None:
        raise FfmpegError(
            "ffmpeg was not found. Install it locally with `brew install ffmpeg` "
            "or run the Docker image, which includes ffmpeg."
        )
    if shutil.which(probe_executable) is None:
        raise FfmpegError(
            "ffprobe was not found. Install ffmpeg locally with `brew install ffmpeg` "
            "or run the Docker image, which includes ffprobe."
        )


def build_extract_audio_command(
    input_path: Path,
    output_path: Path,
    *,
    executable: str = "ffmpeg",
    audio_bitrate_kbps: int = DEFAULT_AUDIO_BITRATE_KBPS,
    audio_tempo: float = DEFAULT_AUDIO_TEMPO,
) -> list[str]:
    return [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-filter:a",
        f"atempo={audio_tempo:g}",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        f"{audio_bitrate_kbps}k",
        str(output_path),
    ]


def build_split_audio_command(
    audio_path: Path,
    output_pattern: Path,
    chunk_seconds: int = DEEPGRAM_CHUNK_SECONDS,
    *,
    executable: str = "ffmpeg",
) -> list[str]:
    return [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(audio_path),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
        "-c",
        "copy",
        str(output_pattern),
    ]


def build_probe_audio_duration_command(
    audio_path: Path,
    *,
    executable: str = "ffprobe",
) -> list[str]:
    return [
        executable,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]


def run_command(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown ffmpeg error"
        raise FfmpegError(detail)


def run_capture_command(command: list[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown ffprobe error"
        raise FfmpegError(detail)
    return completed.stdout.strip()


def extract_audio(
    video_path: Path,
    audio_path: Path,
    *,
    executable: str = "ffmpeg",
    audio_bitrate_kbps: int = DEFAULT_AUDIO_BITRATE_KBPS,
    audio_tempo: float = DEFAULT_AUDIO_TEMPO,
) -> Path:
    run_command(
        build_extract_audio_command(
            video_path,
            audio_path,
            executable=executable,
            audio_bitrate_kbps=audio_bitrate_kbps,
            audio_tempo=audio_tempo,
        )
    )
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise FfmpegError("ffmpeg did not produce an audio file.")
    return audio_path


def probe_audio_duration_seconds(
    audio_path: Path,
    *,
    executable: str = "ffprobe",
) -> float:
    raw_duration = run_capture_command(build_probe_audio_duration_command(audio_path, executable=executable))
    try:
        duration = float(raw_duration)
    except ValueError as exc:
        raise FfmpegError(f"ffprobe returned an invalid audio duration: {raw_duration!r}") from exc
    if duration <= 0:
        raise FfmpegError(f"ffprobe returned a non-positive audio duration: {duration:g}")
    return duration


def prepare_audio_chunks(
    video_path: Path,
    work_dir: Path,
    *,
    executable: str = "ffmpeg",
    probe_executable: str = "ffprobe",
    audio_bitrate_kbps: int = DEFAULT_AUDIO_BITRATE_KBPS,
    audio_tempo: float = DEFAULT_AUDIO_TEMPO,
) -> list[AudioChunk]:
    audio_path = work_dir / "audio.mp3"
    extract_audio(
        video_path,
        audio_path,
        executable=executable,
        audio_bitrate_kbps=audio_bitrate_kbps,
        audio_tempo=audio_tempo,
    )
    return split_audio_to_timed_chunks(
        audio_path,
        work_dir / "chunks",
        executable=executable,
        probe_executable=probe_executable,
    )


async def prepare_audio_chunks_async(
    video_path: Path,
    work_dir: Path,
    *,
    executable: str = "ffmpeg",
    probe_executable: str = "ffprobe",
    audio_bitrate_kbps: int = DEFAULT_AUDIO_BITRATE_KBPS,
    audio_tempo: float = DEFAULT_AUDIO_TEMPO,
) -> list[AudioChunk]:
    return await asyncio.to_thread(
        prepare_audio_chunks,
        video_path,
        work_dir,
        executable=executable,
        probe_executable=probe_executable,
        audio_bitrate_kbps=audio_bitrate_kbps,
        audio_tempo=audio_tempo,
    )


def split_audio_to_timed_chunks(
    audio_path: Path,
    chunks_dir: Path,
    *,
    executable: str = "ffmpeg",
    probe_executable: str = "ffprobe",
    chunk_seconds: int = DEEPGRAM_CHUNK_SECONDS,
) -> list[AudioChunk]:
    duration_seconds = probe_audio_duration_seconds(audio_path, executable=probe_executable)
    if duration_seconds <= chunk_seconds:
        return [AudioChunk(path=audio_path)]

    chunks_dir.mkdir(parents=True, exist_ok=True)
    clear_chunk_dir(chunks_dir)
    run_command(
        build_split_audio_command(
            audio_path,
            chunks_dir / "chunk_%03d.mp3",
            chunk_seconds,
            executable=executable,
        )
    )
    chunks = sorted(chunks_dir.glob("chunk_*.mp3"))
    if not chunks:
        raise FfmpegError(f"Unable to split audio into {chunk_seconds}-second chunks.")

    return [
        AudioChunk(path=chunk, start_seconds=index * chunk_seconds)
        for index, chunk in enumerate(chunks)
    ]


def clear_chunk_dir(chunks_dir: Path) -> None:
    for chunk in chunks_dir.glob("chunk_*.mp3"):
        chunk.unlink()
