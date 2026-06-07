from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path


DEFAULT_AUDIO_BITRATE_KBPS = 64
DEFAULT_AUDIO_TEMPO = 1.0


class FfmpegError(RuntimeError):
    """Raised when ffmpeg is unavailable or fails."""


def ensure_ffmpeg_available(executable: str = "ffmpeg") -> None:
    if shutil.which(executable) is None:
        raise FfmpegError(
            "ffmpeg was not found. Install it locally with `brew install ffmpeg` "
            "or run the Docker image, which includes ffmpeg."
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
    chunk_seconds: int,
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


def run_command(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown ffmpeg error"
        raise FfmpegError(detail)


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


def prepare_audio_chunks(
    video_path: Path,
    work_dir: Path,
    max_audio_bytes: int,
    *,
    executable: str = "ffmpeg",
    audio_bitrate_kbps: int = DEFAULT_AUDIO_BITRATE_KBPS,
    audio_tempo: float = DEFAULT_AUDIO_TEMPO,
) -> list[Path]:
    audio_path = work_dir / "audio.mp3"
    extract_audio(
        video_path,
        audio_path,
        executable=executable,
        audio_bitrate_kbps=audio_bitrate_kbps,
        audio_tempo=audio_tempo,
    )
    if audio_path.stat().st_size <= max_audio_bytes:
        return [audio_path]
    return split_audio_to_chunks(
        audio_path,
        work_dir / "chunks",
        max_audio_bytes,
        executable=executable,
        audio_bitrate_kbps=audio_bitrate_kbps,
    )


async def prepare_audio_chunks_async(
    video_path: Path,
    work_dir: Path,
    max_audio_bytes: int,
    *,
    executable: str = "ffmpeg",
    audio_bitrate_kbps: int = DEFAULT_AUDIO_BITRATE_KBPS,
    audio_tempo: float = DEFAULT_AUDIO_TEMPO,
) -> list[Path]:
    return await asyncio.to_thread(
        prepare_audio_chunks,
        video_path,
        work_dir,
        max_audio_bytes,
        executable=executable,
        audio_bitrate_kbps=audio_bitrate_kbps,
        audio_tempo=audio_tempo,
    )


def split_audio_to_chunks(
    audio_path: Path,
    chunks_dir: Path,
    max_audio_bytes: int,
    *,
    executable: str = "ffmpeg",
    audio_bitrate_kbps: int = DEFAULT_AUDIO_BITRATE_KBPS,
) -> list[Path]:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    bytes_per_second = audio_bitrate_kbps * 1000 / 8

    for safety_factor in (0.90, 0.75, 0.60):
        clear_chunk_dir(chunks_dir)
        chunk_seconds = max(30, int((max_audio_bytes / bytes_per_second) * safety_factor))
        run_command(
            build_split_audio_command(
                audio_path,
                chunks_dir / "chunk_%03d.mp3",
                chunk_seconds,
                executable=executable,
            )
        )
        chunks = sorted(chunks_dir.glob("chunk_*.mp3"))
        if chunks and all(chunk.stat().st_size <= max_audio_bytes for chunk in chunks):
            return chunks

    raise FfmpegError("Unable to split audio into chunks below the configured upload limit.")


def clear_chunk_dir(chunks_dir: Path) -> None:
    for chunk in chunks_dir.glob("chunk_*.mp3"):
        chunk.unlink()
