from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from telegram_transcript.models import AudioChunk


DEFAULT_AUDIO_BITRATE_KBPS = 64
DEFAULT_AUDIO_TEMPO = 1.0
TRANSCRIPTION_CHUNK_SECONDS = 420
MAX_AUDIO_CHUNK_BYTES = 13 * 1024 * 1024
CHUNK_OVERLAP_SECONDS = 1.5
SILENCE_SEARCH_SECONDS = 30.0
SILENCE_NOISE_DB = -35
SILENCE_MIN_DURATION_SECONDS = 0.4
SILENCE_END_RE = re.compile(r"silence_end:\s*(?P<seconds>\d+(?:\.\d+)?)")


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
    del audio_bitrate_kbps
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
        "flac",
        str(output_path),
    ]


def build_split_audio_command(
    audio_path: Path,
    output_pattern: Path,
    chunk_seconds: int = TRANSCRIPTION_CHUNK_SECONDS,
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


def build_detect_silence_command(
    audio_path: Path,
    *,
    executable: str = "ffmpeg",
) -> list[str]:
    return [
        executable,
        "-hide_banner",
        "-i",
        str(audio_path),
        "-af",
        f"silencedetect=noise={SILENCE_NOISE_DB}dB:d={SILENCE_MIN_DURATION_SECONDS:g}",
        "-f",
        "null",
        "-",
    ]


def build_extract_audio_window_command(
    audio_path: Path,
    output_path: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    executable: str = "ffmpeg",
) -> list[str]:
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise ValueError("Audio window must have a non-negative start before its end.")
    return [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(audio_path),
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{end_seconds - start_seconds:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-codec:a",
        "flac",
        str(output_path),
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


def run_combined_capture_command(command: list[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown ffmpeg error"
        raise FfmpegError(detail)
    return "\n".join(part for part in (completed.stdout.strip(), completed.stderr.strip()) if part)


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
    audio_path = work_dir / "audio.flac"
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
        time_scale=audio_tempo,
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
    chunk_seconds: int = TRANSCRIPTION_CHUNK_SECONDS,
    overlap_seconds: float = CHUNK_OVERLAP_SECONDS,
    max_chunk_bytes: int = MAX_AUDIO_CHUNK_BYTES,
    time_scale: float = 1.0,
) -> list[AudioChunk]:
    if time_scale <= 0:
        raise ValueError("time_scale must be greater than zero.")
    duration_seconds = probe_audio_duration_seconds(audio_path, executable=probe_executable)
    if duration_seconds <= chunk_seconds and audio_path.stat().st_size <= max_chunk_bytes:
        return [
            AudioChunk(
                path=audio_path,
                duration_seconds=duration_seconds,
                time_scale=time_scale,
                owned_start_seconds=0.0,
                owned_end_seconds=duration_seconds,
            )
        ]

    if max_chunk_bytes <= 0:
        raise ValueError("max_chunk_bytes must be greater than zero.")

    chunks_dir.mkdir(parents=True, exist_ok=True)
    clear_chunk_dir(chunks_dir)
    silence_output = run_combined_capture_command(
        build_detect_silence_command(audio_path, executable=executable)
    )
    ranges = choose_silence_aware_chunk_ranges(
        duration_seconds,
        parse_silence_end_seconds(silence_output),
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
    )
    chunks: list[AudioChunk] = []

    def extract_bounded_range(start_seconds: float, end_seconds: float) -> None:
        chunk_path = chunks_dir / f"chunk_{len(chunks):03d}.flac"
        run_command(
            build_extract_audio_window_command(
                audio_path,
                chunk_path,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                executable=executable,
            )
        )
        if not chunk_path.exists() or chunk_path.stat().st_size == 0:
            raise FfmpegError(f"Unable to create audio chunk {len(chunks) + 1}.")
        if chunk_path.stat().st_size <= max_chunk_bytes:
            chunks.append(
                AudioChunk(
                    path=chunk_path,
                    start_seconds=start_seconds,
                    duration_seconds=end_seconds - start_seconds,
                    time_scale=time_scale,
                )
            )
            return

        chunk_path.unlink()
        midpoint = choose_size_split_boundary(start_seconds, end_seconds, parse_silence_end_seconds(silence_output))
        if midpoint - start_seconds <= overlap_seconds or end_seconds - midpoint <= overlap_seconds:
            raise FfmpegError("Unable to split audio below the provider upload limit.")
        extract_bounded_range(start_seconds, min(end_seconds, midpoint + overlap_seconds))
        extract_bounded_range(max(start_seconds, midpoint - overlap_seconds), end_seconds)

    for start_seconds, end_seconds in ranges:
        extract_bounded_range(start_seconds, end_seconds)
    ownership_boundaries = [0.0]
    for current, following in zip(chunks, chunks[1:]):
        current_end = current.start_seconds + (current.duration_seconds or 0.0)
        ownership_boundaries.append((current_end + following.start_seconds) / 2)
    ownership_boundaries.append(duration_seconds)
    return [
        AudioChunk(
            path=chunk.path,
            start_seconds=chunk.start_seconds,
            duration_seconds=chunk.duration_seconds,
            time_scale=chunk.time_scale,
            owned_start_seconds=ownership_boundaries[index],
            owned_end_seconds=ownership_boundaries[index + 1],
        )
        for index, chunk in enumerate(chunks)
    ]


def choose_size_split_boundary(
    start_seconds: float,
    end_seconds: float,
    silence_ends: Sequence[float],
) -> float:
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise ValueError("Audio range must have a non-negative start before its end.")

    midpoint = start_seconds + (end_seconds - start_seconds) / 2
    minimum = start_seconds + (end_seconds - start_seconds) * 0.25
    maximum = start_seconds + (end_seconds - start_seconds) * 0.75
    candidates = [value for value in silence_ends if minimum <= value <= maximum]
    return min(candidates, key=lambda value: abs(value - midpoint)) if candidates else midpoint


def parse_silence_end_seconds(output: str) -> tuple[float, ...]:
    return tuple(float(match.group("seconds")) for match in SILENCE_END_RE.finditer(output))


def choose_silence_aware_chunk_ranges(
    duration_seconds: float,
    silence_ends: Sequence[float],
    *,
    chunk_seconds: int = TRANSCRIPTION_CHUNK_SECONDS,
    overlap_seconds: float = CHUNK_OVERLAP_SECONDS,
) -> tuple[tuple[float, float], ...]:
    if duration_seconds <= 0 or chunk_seconds <= 0 or overlap_seconds < 0:
        raise ValueError("Audio duration and chunk size must be positive and overlap cannot be negative.")
    maximum_core_seconds = chunk_seconds - 2 * overlap_seconds
    if maximum_core_seconds <= 0:
        raise ValueError("Chunk size must be greater than twice the overlap.")

    boundaries = [0.0]
    while True:
        remaining_seconds = duration_seconds - boundaries[-1]
        maximum_remaining_seconds = chunk_seconds if len(boundaries) == 1 else chunk_seconds - overlap_seconds
        if remaining_seconds <= maximum_remaining_seconds:
            break

        if remaining_seconds <= 2 * (chunk_seconds - overlap_seconds):
            target = boundaries[-1] + remaining_seconds / 2
        else:
            target = boundaries[-1] + maximum_core_seconds
        candidates = [
            value
            for value in silence_ends
            if boundaries[-1] + 1.0 < value < duration_seconds
            and abs(value - target) <= SILENCE_SEARCH_SECONDS
            and value - boundaries[-1] <= maximum_core_seconds
        ]
        boundary = min(candidates, key=lambda value: abs(value - target)) if candidates else target
        boundaries.append(boundary)
    boundaries.append(duration_seconds)

    ranges = []
    for index, (boundary_start, boundary_end) in enumerate(zip(boundaries, boundaries[1:])):
        start = boundary_start if index == 0 else max(0.0, boundary_start - overlap_seconds)
        end = boundary_end if index == len(boundaries) - 2 else min(duration_seconds, boundary_end + overlap_seconds)
        ranges.append((start, end))
    return tuple(ranges)


def extract_audio_window(
    audio_path: Path,
    output_path: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    executable: str = "ffmpeg",
) -> Path:
    run_command(
        build_extract_audio_window_command(
            audio_path,
            output_path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            executable=executable,
        )
    )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise FfmpegError("ffmpeg did not produce an audio window.")
    return output_path


def clear_chunk_dir(chunks_dir: Path) -> None:
    for pattern in ("chunk_*", "candidate_*"):
        for chunk in chunks_dir.glob(pattern):
            chunk.unlink()
