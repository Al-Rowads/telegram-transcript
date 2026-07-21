from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from telegram_transcript import ffmpeg


def test_build_extract_audio_command() -> None:
    command = ffmpeg.build_extract_audio_command(Path("input.mp4"), Path("output.flac"))

    assert command[:5] == ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    assert "-vn" in command
    assert ["-ac", "1"] == command[command.index("-ac") : command.index("-ac") + 2]
    assert ["-ar", "16000"] == command[command.index("-ar") : command.index("-ar") + 2]
    assert ["-filter:a", "atempo=1"] == command[command.index("-filter:a") : command.index("-filter:a") + 2]
    assert ["-codec:a", "flac"] == command[command.index("-codec:a") : command.index("-codec:a") + 2]
    assert command[-1] == "output.flac"


def test_build_extract_audio_command_accepts_custom_tempo() -> None:
    command = ffmpeg.build_extract_audio_command(
        Path("input.mp4"),
        Path("output.flac"),
        audio_tempo=1.0,
    )

    assert ["-filter:a", "atempo=1"] == command[command.index("-filter:a") : command.index("-filter:a") + 2]


def test_build_probe_audio_duration_command() -> None:
    command = ffmpeg.build_probe_audio_duration_command(Path("audio.mp3"))

    assert command == [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        "audio.mp3",
    ]


def test_probe_audio_duration_rejects_invalid_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ffmpeg, "run_capture_command", lambda _: "not-a-duration")

    with pytest.raises(ffmpeg.FfmpegError, match="invalid audio duration"):
        ffmpeg.probe_audio_duration_seconds(Path("audio.mp3"))


def test_split_audio_to_timed_chunks_returns_original_audio_when_under_420_seconds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"x" * 2000)

    def fail_run_command(command: list[str]) -> None:
        raise AssertionError(f"split command should not run: {command}")

    monkeypatch.setattr(ffmpeg, "probe_audio_duration_seconds", lambda *_, **__: 420)
    monkeypatch.setattr(ffmpeg, "run_command", fail_run_command)

    chunks = ffmpeg.split_audio_to_timed_chunks(audio_path, tmp_path / "chunks")

    assert chunks == [
        ffmpeg.AudioChunk(
            path=audio_path,
            duration_seconds=420,
            owned_start_seconds=0.0,
            owned_end_seconds=420,
        )
    ]


def test_split_audio_to_timed_chunks_uses_silence_boundaries_and_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "audio.flac"
    audio_path.write_bytes(b"x" * 2000)
    chunks_dir = tmp_path / "chunks"

    def fake_run_command(command: list[str]) -> None:
        chunks_dir.mkdir(exist_ok=True)
        Path(command[-1]).write_bytes(b"x" * 500)

    monkeypatch.setattr(ffmpeg, "probe_audio_duration_seconds", lambda *_, **__: 800)
    monkeypatch.setattr(ffmpeg, "run_combined_capture_command", lambda _: "silence_end: 410.0")
    monkeypatch.setattr(ffmpeg, "run_command", fake_run_command)

    chunks = ffmpeg.split_audio_to_timed_chunks(audio_path, chunks_dir)

    assert [chunk.path.name for chunk in chunks] == ["chunk_000.flac", "chunk_001.flac"]
    assert [chunk.start_seconds for chunk in chunks] == [0, 408.5]


def test_chunk_ranges_avoid_a_tiny_tail_just_above_the_limit() -> None:
    ranges = ffmpeg.choose_silence_aware_chunk_ranges(420.1, ())

    assert ranges == ((0.0, 211.55), (208.55, 420.1))


def test_one_hour_audio_is_split_into_nine_bounded_ranges() -> None:
    ranges = ffmpeg.choose_silence_aware_chunk_ranges(3600.0, ())

    assert len(ranges) == 9
    assert ranges[0][0] == 0.0
    assert ranges[-1][1] == 3600.0
    assert all(end - start <= 420.0 for start, end in ranges)


def test_split_audio_recursively_splits_chunks_over_byte_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "audio.flac"
    audio_path.write_bytes(b"x" * 2000)
    chunks_dir = tmp_path / "chunks"

    def fake_run_command(command: list[str]) -> None:
        chunks_dir.mkdir(exist_ok=True)
        duration = float(command[command.index("-t") + 1])
        Path(command[-1]).write_bytes(b"x" * max(1, int(duration)))

    monkeypatch.setattr(ffmpeg, "probe_audio_duration_seconds", lambda *_, **__: 300.0)
    monkeypatch.setattr(ffmpeg, "run_combined_capture_command", lambda _: "")
    monkeypatch.setattr(ffmpeg, "run_command", fake_run_command)

    chunks = ffmpeg.split_audio_to_timed_chunks(
        audio_path,
        chunks_dir,
        max_chunk_bytes=100,
    )

    assert len(chunks) == 4
    assert [chunk.path.name for chunk in chunks] == [
        "chunk_000.flac",
        "chunk_001.flac",
        "chunk_002.flac",
        "chunk_003.flac",
    ]
    assert all(chunk.path.stat().st_size <= 100 for chunk in chunks)
    assert [chunk.start_seconds for chunk in chunks] == sorted(chunk.start_seconds for chunk in chunks)


def test_prepare_audio_chunks_returns_single_audio_when_under_420_seconds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")

    def fake_extract_audio(video: Path, audio: Path, **_: object) -> Path:
        audio.write_bytes(b"audio")
        return audio

    monkeypatch.setattr(ffmpeg, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(ffmpeg, "probe_audio_duration_seconds", lambda *_, **__: 42)

    chunks = ffmpeg.prepare_audio_chunks(video_path, tmp_path)

    assert chunks == [
        ffmpeg.AudioChunk(
            path=tmp_path / "audio.flac",
            duration_seconds=42,
            owned_start_seconds=0.0,
            owned_end_seconds=42,
        )
    ]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_ffmpeg_extracts_audio_from_generated_video(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    audio_path = tmp_path / "audio.flac"

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=1",
            "-shortest",
            str(video_path),
        ],
        check=True,
    )

    ffmpeg.extract_audio(video_path, audio_path)

    assert audio_path.exists()
    assert audio_path.stat().st_size > 0
