from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from telegram_transcript import ffmpeg


def test_build_extract_audio_command() -> None:
    command = ffmpeg.build_extract_audio_command(Path("input.mp4"), Path("output.mp3"))

    assert command[:5] == ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    assert "-vn" in command
    assert ["-ac", "1"] == command[command.index("-ac") : command.index("-ac") + 2]
    assert ["-ar", "16000"] == command[command.index("-ar") : command.index("-ar") + 2]
    assert ["-filter:a", "atempo=1"] == command[command.index("-filter:a") : command.index("-filter:a") + 2]
    assert command[-1] == "output.mp3"


def test_build_extract_audio_command_accepts_custom_tempo() -> None:
    command = ffmpeg.build_extract_audio_command(
        Path("input.mp4"),
        Path("output.mp3"),
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


def test_split_audio_to_timed_chunks_returns_original_audio_when_under_1300_seconds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"x" * 2000)

    def fail_run_command(command: list[str]) -> None:
        raise AssertionError(f"split command should not run: {command}")

    monkeypatch.setattr(ffmpeg, "probe_audio_duration_seconds", lambda *_, **__: 1300)
    monkeypatch.setattr(ffmpeg, "run_command", fail_run_command)

    chunks = ffmpeg.split_audio_to_timed_chunks(audio_path, tmp_path / "chunks")

    assert chunks == [ffmpeg.AudioChunk(path=audio_path)]


def test_split_audio_to_timed_chunks_uses_1300_second_segment_offsets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"x" * 2000)
    chunks_dir = tmp_path / "chunks"

    def fake_run_command(command: list[str]) -> None:
        chunks_dir.mkdir(exist_ok=True)
        (chunks_dir / "chunk_000.mp3").write_bytes(b"x" * 500)
        (chunks_dir / "chunk_001.mp3").write_bytes(b"x" * 500)
        assert command[command.index("-segment_time") + 1] == "1300"

    monkeypatch.setattr(ffmpeg, "probe_audio_duration_seconds", lambda *_, **__: 1300.1)
    monkeypatch.setattr(ffmpeg, "run_command", fake_run_command)

    chunks = ffmpeg.split_audio_to_timed_chunks(audio_path, chunks_dir)

    assert [chunk.path.name for chunk in chunks] == ["chunk_000.mp3", "chunk_001.mp3"]
    assert [chunk.start_seconds for chunk in chunks] == [0, 1300]


def test_prepare_audio_chunks_returns_single_audio_when_under_1300_seconds(
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

    assert chunks == [ffmpeg.AudioChunk(path=tmp_path / "audio.mp3")]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_ffmpeg_extracts_audio_from_generated_video(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    audio_path = tmp_path / "audio.mp3"

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
