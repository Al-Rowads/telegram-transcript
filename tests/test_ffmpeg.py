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


def test_build_extract_audio_command_accepts_default_noise_filter() -> None:
    command = ffmpeg.build_extract_audio_command(
        Path("input.mp4"),
        Path("output.mp3"),
        noise_reduction_filter="anlmdn",
    )

    assert ["-filter:a", "anlmdn,atempo=1"] == command[command.index("-filter:a") : command.index("-filter:a") + 2]


def test_build_extract_audio_command_accepts_extra_noise_filter() -> None:
    command = ffmpeg.build_extract_audio_command(
        Path("input.mp4"),
        Path("output.mp3"),
        audio_tempo=1.25,
        noise_reduction_filter="highpass=f=80,afftdn=nr=15,loudnorm",
    )

    assert ["-filter:a", "highpass=f=80,afftdn=nr=15,loudnorm,atempo=1.25"] == command[
        command.index("-filter:a") : command.index("-filter:a") + 2
    ]


def test_split_audio_to_chunks_retries_until_chunks_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"x" * 2000)
    chunks_dir = tmp_path / "chunks"
    attempts = 0

    def fake_run_command(command: list[str]) -> None:
        nonlocal attempts
        attempts += 1
        chunks_dir.mkdir(exist_ok=True)
        if attempts == 1:
            (chunks_dir / "chunk_000.mp3").write_bytes(b"x" * 1100)
            return
        (chunks_dir / "chunk_000.mp3").write_bytes(b"x" * 500)
        (chunks_dir / "chunk_001.mp3").write_bytes(b"x" * 500)

    monkeypatch.setattr(ffmpeg, "run_command", fake_run_command)

    chunks = ffmpeg.split_audio_to_chunks(audio_path, chunks_dir, max_audio_bytes=1000)

    assert attempts == 2
    assert [chunk.name for chunk in chunks] == ["chunk_000.mp3", "chunk_001.mp3"]


def test_prepare_audio_chunks_returns_single_audio_when_under_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")

    def fake_extract_audio(video: Path, audio: Path, **_: object) -> Path:
        audio.write_bytes(b"audio")
        return audio

    monkeypatch.setattr(ffmpeg, "extract_audio", fake_extract_audio)

    chunks = ffmpeg.prepare_audio_chunks(video_path, tmp_path, max_audio_bytes=100)

    assert chunks == [tmp_path / "audio.mp3"]


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
