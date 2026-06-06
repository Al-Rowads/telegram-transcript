from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from telegram_transcript.transcriber import (
    BAGHDADI_ARABIC_REFINEMENT_REQUEST,
    BAGHDADI_ARABIC_REFINEMENT_SYSTEM_PROMPT,
    DEFAULT_REFINEMENT_MODEL,
    IRAQI_ARABIC_SYSTEM_PROMPT,
    OpenAITranscriber,
    TranscriptionError,
    extract_transcript_text,
)


def test_extract_transcript_text_from_object() -> None:
    assert extract_transcript_text(SimpleNamespace(text="hello")) == "hello"


def test_extract_transcript_text_rejects_missing_text() -> None:
    with pytest.raises(TranscriptionError):
        extract_transcript_text(SimpleNamespace())


def test_transcribe_chunks_preserves_order(tmp_path: Path) -> None:
    first = tmp_path / "first.mp3"
    second = tmp_path / "second.mp3"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    class FakeTranscriptions:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def create(self, *, model: str, file: object, prompt: str) -> object:
            assert model == "whisper-1"
            self.prompts.append(prompt)
            return SimpleNamespace(text=Path(file.name).stem)

    class FakeResponses:
        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        def create(self, *, model: str, instructions: str, input: str) -> object:
            self.calls.append({"model": model, "instructions": instructions, "input": input})
            return SimpleNamespace(output_text="cleaned transcript")

    fake_transcriptions = FakeTranscriptions()
    fake_responses = FakeResponses()
    fake_client = SimpleNamespace(
        audio=SimpleNamespace(transcriptions=fake_transcriptions),
        responses=fake_responses,
    )
    transcriber = OpenAITranscriber(api_key="key", model="whisper-1", client=fake_client)

    assert transcriber.transcribe_chunks([first, second]) == "cleaned transcript"
    assert fake_transcriptions.prompts == [IRAQI_ARABIC_SYSTEM_PROMPT, IRAQI_ARABIC_SYSTEM_PROMPT]
    assert fake_responses.calls == [
        {
            "model": DEFAULT_REFINEMENT_MODEL,
            "instructions": BAGHDADI_ARABIC_REFINEMENT_SYSTEM_PROMPT,
            "input": f"{BAGHDADI_ARABIC_REFINEMENT_REQUEST}\n\nfirst\n\nsecond",
        }
    ]


def test_transcribe_chunks_skips_refinement_for_empty_transcript(tmp_path: Path) -> None:
    audio = tmp_path / "empty.mp3"
    audio.write_bytes(b"empty")

    class FakeTranscriptions:
        def create(self, *, model: str, file: object, prompt: str) -> object:
            return SimpleNamespace(text=" ")

    class FakeResponses:
        def create(self, **_: object) -> object:
            raise AssertionError("Empty transcripts should not be refined.")

    fake_client = SimpleNamespace(
        audio=SimpleNamespace(transcriptions=FakeTranscriptions()),
        responses=FakeResponses(),
    )
    transcriber = OpenAITranscriber(api_key="key", model="whisper-1", client=fake_client)

    assert transcriber.transcribe_chunks([audio]) == ""
