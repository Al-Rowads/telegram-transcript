from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from telegram_transcript.transcriber import (
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

    fake_transcriptions = FakeTranscriptions()
    fake_client = SimpleNamespace(audio=SimpleNamespace(transcriptions=fake_transcriptions))
    transcriber = OpenAITranscriber(api_key="key", model="whisper-1", client=fake_client)

    assert transcriber.transcribe_chunks([first, second]) == "first\n\nsecond"
    assert fake_transcriptions.prompts == [IRAQI_ARABIC_SYSTEM_PROMPT, IRAQI_ARABIC_SYSTEM_PROMPT]
