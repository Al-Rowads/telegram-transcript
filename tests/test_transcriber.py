from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from telegram_transcript.transcriber import (
    BAGHDADI_ARABIC_REFINEMENT_REQUEST,
    BAGHDADI_ARABIC_REFINEMENT_SYSTEM_PROMPT,
    DEFAULT_REFINEMENT_MODEL,
    DEFAULT_TRANSCRIPTION_MODEL,
    IRAQI_ARABIC_TRANSCRIPTION_PROMPT,
    OpenAITranscriber,
    RAW_TRANSCRIPT_END,
    RAW_TRANSCRIPT_START,
    TranscriptionResult,
    TranscriptionError,
    build_refinement_input,
    extract_refined_message,
    extract_transcript_text,
)


def test_extract_transcript_text_from_object() -> None:
    assert extract_transcript_text(SimpleNamespace(text="hello")) == "hello"


def test_extract_transcript_text_rejects_missing_text() -> None:
    with pytest.raises(TranscriptionError):
        extract_transcript_text(SimpleNamespace())


@pytest.mark.asyncio
async def test_transcribe_chunks_reports_progress_and_refines(tmp_path: Path) -> None:
    first = tmp_path / "first.mp3"
    second = tmp_path / "second.mp3"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    class FakeTranscriptions:
        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        def create(self, *, model: str, file: object, prompt: str) -> object:
            self.calls.append({"model": model, "file": Path(file.name).stem, "prompt": prompt})
            return SimpleNamespace(text=Path(file.name).stem)

    class FakeResponses:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, *, model: str, instructions: str, input: str, temperature: int) -> object:
            self.calls.append(
                {
                    "model": model,
                    "instructions": instructions,
                    "input": input,
                    "temperature": temperature,
                }
            )
            return SimpleNamespace(
                output_text=(
                    "<transcription>\n"
                    "first\n\nsecond\n"
                    "</transcription>\n\n"
                    "<translation>\n"
                    "+ first\n"
                    "اول\n"
                    "* first\n"
                    "</translation>"
                )
            )

    fake_transcriptions = FakeTranscriptions()
    fake_responses = FakeResponses()
    fake_client = SimpleNamespace(
        audio=SimpleNamespace(transcriptions=fake_transcriptions),
        responses=fake_responses,
    )
    transcriber = OpenAITranscriber(api_key="key", client=fake_client)
    progress_events: list[tuple[str, dict[str, object]]] = []

    async def record_progress(event: str, data: object) -> None:
        progress_events.append((event, dict(data)))

    result = await transcriber.transcribe_chunks_async_result([first, second], progress_callback=record_progress)

    assert result == TranscriptionResult(
        transcription="first\n\nsecond",
        refined_message="+ first\nاول\n* first",
    )
    assert result.best_text == "+ first\nاول\n* first"
    assert [call["file"] for call in fake_transcriptions.calls] == ["first", "second"]
    assert fake_transcriptions.calls[0]["model"] == DEFAULT_TRANSCRIPTION_MODEL
    assert fake_transcriptions.calls[0]["prompt"] == IRAQI_ARABIC_TRANSCRIPTION_PROMPT
    assert fake_transcriptions.calls[1]["prompt"].startswith(IRAQI_ARABIC_TRANSCRIPTION_PROMPT)
    assert "first" in fake_transcriptions.calls[1]["prompt"]
    assert fake_responses.calls == [
        {
            "model": DEFAULT_REFINEMENT_MODEL,
            "instructions": BAGHDADI_ARABIC_REFINEMENT_SYSTEM_PROMPT,
            "input": build_refinement_input("first\n\nsecond"),
            "temperature": 0,
        }
    ]
    assert [event for event, _ in progress_events] == [
        "transcribing_chunk",
        "chunk_transcribed",
        "transcribing_chunk",
        "chunk_transcribed",
        "refining_transcript",
        "refinement_complete",
    ]
    assert progress_events[0][1]["index"] == 1
    assert progress_events[2][1]["index"] == 2
    assert progress_events[4][1]["model"] == DEFAULT_REFINEMENT_MODEL


def test_extract_refined_message_uses_translation_block() -> None:
    refined = "<transcription>\nraw\n</transcription>\n<translation>\nrefined\n</translation>"

    assert extract_refined_message(refined) == "refined"


def test_extract_refined_message_falls_back_to_full_text_without_translation_block() -> None:
    assert extract_refined_message("cleaned transcript") == "cleaned transcript"


def test_refinement_input_uses_delimited_raw_transcript() -> None:
    raw = "هاي تجربة"

    refinement_input = build_refinement_input(raw)

    assert BAGHDADI_ARABIC_REFINEMENT_REQUEST in refinement_input
    assert RAW_TRANSCRIPT_START in refinement_input
    assert RAW_TRANSCRIPT_END in refinement_input
    assert raw in refinement_input


def test_refinement_prompt_describes_translation_editing_contract() -> None:
    combined_prompt = f"{BAGHDADI_ARABIC_REFINEMENT_SYSTEM_PROMPT}\n{BAGHDADI_ARABIC_REFINEMENT_REQUEST}"

    assert "same transcription" in combined_prompt
    assert "line by line" in combined_prompt
    assert "Persian translation" in combined_prompt
    assert "key word" in combined_prompt
    assert "unseparable words" in combined_prompt
    assert "<transcription>" in combined_prompt
    assert "</transcription>" in combined_prompt
    assert "<translation>" in combined_prompt
    assert "</translation>" in combined_prompt


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
    transcriber = OpenAITranscriber(api_key="key", client=fake_client)

    assert transcriber.transcribe_chunks_result([audio]) == TranscriptionResult(transcription="")
    assert transcriber.transcribe_chunks([audio]) == ""


def test_transcribe_chunks_skips_refinement_when_disabled(tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")

    class FakeTranscriptions:
        def create(self, *, model: str, file: object, prompt: str) -> object:
            return SimpleNamespace(text="raw transcript")

    class FakeResponses:
        def create(self, **_: object) -> object:
            raise AssertionError("Refinement should not run when REFINE=false.")

    fake_client = SimpleNamespace(
        audio=SimpleNamespace(transcriptions=FakeTranscriptions()),
        responses=FakeResponses(),
    )
    transcriber = OpenAITranscriber(api_key="key", client=fake_client, refine=False)

    assert transcriber.transcribe_chunks_result([audio]) == TranscriptionResult(transcription="raw transcript")
    assert transcriber.transcribe_chunks([audio]) == "raw transcript"


@pytest.mark.asyncio
async def test_transcribe_chunks_async_skips_refinement_when_disabled(tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")

    class FakeTranscriptions:
        def create(self, *, model: str, file: object, prompt: str) -> object:
            return SimpleNamespace(text="raw transcript")

    class FakeResponses:
        def create(self, **_: object) -> object:
            raise AssertionError("Refinement should not run when REFINE=false.")

    fake_client = SimpleNamespace(
        audio=SimpleNamespace(transcriptions=FakeTranscriptions()),
        responses=FakeResponses(),
    )
    transcriber = OpenAITranscriber(api_key="key", client=fake_client, refine=False)
    progress_events: list[str] = []

    async def record_progress(event: str, data: object) -> None:
        progress_events.append(event)

    assert await transcriber.transcribe_chunks_async_result(
        [audio],
        progress_callback=record_progress,
    ) == TranscriptionResult(transcription="raw transcript")
    assert await transcriber.transcribe_chunks_async([audio]) == "raw transcript"
    assert progress_events == ["transcribing_chunk", "chunk_transcribed"]


def test_transcribe_chunks_raises_when_refinement_has_no_text(tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")

    class FakeTranscriptions:
        def create(self, *, model: str, file: object, prompt: str) -> object:
            return SimpleNamespace(text="raw transcript")

    class FakeResponses:
        def create(self, *, model: str, instructions: str, input: str, temperature: int) -> object:
            return SimpleNamespace()

    fake_client = SimpleNamespace(
        audio=SimpleNamespace(transcriptions=FakeTranscriptions()),
        responses=FakeResponses(),
    )
    transcriber = OpenAITranscriber(api_key="key", client=fake_client)

    with pytest.raises(TranscriptionError, match="refinement response"):
        transcriber.transcribe_chunks([audio])
