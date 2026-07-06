from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from telegram_transcript.transcriber import (
    BAGHDADI_ARABIC_REFINEMENT_REQUEST,
    BAGHDADI_ARABIC_REFINEMENT_SYSTEM_PROMPT,
    DEFAULT_DEEPGRAM_LANGUAGE,
    DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL,
    DEFAULT_REFINEMENT_MODEL,
    DEFAULT_TRANSCRIPTION_MODEL,
    DeepgramSpeechToTextProvider,
    IRAQI_ARABIC_TRANSCRIPTION_PROMPT,
    OpenAITranscriber,
    OpenAISpeechToTextProvider,
    RAW_TRANSCRIPT_END,
    RAW_TRANSCRIPT_START,
    SpeechTranscriber,
    TranscriptRefiner,
    TranscriptionError,
    build_refinement_input,
    extract_deepgram_transcript_text,
    extract_transcript_text,
)


def test_extract_transcript_text_from_object() -> None:
    assert extract_transcript_text(SimpleNamespace(text="hello")) == "hello"


def test_extract_transcript_text_rejects_missing_text() -> None:
    with pytest.raises(TranscriptionError):
        extract_transcript_text(SimpleNamespace())


def test_extract_deepgram_transcript_text_from_dict() -> None:
    response = {
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "هلا بالعالم",
                        }
                    ]
                }
            ]
        }
    }

    assert extract_deepgram_transcript_text(response) == "هلا بالعالم"


def test_extract_deepgram_transcript_text_rejects_missing_transcript() -> None:
    with pytest.raises(TranscriptionError, match="Deepgram"):
        extract_deepgram_transcript_text({"results": {"channels": []}})


def test_deepgram_provider_uses_nova_3_arabic(tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")

    class FakeMedia:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def transcribe_file(
            self,
            *,
            request: bytes,
            model: str,
            language: str,
            smart_format: bool,
            punctuate: bool,
            paragraphs: bool,
        ) -> object:
            self.calls.append(
                {
                    "request": request,
                    "model": model,
                    "language": language,
                    "smart_format": smart_format,
                    "punctuate": punctuate,
                    "paragraphs": paragraphs,
                }
            )
            return {
                "results": {
                    "channels": [
                        {
                            "alternatives": [
                                {
                                    "transcript": "نص خام",
                                }
                            ]
                        }
                    ]
                }
            }

    fake_media = FakeMedia()
    fake_client = SimpleNamespace(listen=SimpleNamespace(v1=SimpleNamespace(media=fake_media)))
    provider = DeepgramSpeechToTextProvider(api_key="key", client=fake_client)

    assert provider.transcribe_file(audio) == "نص خام"
    assert fake_media.calls == [
        {
            "request": b"audio",
            "model": DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL,
            "language": DEFAULT_DEEPGRAM_LANGUAGE,
            "smart_format": True,
            "punctuate": True,
            "paragraphs": True,
        }
    ]


@pytest.mark.asyncio
async def test_transcribe_chunks_reports_progress_and_refines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_inline(func: object, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)
    first = tmp_path / "first.mp3"
    second = tmp_path / "second.mp3"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    class FakeSpeechToTextProvider:
        provider_name = "openai"
        model = DEFAULT_TRANSCRIPTION_MODEL

        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        def transcribe_file(self, audio_path: Path, *, previous_transcript: str = "") -> str:
            self.calls.append(
                {
                    "file": audio_path.stem,
                    "previous_transcript": previous_transcript,
                }
            )
            return audio_path.stem

    class FakeRefiner:
        model = DEFAULT_REFINEMENT_MODEL

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def refine_transcript(self, transcript: str) -> str:
            self.calls.append({"transcript": transcript})
            return "cleaned transcript"

    fake_provider = FakeSpeechToTextProvider()
    fake_refiner = FakeRefiner()
    transcriber = SpeechTranscriber(
        speech_to_text_provider=fake_provider,
        refiner=fake_refiner,
    )
    progress_events: list[tuple[str, dict[str, object]]] = []

    async def record_progress(event: str, data: object) -> None:
        progress_events.append((event, dict(data)))

    assert await transcriber.transcribe_chunks_async([first, second], progress_callback=record_progress) == "cleaned transcript"
    assert fake_provider.calls == [
        {"file": "first", "previous_transcript": ""},
        {"file": "second", "previous_transcript": "first"},
    ]
    assert fake_refiner.calls == [{"transcript": "first\n\nsecond"}]
    assert [event for event, _ in progress_events] == [
        "transcribing_chunk",
        "chunk_transcribed",
        "transcribing_chunk",
        "chunk_transcribed",
        "refining_transcript",
        "refinement_complete",
    ]
    assert progress_events[0][1]["index"] == 1
    assert progress_events[0][1]["provider"] == "openai"
    assert progress_events[2][1]["index"] == 2
    assert progress_events[4][1]["model"] == DEFAULT_REFINEMENT_MODEL


def test_openai_provider_uses_iraqi_arabic_prompt_context(tmp_path: Path) -> None:
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

    fake_transcriptions = FakeTranscriptions()
    fake_client = SimpleNamespace(audio=SimpleNamespace(transcriptions=fake_transcriptions))
    provider = OpenAISpeechToTextProvider(api_key="key", client=fake_client)

    assert provider.transcribe_file(first) == "first"
    assert provider.transcribe_file(second, previous_transcript="first") == "second"
    assert [call["file"] for call in fake_transcriptions.calls] == ["first", "second"]
    assert fake_transcriptions.calls[0]["model"] == DEFAULT_TRANSCRIPTION_MODEL
    assert fake_transcriptions.calls[0]["prompt"] == IRAQI_ARABIC_TRANSCRIPTION_PROMPT
    assert fake_transcriptions.calls[1]["prompt"].startswith(IRAQI_ARABIC_TRANSCRIPTION_PROMPT)
    assert "first" in fake_transcriptions.calls[1]["prompt"]


def test_transcript_refiner_uses_delimited_raw_transcript() -> None:
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
            return SimpleNamespace(output_text="cleaned transcript")

    fake_responses = FakeResponses()
    fake_client = SimpleNamespace(responses=fake_responses)
    refiner = TranscriptRefiner(api_key="key", client=fake_client)

    assert refiner.refine_transcript("first\n\nsecond") == "cleaned transcript"
    assert fake_responses.calls == [
        {
            "model": DEFAULT_REFINEMENT_MODEL,
            "instructions": BAGHDADI_ARABIC_REFINEMENT_SYSTEM_PROMPT,
            "input": build_refinement_input("first\n\nsecond"),
            "temperature": 0,
        }
    ]


def test_refinement_input_uses_delimited_raw_transcript() -> None:
    raw = "هاي تجربة"

    refinement_input = build_refinement_input(raw)

    assert BAGHDADI_ARABIC_REFINEMENT_REQUEST in refinement_input
    assert RAW_TRANSCRIPT_START in refinement_input
    assert RAW_TRANSCRIPT_END in refinement_input
    assert raw in refinement_input


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

    assert transcriber.transcribe_chunks([audio]) == "raw transcript"


@pytest.mark.asyncio
async def test_transcribe_chunks_async_skips_refinement_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_inline(func: object, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)
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

    assert await transcriber.transcribe_chunks_async([audio], progress_callback=record_progress) == "raw transcript"
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
