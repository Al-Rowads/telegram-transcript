from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from telegram_transcript.transcriber import (
    DEFAULT_DEEPGRAM_LANGUAGE,
    DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL,
    DEFAULT_REFINEMENT_MODEL,
    DeepgramSpeechToTextProvider,
    LINE_TRANSLATION_REQUEST,
    LINE_TRANSLATION_SYSTEM_PROMPT,
    RAW_TRANSCRIPT_END,
    RAW_TRANSCRIPT_START,
    RAW_LINE_TRANSCRIPT_END,
    RAW_LINE_TRANSCRIPT_START,
    SRT_TRANSLATION_REQUEST,
    SRT_TRANSLATION_SYSTEM_PROMPT,
    SpeechTranscriber,
    TranscriptRefiner,
    TranscriptionError,
    build_line_translation_input,
    build_refinement_input,
    extract_deepgram_file_transcription_result,
    extract_deepgram_transcript_text,
    render_srt,
    validate_line_translated_transcript,
    validate_translated_srt,
)
from telegram_transcript.models import AudioChunk, FileTranscriptionResult, SubtitleCue


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


def test_extract_deepgram_transcript_text_prefers_paragraphs() -> None:
    response = {
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "flat transcript should not be used",
                            "paragraphs": {
                                "paragraphs": [
                                    {"text": "first paragraph"},
                                    {"text": "second paragraph"},
                                ]
                            },
                        }
                    ]
                }
            ]
        }
    }

    assert extract_deepgram_transcript_text(response) == "first paragraph\n\nsecond paragraph"


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
            utterances: bool,
        ) -> object:
            self.calls.append(
                {
                    "request": request,
                    "model": model,
                    "language": language,
                    "smart_format": smart_format,
                    "punctuate": punctuate,
                    "paragraphs": paragraphs,
                    "utterances": utterances,
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
            "utterances": True,
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
        provider_name = "deepgram"
        model = DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL

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

        def transcribe_file_result(self, audio_path: Path, *, previous_transcript: str = "") -> FileTranscriptionResult:
            return FileTranscriptionResult(
                transcript=self.transcribe_file(audio_path, previous_transcript=previous_transcript),
                subtitle_cues=(SubtitleCue(0.5, 1.0, audio_path.stem),),
            )

    class FakeRefiner:
        model = DEFAULT_REFINEMENT_MODEL

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def refine_transcript(self, transcript: str) -> str:
            self.calls.append({"transcript": transcript})
            return (
                "1\n"
                "00:00:00,500 --> 00:00:01,000\n"
                "first\n"
                '<font color="green">اول</font>\n\n'
                "2\n"
                "00:00:10,500 --> 00:00:11,000\n"
                "second\n"
                '<font color="green">دوم</font>\n'
            )

        def translate_transcript_lines(self, transcript: str) -> str:
            self.calls.append({"line_transcript": transcript})
            return "first\nاول\n\nsecond\nدوم\n"

    fake_provider = FakeSpeechToTextProvider()
    fake_refiner = FakeRefiner()
    transcriber = SpeechTranscriber(
        speech_to_text_provider=fake_provider,
        refiner=fake_refiner,
    )
    progress_events: list[tuple[str, dict[str, object]]] = []

    async def record_progress(event: str, data: object) -> None:
        progress_events.append((event, dict(data)))

    result = await transcriber.transcribe_chunks_async(
        [AudioChunk(first, 0.0), AudioChunk(second, 10.0)],
        progress_callback=record_progress,
    )

    assert result.raw_transcript == "first\n\nsecond"
    assert result.refined_transcript is None
    assert result.final_transcript == "first\n\nsecond"
    assert result.translated_srt == (
        "1\n"
        "00:00:00,500 --> 00:00:01,000\n"
        "first\n"
        '<font color="green">اول</font>\n\n'
        "2\n"
        "00:00:10,500 --> 00:00:11,000\n"
        "second\n"
        '<font color="green">دوم</font>\n'
    )
    assert result.line_translated_transcript == "first\nاول\n\nsecond\nدوم\n"
    assert result.subtitle_cues == (
        SubtitleCue(0.5, 1.0, "first"),
        SubtitleCue(10.5, 11.0, "second"),
    )
    assert fake_provider.calls == [
        {"file": "first", "previous_transcript": ""},
        {"file": "second", "previous_transcript": "first"},
    ]
    assert fake_refiner.calls == [
        {
            "transcript": (
                "1\n"
                "00:00:00,500 --> 00:00:01,000\n"
                "first\n\n"
                "2\n"
                "00:00:10,500 --> 00:00:11,000\n"
                "second\n"
            )
        },
        {"line_transcript": "first\n\nsecond"},
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
    assert progress_events[0][1]["provider"] == "deepgram"
    assert progress_events[2][1]["index"] == 2
    assert progress_events[4][1]["model"] == DEFAULT_REFINEMENT_MODEL
    assert progress_events[5][1]["line_translated_transcript_chars"] == len("first\nاول\n\nsecond\nدوم\n")


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

    assert refiner.refine_transcript("1\n00:00:00,000 --> 00:00:01,000\nهاي\n") == "cleaned transcript"
    assert fake_responses.calls == [
        {
            "model": DEFAULT_REFINEMENT_MODEL,
            "instructions": SRT_TRANSLATION_SYSTEM_PROMPT,
            "input": build_refinement_input("1\n00:00:00,000 --> 00:00:01,000\nهاي\n"),
            "temperature": 0,
        }
    ]


def test_transcript_refiner_uses_delimited_line_translation_transcript() -> None:
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
            return SimpleNamespace(output_text="هاي\nسلام\n")

    fake_responses = FakeResponses()
    fake_client = SimpleNamespace(responses=fake_responses)
    refiner = TranscriptRefiner(api_key="key", client=fake_client)

    assert refiner.translate_transcript_lines("هاي\n") == "هاي\nسلام"
    assert fake_responses.calls == [
        {
            "model": DEFAULT_REFINEMENT_MODEL,
            "instructions": LINE_TRANSLATION_SYSTEM_PROMPT,
            "input": build_line_translation_input("هاي\n"),
            "temperature": 0,
        }
    ]


def test_refinement_input_uses_delimited_raw_transcript() -> None:
    raw = "1\n00:00:00,000 --> 00:00:01,000\nهاي تجربة\n"

    refinement_input = build_refinement_input(raw)

    assert SRT_TRANSLATION_REQUEST in refinement_input
    assert RAW_TRANSCRIPT_START in refinement_input
    assert RAW_TRANSCRIPT_END in refinement_input
    assert raw in refinement_input


def test_line_translation_input_uses_delimited_raw_transcript() -> None:
    raw = "هاي تجربة\nسطر ثاني\n"

    translation_input = build_line_translation_input(raw)

    assert LINE_TRANSLATION_REQUEST in translation_input
    assert RAW_LINE_TRANSCRIPT_START in translation_input
    assert RAW_LINE_TRANSCRIPT_END in translation_input
    assert raw in translation_input


def test_transcribe_chunks_skips_refinement_for_empty_transcript(tmp_path: Path) -> None:
    audio = tmp_path / "empty.mp3"
    audio.write_bytes(b"empty")

    class FakeSpeechToTextProvider:
        provider_name = "deepgram"
        model = DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL

        def transcribe_file(self, audio_path: Path, *, previous_transcript: str = "") -> str:
            return " "

    class FakeRefiner:
        model = DEFAULT_REFINEMENT_MODEL

        def refine_transcript(self, transcript: str) -> str:
            raise AssertionError("Empty transcripts should not be refined.")

    transcriber = SpeechTranscriber(
        speech_to_text_provider=FakeSpeechToTextProvider(),
        refiner=FakeRefiner(),
    )

    assert transcriber.transcribe_chunks([audio]) == ""


def test_transcribe_chunks_skips_refinement_when_disabled(tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")

    class FakeSpeechToTextProvider:
        provider_name = "deepgram"
        model = DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL

        def transcribe_file(self, audio_path: Path, *, previous_transcript: str = "") -> str:
            return "raw transcript"

    transcriber = SpeechTranscriber(speech_to_text_provider=FakeSpeechToTextProvider())

    assert transcriber.transcribe_chunks([audio]) == "raw transcript"


def test_transcribe_chunks_skips_refinement_without_subtitle_cues(tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")

    class FakeSpeechToTextProvider:
        provider_name = "deepgram"
        model = DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL

        def transcribe_file(self, audio_path: Path, *, previous_transcript: str = "") -> str:
            return "raw transcript"

    class FakeRefiner:
        model = DEFAULT_REFINEMENT_MODEL

        def refine_transcript(self, transcript: str) -> str:
            raise AssertionError("SRT refinement requires subtitle cues.")

    transcriber = SpeechTranscriber(
        speech_to_text_provider=FakeSpeechToTextProvider(),
        refiner=FakeRefiner(),
    )

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

    class FakeSpeechToTextProvider:
        provider_name = "deepgram"
        model = DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL

        def transcribe_file(self, audio_path: Path, *, previous_transcript: str = "") -> str:
            return "raw transcript"

    transcriber = SpeechTranscriber(speech_to_text_provider=FakeSpeechToTextProvider())
    progress_events: list[str] = []

    async def record_progress(event: str, data: object) -> None:
        progress_events.append(event)

    result = await transcriber.transcribe_chunks_async([audio], progress_callback=record_progress)

    assert result.raw_transcript == "raw transcript"
    assert result.refined_transcript is None
    assert progress_events == ["transcribing_chunk", "chunk_transcribed"]


def test_extract_deepgram_file_transcription_result_prefers_utterance_cues() -> None:
    response = {
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "هلا بالعالم",
                            "words": [
                                {"start": 0.1, "end": 0.3, "word": "هلا"},
                            ],
                        }
                    ]
                }
            ],
            "utterances": [
                {"start": 0.1, "end": 1.4, "transcript": "هلا بالعالم"},
            ],
        }
    }

    result = extract_deepgram_file_transcription_result(response)

    assert result.transcript == "هلا بالعالم"
    assert result.subtitle_cues == (SubtitleCue(0.1, 1.4, "هلا بالعالم"),)


def test_extract_deepgram_file_transcription_result_falls_back_to_word_cues() -> None:
    response = {
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "هلا بالعالم",
                            "words": [
                                {"start": 0.1, "end": 0.3, "punctuated_word": "هلا"},
                                {"start": 0.4, "end": 0.8, "word": "بالعالم"},
                            ],
                        }
                    ]
                }
            ],
        }
    }

    result = extract_deepgram_file_transcription_result(response)

    assert result.subtitle_cues == (
        SubtitleCue(0.1, 0.3, "هلا"),
        SubtitleCue(0.4, 0.8, "بالعالم"),
    )


def test_render_srt_formats_cues() -> None:
    assert render_srt((SubtitleCue(0.001, 61.234, "hello world"),)) == (
        "1\n"
        "00:00:00,001 --> 00:01:01,234\n"
        "hello world\n"
    )


def test_validate_translated_srt_accepts_green_persian_line() -> None:
    raw_srt = "1\n00:00:00,001 --> 00:00:01,000\nهلا بالعالم\n"
    translated_srt = (
        "1\n"
        "00:00:00,001 --> 00:00:01,000\n"
        "هلا بالعالم\n"
        '<font color="green">سلام دنیا</font>\n'
    )

    assert validate_translated_srt(raw_srt=raw_srt, translated_srt=translated_srt) == translated_srt


def test_validate_translated_srt_wraps_plain_persian_line() -> None:
    raw_srt = "1\n00:00:00,001 --> 00:00:01,000\nهلا بالعالم\n"
    translated_srt = "1\n00:00:00,001 --> 00:00:01,000\nهلا بالعالم\nسلام دنیا\n"

    assert validate_translated_srt(raw_srt=raw_srt, translated_srt=translated_srt) == (
        "1\n"
        "00:00:00,001 --> 00:00:01,000\n"
        "هلا بالعالم\n"
        '<font color="green">سلام دنیا</font>\n'
    )


def test_validate_line_translated_transcript_accepts_pairs() -> None:
    raw_transcript = "هلا بالعالم\n\nشلونك؟\n"
    translated_transcript = "هلا بالعالم\nسلام دنیا\n\nشلونك؟\nحالت چطوره؟\n"

    assert validate_line_translated_transcript(
        raw_transcript=raw_transcript,
        translated_transcript=translated_transcript,
    ) == "هلا بالعالم\nسلام دنیا\n\nشلونك؟\nحالت چطوره؟\n"


def test_validate_line_translated_transcript_canonicalizes_missing_blank_lines() -> None:
    raw_transcript = "هلا بالعالم\nشلونك؟\n"
    translated_transcript = "هلا بالعالم\nسلام دنیا\nشلونك؟\nحالت چطوره؟\n"

    assert validate_line_translated_transcript(
        raw_transcript=raw_transcript,
        translated_transcript=translated_transcript,
    ) == "هلا بالعالم\nسلام دنیا\n\nشلونك؟\nحالت چطوره؟\n"


def test_validate_line_translated_transcript_rejects_changed_arabic_text() -> None:
    raw_transcript = "هلا بالعالم\n"
    translated_transcript = "مرحبا بالعالم\nسلام دنیا\n"

    with pytest.raises(TranscriptionError, match="Arabic"):
        validate_line_translated_transcript(
            raw_transcript=raw_transcript,
            translated_transcript=translated_transcript,
        )


def test_validate_line_translated_transcript_rejects_missing_translation_line() -> None:
    raw_transcript = "هلا بالعالم\n"
    translated_transcript = "هلا بالعالم\n"

    with pytest.raises(TranscriptionError, match="exactly one Persian line"):
        validate_line_translated_transcript(
            raw_transcript=raw_transcript,
            translated_transcript=translated_transcript,
        )


def test_validate_translated_srt_rejects_changed_timestamp() -> None:
    raw_srt = "1\n00:00:00,001 --> 00:00:01,000\nهلا بالعالم\n"
    translated_srt = (
        "1\n"
        "00:00:00,002 --> 00:00:01,000\n"
        "هلا بالعالم\n"
        '<font color="green">سلام دنیا</font>\n'
    )

    with pytest.raises(TranscriptionError, match="timestamp"):
        validate_translated_srt(raw_srt=raw_srt, translated_srt=translated_srt)


def test_validate_translated_srt_rejects_changed_arabic_text() -> None:
    raw_srt = "1\n00:00:00,001 --> 00:00:01,000\nهلا بالعالم\n"
    translated_srt = (
        "1\n"
        "00:00:00,001 --> 00:00:01,000\n"
        "مرحبا بالعالم\n"
        '<font color="green">سلام دنیا</font>\n'
    )

    with pytest.raises(TranscriptionError, match="Arabic"):
        validate_translated_srt(raw_srt=raw_srt, translated_srt=translated_srt)


def test_validate_translated_srt_rejects_missing_translation_line() -> None:
    raw_srt = "1\n00:00:00,001 --> 00:00:01,000\nهلا بالعالم\n"
    translated_srt = "1\n00:00:00,001 --> 00:00:01,000\nهلا بالعالم\n"

    with pytest.raises(TranscriptionError, match="exactly one Persian line"):
        validate_translated_srt(raw_srt=raw_srt, translated_srt=translated_srt)


def test_transcript_refiner_raises_when_refinement_has_no_text() -> None:
    class FakeResponses:
        def create(self, *, model: str, instructions: str, input: str, temperature: int) -> object:
            return SimpleNamespace()

    fake_client = SimpleNamespace(responses=FakeResponses())
    refiner = TranscriptRefiner(api_key="key", client=fake_client)
    with pytest.raises(TranscriptionError, match="refinement response"):
        refiner.refine_transcript("1\n00:00:00,000 --> 00:00:01,000\nهاي\n")
