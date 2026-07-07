from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from telegram_transcript.transcriber import (
    DEFAULT_DEEPGRAM_LANGUAGE,
    DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL,
    DEFAULT_GEMINI_TRANSCRIPTION_MODEL,
    DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
    DEFAULT_REFINEMENT_MODEL,
    DeepgramSpeechToTextProvider,
    GeminiSpeechToTextProvider,
    OpenAISpeechToTextProvider,
    RAW_TRANSCRIPT_END,
    RAW_TRANSCRIPT_START,
    SRT_TRANSLATION_REQUEST,
    SRT_TRANSLATION_SYSTEM_PROMPT,
    SpeechTranscriber,
    TranscriptRefiner,
    TranscriptionError,
    assemble_srt_chunks,
    build_refinement_input,
    extract_deepgram_file_transcription_result,
    extract_deepgram_transcript_text,
    parse_srt_blocks,
    render_line_translated_transcript_from_srt,
    render_srt,
    split_srt_by_byte_limit,
)
from telegram_transcript import transcriber as transcriber_module
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


def test_openai_provider_uses_diarized_json_and_renders_srt_cues(tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")

    class FakeTranscriptions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(
            self,
            *,
            file: object,
            model: str,
            response_format: str,
            chunking_strategy: str,
            temperature: int,
        ) -> object:
            self.calls.append(
                {
                    "file_name": file.name,
                    "model": model,
                    "response_format": response_format,
                    "chunking_strategy": chunking_strategy,
                    "temperature": temperature,
                }
            )
            return SimpleNamespace(
                segments=[
                    SimpleNamespace(start=0.0, end=1.25, speaker="speaker_0", text="هلا"),
                    SimpleNamespace(start=1.25, end=2.0, speaker="speaker_1", text="شلونك"),
                ],
            )

    fake_transcriptions = FakeTranscriptions()
    fake_client = SimpleNamespace(audio=SimpleNamespace(transcriptions=fake_transcriptions))
    provider = OpenAISpeechToTextProvider(api_key="key", client=fake_client)

    result = provider.transcribe_file_result(audio)

    assert result.transcript == "speaker_0: هلا\n\nspeaker_1: شلونك"
    assert result.subtitle_cues == (
        SubtitleCue(0.0, 1.25, "speaker_0: هلا"),
        SubtitleCue(1.25, 2.0, "speaker_1: شلونك"),
    )
    assert fake_transcriptions.calls == [
        {
            "file_name": str(audio),
            "model": DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
            "response_format": "diarized_json",
            "chunking_strategy": "auto",
            "temperature": 0,
        }
    ]


def test_openai_provider_rejects_missing_diarized_segments(tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")

    class FakeTranscriptions:
        def create(self, **_: object) -> object:
            return SimpleNamespace(text="plain transcript")

    fake_client = SimpleNamespace(audio=SimpleNamespace(transcriptions=FakeTranscriptions()))
    provider = OpenAISpeechToTextProvider(api_key="key", client=fake_client)

    with pytest.raises(TranscriptionError, match="OpenAI diarized"):
        provider.transcribe_file_result(audio)


def test_gemini_provider_requests_srt_and_parses_response(tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")

    class FakeFiles:
        def __init__(self) -> None:
            self.uploads: list[str] = []

        def upload(self, *, file: str) -> object:
            self.uploads.append(file)
            return SimpleNamespace(uri="uploaded-audio")

    class FakeModels:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate_content(self, *, model: str, contents: list[object], config: object) -> object:
            self.calls.append({"model": model, "contents": contents, "config": config})
            return SimpleNamespace(
                text=(
                    "```srt\n"
                    "1\n"
                    "00:00:00,000 --> 00:00:01,250\n"
                    "هلا\n\n"
                    "2\n"
                    "00:00:01,250 --> 00:00:02,000\n"
                    "شلونك\n"
                    "```"
                )
            )

    fake_files = FakeFiles()
    fake_models = FakeModels()
    fake_client = SimpleNamespace(files=fake_files, models=fake_models)
    provider = GeminiSpeechToTextProvider(api_key="key", client=fake_client)

    result = provider.transcribe_file_result(audio)

    assert result.transcript == "هلا\n\nشلونك"
    assert result.subtitle_cues == (
        SubtitleCue(0.0, 1.25, "هلا"),
        SubtitleCue(1.25, 2.0, "شلونك"),
    )
    assert fake_files.uploads == [str(audio)]
    assert fake_models.calls[0]["model"] == DEFAULT_GEMINI_TRANSCRIPTION_MODEL
    assert "SRT" in fake_models.calls[0]["contents"][0]
    assert fake_models.calls[0]["contents"][1].uri == "uploaded-audio"


def test_gemini_provider_rejects_invalid_srt(tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")

    class FakeFiles:
        def upload(self, *, file: str) -> object:
            return SimpleNamespace(uri=file)

    class FakeModels:
        def generate_content(self, **_: object) -> object:
            return SimpleNamespace(text="plain transcript")

    fake_client = SimpleNamespace(files=FakeFiles(), models=FakeModels())
    provider = GeminiSpeechToTextProvider(api_key="key", client=fake_client)

    with pytest.raises(TranscriptionError, match="SRT"):
        provider.transcribe_file_result(audio)


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
    assert progress_events[0][1]["provider"] == "deepgram"
    assert progress_events[2][1]["index"] == 2
    assert progress_events[4][1]["model"] == DEFAULT_REFINEMENT_MODEL
    assert progress_events[5][1]["line_translated_transcript_chars"] == len("first\nاول\n\nsecond\nدوم\n")


@pytest.mark.asyncio
async def test_transcribe_chunks_async_refines_large_srt_in_valid_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_inline(func: object, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)
    monkeypatch.setattr(transcriber_module, "MAX_SRT_TRANSLATION_CHUNK_BYTES", 75)
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")

    class FakeSpeechToTextProvider:
        provider_name = "deepgram"
        model = DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL

        def transcribe_file_result(self, audio_path: Path, *, previous_transcript: str = "") -> FileTranscriptionResult:
            return FileTranscriptionResult(
                transcript="first second third",
                subtitle_cues=(
                    SubtitleCue(0.0, 1.0, "first"),
                    SubtitleCue(1.0, 2.0, "second"),
                    SubtitleCue(2.0, 3.0, "third"),
                ),
            )

    class FakeRefiner:
        model = DEFAULT_REFINEMENT_MODEL

        def __init__(self) -> None:
            self.calls: list[str] = []

        def refine_transcript(self, transcript: str) -> str:
            self.calls.append(transcript)
            translated_blocks = []
            for block in parse_srt_blocks(transcript):
                translated_blocks.append(
                    "\n".join(
                        [
                            block.index,
                            block.timestamp,
                            *block.text_lines,
                            f'<font color="green">ترجمه {block.index}</font>',
                        ]
                    )
                )
            return "\n\n".join(translated_blocks) + "\n"

    fake_refiner = FakeRefiner()
    transcriber = SpeechTranscriber(
        speech_to_text_provider=FakeSpeechToTextProvider(),
        refiner=fake_refiner,
    )
    progress_events: list[tuple[str, dict[str, object]]] = []

    async def record_progress(event: str, data: object) -> None:
        progress_events.append((event, dict(data)))

    result = await transcriber.transcribe_chunks_async([audio], progress_callback=record_progress)

    raw_srt = render_srt(
        (
            SubtitleCue(0.0, 1.0, "first"),
            SubtitleCue(1.0, 2.0, "second"),
            SubtitleCue(2.0, 3.0, "third"),
        )
    )
    assert len(fake_refiner.calls) == 3
    assert all(len(call.encode("utf-8")) <= 75 for call in fake_refiner.calls)
    assert assemble_srt_chunks(fake_refiner.calls) == raw_srt
    assert result.translated_srt == (
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "first\n"
        '<font color="green">ترجمه 1</font>\n\n'
        "2\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "second\n"
        '<font color="green">ترجمه 2</font>\n\n'
        "3\n"
        "00:00:02,000 --> 00:00:03,000\n"
        "third\n"
        '<font color="green">ترجمه 3</font>\n'
    )
    assert result.line_translated_transcript == (
        "first\nترجمه 1\n\nsecond\nترجمه 2\n\nthird\nترجمه 3\n"
    )
    assert [
        (data["index"], data["total"])
        for event, data in progress_events
        if event == "refining_transcript"
    ] == [(1, 3), (2, 3), (3, 3)]


@pytest.mark.asyncio
async def test_transcribe_chunks_async_accepts_flexible_openai_translation(
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

        def transcribe_file_result(self, audio_path: Path, *, previous_transcript: str = "") -> FileTranscriptionResult:
            return FileTranscriptionResult(
                transcript="first second third fourth",
                subtitle_cues=(
                    SubtitleCue(0.0, 1.0, "first"),
                    SubtitleCue(1.0, 2.0, "second"),
                    SubtitleCue(2.0, 3.0, "third"),
                    SubtitleCue(3.0, 4.0, "fourth"),
                ),
            )

    class FakeRefiner:
        model = DEFAULT_REFINEMENT_MODEL

        def refine_transcript(self, transcript: str) -> str:
            return (
                "1\n"
                "00:00:00,000 --> 00:00:01,000\n"
                "first\n\n"
                "2\n"
                "00:00:01,000 --> 00:00:02,000\n"
                "second\n"
                "ترجمه دوم خط اول\n"
                "ترجمه دوم خط دوم\n\n"
                "3\n"
                "00:00:02,000 --> 00:00:03,000\n"
                "third changed by model\n"
                '<font color="green">ترجمه سوم</font>\n\n'
                "4\n"
                "00:00:03,000 --> 00:00:04,000\n"
                "fourth\n"
                "extra model note\n"
                '<font color="green">ترجمه چهارم</font>\n'
            )

    transcriber = SpeechTranscriber(
        speech_to_text_provider=FakeSpeechToTextProvider(),
        refiner=FakeRefiner(),
    )

    result = await transcriber.transcribe_chunks_async([audio])

    assert result.translated_srt == (
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "first\n\n"
        "2\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "second\n"
        "ترجمه دوم خط اول\n"
        "ترجمه دوم خط دوم\n\n"
        "3\n"
        "00:00:02,000 --> 00:00:03,000\n"
        "third changed by model\n"
        '<font color="green">ترجمه سوم</font>\n\n'
        "4\n"
        "00:00:03,000 --> 00:00:04,000\n"
        "fourth\n"
        "extra model note\n"
        '<font color="green">ترجمه چهارم</font>\n'
    )
    assert result.line_translated_transcript == (
        "first\n\n"
        "second\n"
        "ترجمه دوم خط اول\n"
        "ترجمه دوم خط دوم\n\n"
        "third changed by model\n"
        "ترجمه سوم\n\n"
        "fourth\n"
        "extra model note\n"
        "ترجمه چهارم\n"
    )


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


def test_transcript_refiner_accepts_empty_response_text() -> None:
    class FakeResponses:
        def create(self, *, model: str, instructions: str, input: str, temperature: int) -> object:
            return SimpleNamespace(output_text=" ")

    fake_client = SimpleNamespace(responses=FakeResponses())
    refiner = TranscriptRefiner(api_key="key", client=fake_client)

    assert refiner.refine_transcript("1\n00:00:00,000 --> 00:00:01,000\nهاي\n") == ""


def test_refinement_input_uses_delimited_raw_transcript() -> None:
    raw = "1\n00:00:00,000 --> 00:00:01,000\nهاي تجربة\n"

    refinement_input = build_refinement_input(raw)

    assert SRT_TRANSLATION_REQUEST in refinement_input
    assert RAW_TRANSCRIPT_START in refinement_input
    assert RAW_TRANSCRIPT_END in refinement_input
    assert raw in refinement_input


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


def test_split_srt_by_byte_limit_preserves_complete_blocks() -> None:
    srt = render_srt(
        (
            SubtitleCue(0.0, 1.0, "first"),
            SubtitleCue(1.0, 2.0, "second"),
            SubtitleCue(2.0, 3.0, "third"),
        )
    )

    chunks = split_srt_by_byte_limit(srt, max_bytes=75)

    assert len(chunks) == 3
    assert all(len(chunk.encode("utf-8")) <= 75 for chunk in chunks)
    assert assemble_srt_chunks(chunks) == srt
    assert [block.index for chunk in chunks for block in parse_srt_blocks(chunk)] == ["1", "2", "3"]


def test_split_srt_by_byte_limit_keeps_oversized_single_block_intact() -> None:
    srt = render_srt((SubtitleCue(0.0, 1.0, "x" * 100),))

    chunks = split_srt_by_byte_limit(srt, max_bytes=40)

    assert chunks == (srt,)
    assert len(chunks[0].encode("utf-8")) > 40
    assert assemble_srt_chunks(chunks) == srt


def test_assemble_srt_chunks_reassembles_through_parsed_blocks() -> None:
    first = "1\n00:00:00,000 --> 00:00:01,000\nهاي\n<font color=\"green\">سلام</font>\n"
    second = "2\n00:00:01,000 --> 00:00:02,000\nشلونك\n<font color=\"green\">چطوری</font>\n"

    assert assemble_srt_chunks((first, second)) == (
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "هاي\n"
        '<font color="green">سلام</font>\n\n'
        "2\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "شلونك\n"
        '<font color="green">چطوری</font>\n'
    )


def test_render_line_translated_transcript_from_srt_removes_srt_structure_and_font_markup() -> None:
    translated_srt = (
        "1\n"
        "00:00:00,001 --> 00:00:01,000\n"
        "هلا بالعالم\n"
        '<font color="green">سلام دنیا</font>\n\n'
        "2\n"
        "00:00:01,500 --> 00:00:02,000\n"
        "شلونك؟\n"
        '<font color="green">حالت چطوره؟</font>\n'
    )

    assert render_line_translated_transcript_from_srt(translated_srt) == (
        "هلا بالعالم\nسلام دنیا\n\nشلونك؟\nحالت چطوره؟\n"
    )


def test_render_line_translated_transcript_from_srt_keeps_plain_translation_text() -> None:
    translated_srt = "1\n00:00:00,001 --> 00:00:01,000\nهلا بالعالم\nسلام دنیا\n"

    assert render_line_translated_transcript_from_srt(translated_srt) == "هلا بالعالم\nسلام دنیا\n"


def test_transcript_refiner_raises_when_refinement_has_no_text() -> None:
    class FakeResponses:
        def create(self, *, model: str, instructions: str, input: str, temperature: int) -> object:
            return SimpleNamespace()

    fake_client = SimpleNamespace(responses=FakeResponses())
    refiner = TranscriptRefiner(api_key="key", client=fake_client)
    with pytest.raises(TranscriptionError, match="refinement response"):
        refiner.refine_transcript("1\n00:00:00,000 --> 00:00:01,000\nهاي\n")
