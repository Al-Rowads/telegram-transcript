from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import telegram_transcript.transcriber as transcriber_module
from telegram_transcript.transcriber import (
    DEFAULT_DEEPGRAM_LANGUAGE,
    DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL,
    DEFAULT_GEMINI_TRANSCRIPTION_MODEL,
    DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
    DEFAULT_REFINEMENT_MODEL,
    DeepgramSpeechToTextProvider,
    LowConfidenceTranscriptCorrector,
    GeminiSpeechToTextProvider,
    OpenAISpeechToTextProvider,
    CUE_TRANSLATION_END,
    CUE_TRANSLATION_START,
    SRT_TRANSLATION_REQUEST,
    SRT_TRANSLATION_SYSTEM_PROMPT,
    SRT_TRANSLATION_COHESIVE_SYSTEM_PROMPT,
    SpeechTranscriber,
    TranscriptRefiner,
    TranscriptionError,
    assemble_srt_chunks,
    build_refinement_input,
    extract_deepgram_file_transcription_result,
    extract_deepgram_transcript_text,
    extract_deepgram_transcript_words,
    merge_overlapping_subtitle_cues,
    parse_srt_batch_translation_response,
    parse_srt_cue_translation_response,
    parse_srt_blocks,
    render_line_translated_transcript_from_srt,
    render_srt,
    render_translated_srt_block,
    split_srt_by_byte_limit,
)
from telegram_transcript.models import AudioChunk, FileTranscriptionResult, SubtitleCue, TranscriptWord


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


def test_extract_deepgram_transcript_words_preserves_timing_and_confidence() -> None:
    response = {
        "results": {
            "channels": [{"alternatives": [{"words": [
                {"word": "شلونك", "punctuated_word": "شلونك؟", "start": 1.0, "end": 1.5, "confidence": 0.61}
            ]}]}]
        }
    }

    assert extract_deepgram_transcript_words(response) == (
        TranscriptWord(1.0, 1.5, "شلونك؟", 0.61),
    )


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


def test_deepgram_provider_passes_deduplicated_keyterms(tmp_path: Path) -> None:
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    calls: list[dict[str, object]] = []

    class FakeMedia:
        def transcribe_file(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return {"results": {"channels": [{"alternatives": [{"transcript": "نص"}]}]}}

    provider = DeepgramSpeechToTextProvider(
        api_key="key",
        keyterms=("اسم", "مصطلح", "اسم"),
        client=SimpleNamespace(listen=SimpleNamespace(v1=SimpleNamespace(media=FakeMedia()))),
    )

    provider.transcribe_file(audio)

    assert calls[0]["language"] == "ar-IQ"
    assert calls[0]["keyterm"] == ("اسم", "مصطلح")


def test_openai_provider_uses_openrouter_transcription_endpoint(tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return

        def json(self) -> dict[str, str]:
            return {"text": "هلا شلونك"}

    class FakeSession:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def post(self, url: str, **kwargs: object) -> FakeResponse:
            self.calls.append({"url": url, **kwargs})
            return FakeResponse()

    fake_session = FakeSession()
    provider = OpenAISpeechToTextProvider(api_key="key", client=fake_session)

    result = provider.transcribe_file_result(audio)

    assert result.transcript == "هلا شلونك"
    assert result.subtitle_cues == ()
    assert fake_session.calls[0]["url"].endswith("/audio/transcriptions")
    assert fake_session.calls[0]["headers"]["Authorization"] == "Bearer key"
    assert fake_session.calls[0]["json"] == {
        "input_audio": {"data": "YXVkaW8=", "format": "mp3"},
        "model": DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
        "language": "ar",
        "temperature": 0,
    }


def test_openai_provider_rejects_missing_text(tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return

        def json(self) -> dict[str, str]:
            return {"text": ""}

    fake_client = SimpleNamespace(post=lambda *args, **kwargs: FakeResponse())
    provider = OpenAISpeechToTextProvider(api_key="key", client=fake_client)

    with pytest.raises(TranscriptionError, match="OpenRouter transcription"):
        provider.transcribe_file_result(audio)


def test_gemini_provider_requests_srt_and_parses_response(tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")

    class FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=(
                    "```srt\n"
                    "1\n"
                    "00:00:00,000 --> 00:00:01,250\n"
                    "هلا\n\n"
                    "2\n"
                    "00:00:01,250 --> 00:00:02,000\n"
                    "شلونك\n"
                    "```"
                )))]
            )

    fake_completions = FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))
    provider = GeminiSpeechToTextProvider(api_key="key", client=fake_client)

    result = provider.transcribe_file_result(audio)

    assert result.transcript == "هلا\n\nشلونك"
    assert result.subtitle_cues == (
        SubtitleCue(0.0, 1.25, "هلا"),
        SubtitleCue(1.25, 2.0, "شلونك"),
    )
    assert fake_completions.calls[0]["model"] == DEFAULT_GEMINI_TRANSCRIPTION_MODEL
    content = fake_completions.calls[0]["messages"][0]["content"]
    assert "SRT" in content[0]["text"]
    assert content[1]["input_audio"] == {"data": "YXVkaW8=", "format": "mp3"}


def test_gemini_provider_rejects_invalid_srt(tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")

    class FakeCompletions:
        def create(self, **_: object) -> object:
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="plain transcript"))])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
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
            self.calls: list[object] = []

        def translate_srt_block(self, block: object, previous_context: object = ()) -> str:
            self.calls.append(block)
            return "اول" if getattr(block, "index") == "1" else "دوم"

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
    assert [getattr(block, "index") for block in fake_refiner.calls] == ["1", "2"]
    assert [getattr(block, "timestamp") for block in fake_refiner.calls] == [
        "00:00:00,500 --> 00:00:01,000",
        "00:00:10,500 --> 00:00:11,000",
    ]
    assert [event for event, _ in progress_events] == [
        "transcribing_chunk",
        "chunk_transcribed",
        "transcribing_chunk",
        "chunk_transcribed",
        "refining_transcript",
        "refining_transcript",
        "refinement_complete",
    ]
    assert progress_events[0][1]["index"] == 1
    assert progress_events[0][1]["provider"] == "deepgram"
    assert progress_events[2][1]["index"] == 2
    assert progress_events[4][1]["model"] == DEFAULT_REFINEMENT_MODEL
    assert progress_events[4][1]["index"] == 1
    assert progress_events[5][1]["index"] == 2
    assert progress_events[6][1]["line_translated_transcript_chars"] == len("first\nاول\n\nsecond\nدوم\n")


@pytest.mark.asyncio
async def test_transcribe_chunks_async_translates_each_srt_cue(
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
            self.calls: list[object] = []

        def translate_srt_block(self, block: object, previous_context: object = ()) -> str:
            self.calls.append(block)
            return f"ترجمه {getattr(block, 'index')}"

    fake_refiner = FakeRefiner()
    transcriber = SpeechTranscriber(
        speech_to_text_provider=FakeSpeechToTextProvider(),
        refiner=fake_refiner,
    )
    progress_events: list[tuple[str, dict[str, object]]] = []

    async def record_progress(event: str, data: object) -> None:
        progress_events.append((event, dict(data)))

    result = await transcriber.transcribe_chunks_async([audio], progress_callback=record_progress)

    assert len(fake_refiner.calls) == 3
    assert [getattr(block, "text_lines") for block in fake_refiner.calls] == [
        ("first",),
        ("second",),
        ("third",),
    ]
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
async def test_transcribe_chunks_async_collapses_multiline_persian_translation(
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
                transcript="first second",
                subtitle_cues=(
                    SubtitleCue(0.0, 1.0, "first\nsecond"),
                ),
            )

    class FakeRefiner:
        model = DEFAULT_REFINEMENT_MODEL

        def translate_srt_block(self, block: object, previous_context: object = ()) -> str:
            return "ترجمه خط اول\nترجمه   خط دوم"

    transcriber = SpeechTranscriber(
        speech_to_text_provider=FakeSpeechToTextProvider(),
        refiner=FakeRefiner(),
    )

    result = await transcriber.transcribe_chunks_async([audio])

    assert result.translated_srt == (
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "first\n"
        "second\n"
        '<font color="green">ترجمه خط اول ترجمه خط دوم</font>\n'
    )
    assert result.line_translated_transcript == (
        "first\n"
        "second\n"
        "ترجمه خط اول ترجمه خط دوم\n"
    )


def test_transcript_refiner_uses_structured_one_cue_translation_request() -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=(
                    '{"translations":[{"index":"1","translation":"سلام"}]}'
                )))]
            )

    fake_completions = FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))
    refiner = TranscriptRefiner(api_key="key", client=fake_client)

    assert refiner.refine_transcript("1\n00:00:00,000 --> 00:00:01,000\nهاي\n") == (
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "هاي\n"
        '<font color="green">سلام</font>\n'
    )
    call = fake_completions.calls[0]
    assert call["model"] == DEFAULT_REFINEMENT_MODEL
    assert call["messages"][0] == {"role": "system", "content": SRT_TRANSLATION_SYSTEM_PROMPT}
    assert call["messages"][1]["content"] == build_refinement_input(
        "1\n00:00:00,000 --> 00:00:01,000\nهاي\n"
    )
    assert call["temperature"] == 0
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["strict"] is True
    assert call["provider"] == {"require_parameters": True}


def test_cohesive_refiner_includes_only_supplied_previous_context() -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=(
                    '{"translations":[{"index":"7","translation":"هفتم"}]}'
                )))]
            )

    completions = FakeCompletions()
    refiner = TranscriptRefiner(
        api_key="key",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        system_prompt=SRT_TRANSLATION_COHESIVE_SYSTEM_PROMPT,
        prompt_key="v2",
    )
    blocks = tuple(
        parse_srt_blocks(f"{index}\n00:00:0{index},000 --> 00:00:0{index},500\nArabic {index}\n")[0]
        for index in range(1, 8)
    )

    refiner.translate_srt_block(
        blocks[-1],
        tuple((block, f"Persian {block.index}") for block in blocks[1:6]),
    )

    messages = completions.calls[0]["messages"]
    assert messages[0]["content"] == SRT_TRANSLATION_COHESIVE_SYSTEM_PROMPT
    user_content = messages[1]["content"]
    assert "Arabic 1" not in user_content
    for index in range(2, 7):
        assert f"Arabic {index}" in user_content
        assert f"Persian {index}" in user_content
    assert "Arabic 7" in user_content


def test_transcript_refiner_rejects_empty_translation() -> None:
    class FakeCompletions:
        def create(self, **_: object) -> object:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=(
                    '{"translations":[{"index":"1","translation":" "}]}'
                )))]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    refiner = TranscriptRefiner(api_key="key", client=fake_client)

    with pytest.raises(TranscriptionError, match="empty"):
        refiner.refine_transcript("1\n00:00:00,000 --> 00:00:01,000\nهاي\n")


def test_refinement_input_uses_delimited_raw_transcript() -> None:
    raw = "1\n00:00:00,000 --> 00:00:01,000\nهاي تجربة\n"

    refinement_input = build_refinement_input(raw)

    assert SRT_TRANSLATION_REQUEST in refinement_input
    assert "<target_cues>" in refinement_input
    assert "</target_cues>" in refinement_input
    assert raw.strip() in refinement_input


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


def test_render_translated_srt_block_preserves_original_cue_and_adds_one_persian_line() -> None:
    block = parse_srt_blocks("7\n00:00:01,000 --> 00:00:02,000\nline one\nline two\n")[0]

    assert render_translated_srt_block(block, " ترجمه اول\nترجمه دوم ") == (
        "7\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "line one\n"
        "line two\n"
        '<font color="green">ترجمه اول ترجمه دوم</font>\n'
    )


def test_parse_srt_cue_translation_response_rejects_malformed_json() -> None:
    with pytest.raises(TranscriptionError, match="valid JSON"):
        parse_srt_cue_translation_response("not json")


def test_parse_srt_cue_translation_response_rejects_missing_translation() -> None:
    with pytest.raises(TranscriptionError, match="translation string"):
        parse_srt_cue_translation_response('{"text": "سلام"}')


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
    class FakeCompletions:
        def create(self, **_: object) -> object:
            return SimpleNamespace()

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    refiner = TranscriptRefiner(api_key="key", client=fake_client)
    with pytest.raises(TranscriptionError, match="completion choice"):
        refiner.refine_transcript("1\n00:00:00,000 --> 00:00:01,000\nهاي\n")


def test_low_confidence_corrector_rechecks_only_flagged_cues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "audio.flac"
    audio_path.write_bytes(b"audio")
    extracted_windows: list[tuple[float, float]] = []

    def fake_extract_window(
        source: Path,
        target: Path,
        *,
        start_seconds: float,
        end_seconds: float,
    ) -> Path:
        assert source == audio_path
        extracted_windows.append((start_seconds, end_seconds))
        target.write_bytes(b"window")
        return target

    class FakeSecondary:
        def transcribe_candidate(self, audio: Path, **kwargs: str) -> str:
            assert audio.suffix == ".flac"
            assert kwargs["primary_text"] == "غلط"
            return "صحيح"

    class FakeResolver:
        def resolve(self, **kwargs: str) -> str:
            assert kwargs["primary_text"] == "غلط"
            assert kwargs["secondary_text"] == "صحيح"
            return kwargs["secondary_text"]

    monkeypatch.setattr(transcriber_module, "extract_audio_window", fake_extract_window)
    corrector = LowConfidenceTranscriptCorrector(
        secondary_provider=FakeSecondary(),
        resolver=FakeResolver(),
    )
    result = FileTranscriptionResult(
        transcript="غلط\n\nثابت",
        subtitle_cues=(
            SubtitleCue(2.0, 3.0, "غلط"),
            SubtitleCue(4.0, 5.0, "ثابت"),
        ),
        words=(
            TranscriptWord(2.1, 2.8, "غلط", 0.4),
            TranscriptWord(4.1, 4.8, "ثابت", 0.99),
        ),
    )

    corrected = corrector.correct_file_result(audio_path, result)

    assert extracted_windows == [(0.5, 4.5)]
    assert tuple(cue.text for cue in corrected.subtitle_cues) == ("صحيح", "ثابت")
    assert corrected.transcript == "صحيح\n\nثابت"


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ('{"translations":[]}', "omitted"),
        (
            '{"translations":[{"index":"1","translation":"الف"},'
            '{"index":"1","translation":"ب"}]}',
            "duplicate",
        ),
        ('{"translations":[{"index":"9","translation":"الف"}]}', "unknown"),
        ('{"translations":[{"index":"1","translation":"الف\\nب"}]}', "multiline"),
    ],
)
def test_parse_srt_batch_translation_response_rejects_invalid_results(response: str, message: str) -> None:
    with pytest.raises(TranscriptionError, match=message):
        parse_srt_batch_translation_response(response, expected_indexes=("1",))


def test_batch_translation_retries_then_bisects_malformed_response() -> None:
    responses = iter(
        [
            "not json",
            "not json",
            '{"translations":[{"index":"1","translation":"اول"}]}',
            '{"translations":[{"index":"2","translation":"دوم"}]}',
        ]
    )

    class FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))]
            )

    completions = FakeCompletions()
    refiner = TranscriptRefiner(
        api_key="key",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    blocks = parse_srt_blocks(
        "1\n00:00:00,000 --> 00:00:01,000\nاول\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nدوم\n"
    )

    assert refiner.translate_srt_blocks(blocks) == ("اول", "دوم")
    assert len(completions.calls) == 4


@pytest.mark.asyncio
async def test_transcribe_chunks_translates_twelve_cue_batches_with_bidirectional_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_inline(func: object, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)
    audio_path = tmp_path / "audio.flac"
    audio_path.write_bytes(b"audio")

    class FakeProvider:
        provider_name = "deepgram"
        model = "nova-3"

        def transcribe_file_result(self, audio: Path, *, previous_transcript: str = "") -> FileTranscriptionResult:
            cues = tuple(SubtitleCue(float(index), float(index + 1), f"Arabic {index + 1}") for index in range(13))
            return FileTranscriptionResult(transcript="raw", subtitle_cues=cues)

    class FakeRefiner:
        model = "translation-model"

        def __init__(self) -> None:
            self.calls: list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = []

        def translate_srt_blocks(
            self,
            blocks: object,
            *,
            previous_context: object = (),
            following_context: object = (),
        ) -> tuple[str, ...]:
            target = tuple(block.index for block in blocks)
            previous = tuple(block.index for block, _ in previous_context)
            following = tuple(block.index for block in following_context)
            self.calls.append((target, previous, following))
            return tuple(f"Persian {index}" for index in target)

    refiner = FakeRefiner()
    result = await SpeechTranscriber(
        speech_to_text_provider=FakeProvider(),
        refiner=refiner,
    ).transcribe_chunks_async((audio_path,))

    assert refiner.calls[0] == (tuple(str(index) for index in range(1, 13)), (), ("13",))
    assert refiner.calls[1] == (("13",), ("8", "9", "10", "11", "12"), ())
    assert result.subtitle_cues[0].start_seconds == 0.0
    assert result.subtitle_cues[-1].end_seconds == 13.0


def test_merge_overlapping_subtitle_cues_deduplicates_chunk_overlap() -> None:
    existing = (SubtitleCue(9.0, 10.5, "نفس الكلام"),)
    incoming = (
        SubtitleCue(9.5, 10.8, "نفس الكلام"),
        SubtitleCue(10.8, 11.5, "كلام جديد"),
    )

    assert merge_overlapping_subtitle_cues(existing, incoming) == (
        existing[0],
        incoming[1],
    )
