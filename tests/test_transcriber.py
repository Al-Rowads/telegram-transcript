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
    DEFAULT_TRANSCRIPTION_REFINEMENT_MODEL,
    DEFAULT_WHISPER_LARGE_V3_TRANSCRIPTION_MODEL,
    CORRECTION_FAILURE_WARNING,
    IRAQI_ARABIC_TRANSCRIPTION_REFINEMENT_SYSTEM_PROMPT,
    MAX_SRT_REFINEMENT_CHUNK_BYTES,
    TRANSCRIPTION_REFINEMENT_FAILURE_WARNING,
    TRANSCRIPTION_REFINEMENT_PLACEHOLDER,
    TRANSLATION_FAILURE_WARNING,
    DeepgramSpeechToTextProvider,
    GeminiAudioCorrectionProvider,
    LowConfidenceTranscriptCorrector,
    GeminiSpeechToTextProvider,
    IraqiArabicTranscriptRefiner,
    OpenAISpeechToTextProvider,
    OpenRouterWhisperSpeechToTextProvider,
    CUE_TRANSLATION_END,
    CUE_TRANSLATION_START,
    SRT_TRANSLATION_REQUEST,
    SRT_TRANSLATION_SYSTEM_PROMPT,
    SRT_TRANSLATION_COHESIVE_SYSTEM_PROMPT,
    SpeechTranscriber,
    TranscriptCandidateResolver,
    TranscriptRefiner,
    TranscriptionError,
    assemble_srt_chunks,
    build_transcription_refinement_system_prompt,
    build_refinement_input,
    extract_deepgram_file_transcription_result,
    extract_deepgram_transcript_text,
    extract_deepgram_transcript_words,
    extract_openai_file_transcription_result,
    merge_overlapping_subtitle_cues,
    parse_srt_batch_translation_response,
    parse_srt_cue_translation_response,
    parse_srt_blocks,
    render_line_translated_transcript_from_srt,
    render_srt,
    render_translated_srt_block,
    split_srt_by_byte_limit,
    split_text_by_utf8_byte_limit,
    validate_refined_srt_blocks,
    validate_file_transcription_result,
)
from telegram_transcript.models import AudioChunk, FileTranscriptionResult, SubtitleCue, TranscriptWord


@pytest.fixture(autouse=True)
def stub_audio_duration_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcriber_module, "probe_audio_duration_seconds", lambda _: 3600.0)


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
            request_options: dict[str, object],
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
                    "request_options": request_options,
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
            "request_options": {
                "timeout_in_seconds": 300,
                "max_retries": 2,
            },
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


def test_openai_provider_uses_direct_verbose_transcription_with_segments(tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")

    class FakeTranscriptions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            audio_file = kwargs.pop("file")
            self.calls.append({"file": audio_file.read(), **kwargs})
            return SimpleNamespace(
                text="هلا شلونك",
                segments=[SimpleNamespace(start=0.0, end=1.5, text="هلا شلونك")],
            )

    fake_transcriptions = FakeTranscriptions()
    fake_client = SimpleNamespace(audio=SimpleNamespace(transcriptions=fake_transcriptions))
    provider = OpenAISpeechToTextProvider(api_key="key", client=fake_client)

    result = provider.transcribe_file_result(audio)

    assert result.transcript == "هلا شلونك"
    assert result.subtitle_cues == (SubtitleCue(0.0, 1.5, "هلا شلونك"),)
    assert fake_transcriptions.calls == [{
        "file": b"audio",
        "model": DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
        "language": "ar",
        "temperature": 0,
        "response_format": "verbose_json",
        "timestamp_granularities": ["segment"],
    }]


def test_openai_provider_preserves_empty_response_for_silence_detection(tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")

    fake_transcriptions = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(text="", segments=[]))
    fake_client = SimpleNamespace(audio=SimpleNamespace(transcriptions=fake_transcriptions))
    provider = OpenAISpeechToTextProvider(api_key="key", client=fake_client)

    assert provider.transcribe_file_result(audio) == FileTranscriptionResult(transcript="")


def test_openrouter_whisper_provider_requests_timestamped_segments(tmp_path: Path) -> None:
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")

    class FakeTranscriptions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            audio_file = kwargs.pop("file")
            self.calls.append({"file": audio_file.read(), **kwargs})
            return SimpleNamespace(
                text="هلا شلونك",
                segments=[SimpleNamespace(start=0.0, end=1.5, text="هلا شلونك")],
            )

    transcriptions = FakeTranscriptions()
    provider = OpenRouterWhisperSpeechToTextProvider(
        api_key="key",
        client=SimpleNamespace(audio=SimpleNamespace(transcriptions=transcriptions)),
    )

    result = provider.transcribe_file_result(audio, previous_transcript="النص السابق")

    assert provider.provider_name == "whisper"
    assert result == FileTranscriptionResult(
        transcript="هلا شلونك",
        subtitle_cues=(SubtitleCue(0.0, 1.5, "هلا شلونك"),),
    )
    assert transcriptions.calls == [{
        "file": b"audio",
        "model": DEFAULT_WHISPER_LARGE_V3_TRANSCRIPTION_MODEL,
        "language": "ar",
        "temperature": 0,
        "response_format": "verbose_json",
        "timestamp_granularities": ["segment"],
        "prompt": "The audio is Iraqi Arabic. The immediately preceding transcript was: النص السابق",
        "extra_body": {"provider": {"require_parameters": True}},
    }]


def test_openrouter_whisper_provider_preserves_empty_response_for_silence_detection(tmp_path: Path) -> None:
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    transcriptions = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(text="", segments=[]))
    provider = OpenRouterWhisperSpeechToTextProvider(
        api_key="key",
        client=SimpleNamespace(audio=SimpleNamespace(transcriptions=transcriptions)),
    )

    assert provider.transcribe_file_result(audio) == FileTranscriptionResult(transcript="")


def test_openrouter_whisper_provider_wraps_api_errors(tmp_path: Path) -> None:
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")

    class FailingTranscriptions:
        def create(self, **kwargs: object) -> object:
            raise transcriber_module.OpenAIError("request failed")

    provider = OpenRouterWhisperSpeechToTextProvider(
        api_key="key",
        client=SimpleNamespace(audio=SimpleNamespace(transcriptions=FailingTranscriptions())),
    )

    with pytest.raises(TranscriptionError, match="OpenRouter Whisper transcription request failed"):
        provider.transcribe_file_result(audio)


@pytest.mark.parametrize(
    "cues",
    [
        (SubtitleCue(-0.1, 1.0, "negative"),),
        (SubtitleCue(1.0, 1.0, "zero duration"),),
        (SubtitleCue(0.0, 13.0, "past tolerance"),),
        (SubtitleCue(2.0, 3.0, "later"), SubtitleCue(1.0, 2.0, "out of order")),
    ],
)
def test_validate_file_transcription_result_rejects_invalid_timestamps(
    cues: tuple[SubtitleCue, ...],
) -> None:
    with pytest.raises(TranscriptionError, match="invalid subtitle timestamps"):
        validate_file_transcription_result(
            FileTranscriptionResult(transcript="spoken", subtitle_cues=cues),
            duration_seconds=10.0,
            provider_name="test",
        )


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


def test_gemini_provider_preserves_empty_response_for_silence_detection(tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")

    class FakeCompletions:
        def create(self, **_: object) -> object:
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    provider = GeminiSpeechToTextProvider(api_key="key", client=fake_client)

    assert provider.transcribe_file_result(audio) == FileTranscriptionResult(transcript="")


def test_gemini_audio_correction_passes_openrouter_routing_in_extra_body(tmp_path: Path) -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"transcription":"صحيح"}'))]
            )

    audio = tmp_path / "window.flac"
    audio.write_bytes(b"audio")
    completions = FakeCompletions()
    provider = GeminiAudioCorrectionProvider(
        api_key="key",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    assert provider.transcribe_candidate(
        audio,
        primary_text="غلط",
        previous_text="قبل",
        following_text="بعد",
    ) == "صحيح"
    assert completions.calls[0]["extra_body"] == {
        "provider": {"require_parameters": True},
    }
    assert "provider" not in completions.calls[0]


def test_candidate_resolver_passes_openrouter_routing_in_extra_body() -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"choice":"secondary"}'))]
            )

    completions = FakeCompletions()
    resolver = TranscriptCandidateResolver(
        api_key="key",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    assert resolver.resolve(
        primary_text="غلط",
        secondary_text="صحيح",
        previous_text="قبل",
        following_text="بعد",
    ) == "صحيح"
    assert completions.calls[0]["extra_body"] == {
        "provider": {"require_parameters": True},
    }
    assert "provider" not in completions.calls[0]


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
        "translating_subtitles",
        "translating_subtitles",
        "translation_complete",
    ]
    assert progress_events[0][1]["index"] == 1
    assert progress_events[0][1]["provider"] == "deepgram"
    assert progress_events[2][1]["index"] == 2
    assert progress_events[4][1]["model"] == DEFAULT_REFINEMENT_MODEL
    assert progress_events[4][1]["index"] == 1
    assert progress_events[5][1]["index"] == 2
    assert progress_events[6][1]["line_translated_transcript_chars"] == len("first\nاول\n\nsecond\nدوم\n")


@pytest.mark.asyncio
async def test_transcribe_chunks_falls_back_only_for_failed_chunk_and_restarts_primary_next_chunk(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.flac"
    second = tmp_path / "second.flac"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    class PrimaryProvider:
        provider_name = "gemini"
        model = "gemini-model"

        def __init__(self) -> None:
            self.calls: list[str] = []

        def transcribe_file_result(self, audio_path: Path, *, previous_transcript: str = "") -> FileTranscriptionResult:
            self.calls.append(audio_path.stem)
            if audio_path == first:
                raise TranscriptionError("Gemini response did not include SRT cues.")
            return FileTranscriptionResult(
                transcript="primary second",
                subtitle_cues=(SubtitleCue(0.0, 1.0, "primary second"),),
            )

    class FallbackProvider:
        provider_name = "deepgram"
        model = "deepgram-model"

        def __init__(self) -> None:
            self.calls: list[str] = []

        def transcribe_file_result(self, audio_path: Path, *, previous_transcript: str = "") -> FileTranscriptionResult:
            self.calls.append(audio_path.stem)
            return FileTranscriptionResult(
                transcript="fallback first",
                subtitle_cues=(SubtitleCue(0.0, 1.0, "fallback first"),),
            )

    class UnusedProvider:
        provider_name = "openai"
        model = "whisper-1"

        def __init__(self) -> None:
            self.calls: list[str] = []

        def transcribe_file_result(self, audio_path: Path, *, previous_transcript: str = "") -> FileTranscriptionResult:
            self.calls.append(audio_path.stem)
            raise AssertionError("OpenAI should not be needed")

    primary = PrimaryProvider()
    fallback = FallbackProvider()
    unused = UnusedProvider()
    events: list[tuple[str, dict[str, object]]] = []

    async def record_progress(event: str, data: object) -> None:
        events.append((event, dict(data)))

    result = await SpeechTranscriber(
        speech_to_text_provider=primary,
        fallback_speech_to_text_providers=(fallback, unused),
    ).transcribe_chunks_async(
        (
            AudioChunk(first, 0.0, 5.0),
            AudioChunk(second, 10.0, 5.0),
        ),
        progress_callback=record_progress,
    )

    assert result.raw_transcript == "fallback first\n\nprimary second"
    assert result.subtitle_cues == (
        SubtitleCue(0.0, 1.0, "fallback first"),
        SubtitleCue(10.0, 11.0, "primary second"),
    )
    assert primary.calls == ["first", "second"]
    assert fallback.calls == ["first"]
    assert unused.calls == []
    fallback_event = next(data for event, data in events if event == "provider_fallback")
    assert fallback_event["failed_provider"] == "gemini"
    assert fallback_event["next_provider"] == "deepgram"


@pytest.mark.asyncio
async def test_transcribe_chunks_treats_all_empty_provider_results_as_silence(tmp_path: Path) -> None:
    audio = tmp_path / "silence.flac"
    audio.write_bytes(b"silence")

    class EmptyProvider:
        model = "model"

        def __init__(self, provider_name: str) -> None:
            self.provider_name = provider_name

        def transcribe_file_result(self, audio_path: Path, *, previous_transcript: str = "") -> FileTranscriptionResult:
            return FileTranscriptionResult(transcript="")

    providers = tuple(EmptyProvider(name) for name in ("gemini", "deepgram", "whisper", "openai"))
    result = await SpeechTranscriber(
        speech_to_text_provider=providers[0],
        fallback_speech_to_text_providers=providers[1:],
    ).transcribe_chunks_async((AudioChunk(audio, duration_seconds=10.0),))

    assert result.raw_transcript == ""
    assert result.subtitle_cues == ()


@pytest.mark.asyncio
async def test_transcribe_chunks_rejects_mixed_errors_and_empty_results(tmp_path: Path) -> None:
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")

    class Provider:
        model = "model"

        def __init__(self, provider_name: str, *, fail: bool = False) -> None:
            self.provider_name = provider_name
            self.fail = fail

        def transcribe_file_result(self, audio_path: Path, *, previous_transcript: str = "") -> FileTranscriptionResult:
            if self.fail:
                raise TranscriptionError("request failed")
            return FileTranscriptionResult(transcript="")

    providers = (
        Provider("gemini", fail=True),
        Provider("deepgram"),
        Provider("whisper"),
        Provider("openai"),
    )

    with pytest.raises(TranscriptionError, match="All transcription providers failed"):
        await SpeechTranscriber(
            speech_to_text_provider=providers[0],
            fallback_speech_to_text_providers=providers[1:],
        ).transcribe_chunks_async((AudioChunk(audio, duration_seconds=10.0),))


@pytest.mark.asyncio
async def test_transcribe_chunks_falls_back_when_provider_returns_text_without_timestamps(tmp_path: Path) -> None:
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")

    class TextOnlyProvider:
        provider_name = "gemini"
        model = "gemini-model"

        def transcribe_file_result(self, audio_path: Path, *, previous_transcript: str = "") -> FileTranscriptionResult:
            return FileTranscriptionResult(transcript="text without timestamps")

    class TimestampProvider:
        provider_name = "deepgram"
        model = "nova-3"

        def transcribe_file_result(self, audio_path: Path, *, previous_transcript: str = "") -> FileTranscriptionResult:
            return FileTranscriptionResult(
                transcript="valid",
                subtitle_cues=(SubtitleCue(0.0, 1.0, "valid"),),
            )

    result = await SpeechTranscriber(
        speech_to_text_provider=TextOnlyProvider(),
        fallback_speech_to_text_providers=(TimestampProvider(),),
    ).transcribe_chunks_async((AudioChunk(audio, duration_seconds=10.0),))

    assert result.raw_transcript == "valid"
    assert result.subtitle_cues == (SubtitleCue(0.0, 1.0, "valid"),)


@pytest.mark.asyncio
async def test_transcribe_chunks_retries_openai_after_whisper_returns_no_timestamps(tmp_path: Path) -> None:
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")
    calls: list[str] = []

    class WhisperProvider:
        provider_name = "whisper"
        model = DEFAULT_WHISPER_LARGE_V3_TRANSCRIPTION_MODEL

        def transcribe_file_result(self, audio_path: Path, *, previous_transcript: str = "") -> FileTranscriptionResult:
            calls.append(self.provider_name)
            return FileTranscriptionResult(transcript="text without timestamps")

    class OpenAIProvider:
        provider_name = "openai"
        model = DEFAULT_OPENAI_TRANSCRIPTION_MODEL

        def transcribe_file_result(self, audio_path: Path, *, previous_transcript: str = "") -> FileTranscriptionResult:
            calls.append(self.provider_name)
            return FileTranscriptionResult(
                transcript="valid",
                subtitle_cues=(SubtitleCue(0.0, 1.0, "valid"),),
            )

    events: list[tuple[str, dict[str, object]]] = []

    async def record_progress(event: str, data: object) -> None:
        events.append((event, dict(data)))

    result = await SpeechTranscriber(
        speech_to_text_provider=WhisperProvider(),
        fallback_speech_to_text_providers=(OpenAIProvider(),),
    ).transcribe_chunks_async(
        (AudioChunk(audio, duration_seconds=10.0),),
        progress_callback=record_progress,
    )

    assert calls == ["whisper", "openai"]
    assert result.subtitle_cues == (SubtitleCue(0.0, 1.0, "valid"),)
    fallback_event = next(data for event, data in events if event == "provider_fallback")
    assert fallback_event["failed_provider"] == "whisper"
    assert fallback_event["next_provider"] == "openai"


@pytest.mark.asyncio
async def test_transcribe_chunks_stops_after_valid_whisper_srt(tmp_path: Path) -> None:
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")

    class WhisperProvider:
        provider_name = "whisper"
        model = DEFAULT_WHISPER_LARGE_V3_TRANSCRIPTION_MODEL

        def transcribe_file_result(self, audio_path: Path, *, previous_transcript: str = "") -> FileTranscriptionResult:
            return FileTranscriptionResult(
                transcript="valid",
                subtitle_cues=(SubtitleCue(0.0, 1.0, "valid"),),
            )

    class UnusedOpenAIProvider:
        provider_name = "openai"
        model = DEFAULT_OPENAI_TRANSCRIPTION_MODEL

        def transcribe_file_result(self, audio_path: Path, *, previous_transcript: str = "") -> FileTranscriptionResult:
            raise AssertionError("OpenAI should not be called after valid Whisper timestamps")

    result = await SpeechTranscriber(
        speech_to_text_provider=WhisperProvider(),
        fallback_speech_to_text_providers=(UnusedOpenAIProvider(),),
    ).transcribe_chunks_async((AudioChunk(audio, duration_seconds=10.0),))

    assert result.subtitle_cues == (SubtitleCue(0.0, 1.0, "valid"),)


@pytest.mark.asyncio
async def test_transcribe_chunks_refines_before_translation(tmp_path: Path) -> None:
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")

    class Provider:
        provider_name = "gemini"
        model = "gemini-model"

        def transcribe_file_result(self, audio_path: Path, *, previous_transcript: str = "") -> FileTranscriptionResult:
            return FileTranscriptionResult(
                transcript="قال أقدر",
                subtitle_cues=(SubtitleCue(0.0, 1.0, "قال أقدر"),),
            )

    class ArabicRefiner:
        model = "arabic-refinement-model"

        def refine_srt(self, srt: str) -> str:
            assert "قال أقدر" in srt
            return srt.replace("قال أقدر", "كال أكدر")

    class Translator:
        model = "translation-model"

        def __init__(self) -> None:
            self.source_texts: list[tuple[str, ...]] = []

        def translate_srt_blocks(
            self,
            blocks: object,
            *,
            previous_context: object = (),
            following_context: object = (),
        ) -> tuple[str, ...]:
            del previous_context, following_context
            self.source_texts.append(tuple(" ".join(block.text_lines) for block in blocks))
            return ("گفت می‌توانم",)

    translator = Translator()
    events: list[str] = []

    async def record_progress(event: str, data: object) -> None:
        del data
        events.append(event)

    result = await SpeechTranscriber(
        speech_to_text_provider=Provider(),
        transcription_refiner=ArabicRefiner(),
        refiner=translator,
    ).transcribe_chunks_async(
        (AudioChunk(audio, duration_seconds=10.0),),
        progress_callback=record_progress,
    )

    assert result.raw_transcript == "قال أقدر"
    assert result.refined_transcript == "كال أكدر"
    assert result.final_transcript == "كال أكدر"
    assert result.subtitle_cues == (SubtitleCue(0.0, 1.0, "كال أكدر"),)
    assert translator.source_texts == [("كال أكدر",)]
    assert result.translated_srt is not None and "كال أكدر" in result.translated_srt
    assert events == [
        "transcribing_chunk",
        "chunk_transcribed",
        "refining_transcription",
        "transcription_refinement_complete",
        "translating_subtitles",
        "translation_complete",
    ]


@pytest.mark.asyncio
async def test_transcribe_chunks_refines_when_translation_is_disabled(tmp_path: Path) -> None:
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")

    class Provider:
        provider_name = "gemini"
        model = "gemini-model"

        def transcribe_file_result(self, audio_path: Path, *, previous_transcript: str = "") -> FileTranscriptionResult:
            return FileTranscriptionResult(
                transcript="قال",
                subtitle_cues=(SubtitleCue(0.0, 1.0, "قال"),),
            )

    class ArabicRefiner:
        model = "arabic-refinement-model"

        def refine_srt(self, srt: str) -> str:
            return srt.replace("قال", "كال")

    result = await SpeechTranscriber(
        speech_to_text_provider=Provider(),
        transcription_refiner=ArabicRefiner(),
    ).transcribe_chunks_async((AudioChunk(audio, duration_seconds=10.0),))

    assert result.raw_transcript == "قال"
    assert result.refined_transcript == "كال"
    assert result.final_transcript == "كال"
    assert result.subtitle_cues == (SubtitleCue(0.0, 1.0, "كال"),)
    assert result.translated_srt is None


@pytest.mark.asyncio
async def test_transcribe_chunks_falls_back_atomically_when_refinement_fails(tmp_path: Path) -> None:
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")

    class Provider:
        provider_name = "gemini"
        model = "gemini-model"

        def transcribe_file_result(self, audio_path: Path, *, previous_transcript: str = "") -> FileTranscriptionResult:
            return FileTranscriptionResult(
                transcript="قال",
                subtitle_cues=(SubtitleCue(0.0, 1.0, "قال"),),
            )

    class FailingArabicRefiner:
        model = "arabic-refinement-model"

        def refine_srt(self, srt: str) -> str:
            raise TranscriptionError("invalid refined SRT")

    class Translator:
        model = "translation-model"

        def translate_srt_blocks(
            self,
            blocks: object,
            *,
            previous_context: object = (),
            following_context: object = (),
        ) -> tuple[str, ...]:
            del previous_context, following_context
            assert tuple(" ".join(block.text_lines) for block in blocks) == ("قال",)
            return ("گفت",)

    result = await SpeechTranscriber(
        speech_to_text_provider=Provider(),
        transcription_refiner=FailingArabicRefiner(),
        refiner=Translator(),
    ).transcribe_chunks_async((AudioChunk(audio, duration_seconds=10.0),))

    assert result.refined_transcript is None
    assert result.final_transcript == "قال"
    assert result.subtitle_cues == (SubtitleCue(0.0, 1.0, "قال"),)
    assert result.translated_srt is not None and "قال" in result.translated_srt
    assert result.warnings == (TRANSCRIPTION_REFINEMENT_FAILURE_WARNING,)


@pytest.mark.asyncio
async def test_transcribe_chunks_keeps_raw_srt_when_translation_fails(tmp_path: Path) -> None:
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")

    class Provider:
        provider_name = "gemini"
        model = "gemini-model"

        def transcribe_file_result(self, audio_path: Path, *, previous_transcript: str = "") -> FileTranscriptionResult:
            return FileTranscriptionResult(
                transcript="raw",
                subtitle_cues=(SubtitleCue(0.0, 1.0, "raw"),),
            )

    class FailingRefiner:
        model = "translation-model"

        def translate_srt_blocks(self, *args: object, **kwargs: object) -> tuple[str, ...]:
            raise TranscriptionError("translation failed")

    events: list[str] = []

    async def record_progress(event: str, data: object) -> None:
        events.append(event)

    result = await SpeechTranscriber(
        speech_to_text_provider=Provider(),
        refiner=FailingRefiner(),
    ).transcribe_chunks_async(
        (AudioChunk(audio, duration_seconds=10.0),),
        progress_callback=record_progress,
    )

    assert result.raw_transcript == "raw"
    assert result.subtitle_cues == (SubtitleCue(0.0, 1.0, "raw"),)
    assert result.translated_srt is None
    assert result.warnings == (TRANSLATION_FAILURE_WARNING,)
    assert "translation_failed" in events


@pytest.mark.asyncio
async def test_transcribe_chunks_keeps_deepgram_result_when_confidence_correction_fails(tmp_path: Path) -> None:
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"audio")

    class Provider:
        provider_name = "deepgram"
        model = "nova-3"

        def transcribe_file_result(self, audio_path: Path, *, previous_transcript: str = "") -> FileTranscriptionResult:
            return FileTranscriptionResult(
                transcript="original",
                subtitle_cues=(SubtitleCue(0.0, 1.0, "original"),),
            )

    class FailingCorrector:
        def correct_file_result(self, audio_path: Path, result: FileTranscriptionResult) -> FileTranscriptionResult:
            raise TranscriptionError("correction failed")

    result = await SpeechTranscriber(
        speech_to_text_provider=Provider(),
        correctors_by_provider={"deepgram": FailingCorrector()},
    ).transcribe_chunks_async((AudioChunk(audio, duration_seconds=10.0),))

    assert result.raw_transcript == "original"
    assert result.warnings == (CORRECTION_FAILURE_WARNING,)


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
        if event == "translating_subtitles"
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


def test_iraqi_refiner_uses_exact_system_prompt_and_preserves_srt_identity() -> None:
    raw_srt = "1\n00:00:00,000 --> 00:00:01,000\nقال أقدر\n"
    refined_srt = "1\n00:00:00,000 --> 00:00:01,000\nكال أكدر\n"

    class FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=refined_srt))]
            )

    completions = FakeCompletions()
    refiner = IraqiArabicTranscriptRefiner(
        api_key="key",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    assert refiner.refine_srt(raw_srt) == refined_srt
    assert len(completions.calls) == 1
    call = completions.calls[0]
    assert call["model"] == DEFAULT_TRANSCRIPTION_REFINEMENT_MODEL
    assert call["messages"] == [{
        "role": "system",
        "content": IRAQI_ARABIC_TRANSCRIPTION_REFINEMENT_SYSTEM_PROMPT.replace(
            TRANSCRIPTION_REFINEMENT_PLACEHOLDER,
            raw_srt.strip(),
            1,
        ),
    }]
    assert call["extra_body"] == {"provider": {"require_parameters": True}}


@pytest.mark.parametrize(
    ("refined_srt", "message"),
    [
        ("2\n00:00:00,000 --> 00:00:01,000\nكال\n", "cue number"),
        ("1\n00:00:00,100 --> 00:00:01,000\nكال\n", "timestamp"),
        (
            "1\n00:00:00,000 --> 00:00:01,000\nكال\n\n"
            "2\n00:00:01,000 --> 00:00:02,000\nأكدر\n",
            "number of subtitle cues",
        ),
    ],
)
def test_validate_refined_srt_blocks_rejects_structural_changes(
    refined_srt: str,
    message: str,
) -> None:
    expected = parse_srt_blocks("1\n00:00:00,000 --> 00:00:01,000\nقال\n")

    with pytest.raises(TranscriptionError, match=message):
        validate_refined_srt_blocks(refined_srt, expected_blocks=expected)


def test_iraqi_refiner_splits_oversized_cue_without_exceeding_byte_limit() -> None:
    raw_srt = (
        "1\n"
        "00:00:00,000 --> 00:00:30,000\n"
        + ("كلمة " * 5000).strip()
        + "\n"
    )
    prefix = IRAQI_ARABIC_TRANSCRIPTION_REFINEMENT_SYSTEM_PROMPT.partition(
        TRANSCRIPTION_REFINEMENT_PLACEHOLDER
    )[0]

    class EchoCompletions:
        def __init__(self) -> None:
            self.srt_byte_sizes: list[int] = []

        def create(self, **kwargs: object) -> object:
            messages = kwargs["messages"]
            content = messages[0]["content"]
            source_srt = content.removeprefix(prefix)
            self.srt_byte_sizes.append(len(source_srt.encode("utf-8")))
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=source_srt))]
            )

    completions = EchoCompletions()
    refiner = IraqiArabicTranscriptRefiner(
        api_key="key",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    refined = refiner.refine_srt(raw_srt)

    assert len(completions.srt_byte_sizes) > 1
    assert all(size <= MAX_SRT_REFINEMENT_CHUNK_BYTES for size in completions.srt_byte_sizes)
    assert parse_srt_blocks(refined)[0].index == "1"
    assert parse_srt_blocks(refined)[0].timestamp == "00:00:00,000 --> 00:00:30,000"
    assert " ".join(parse_srt_blocks(refined)[0].text_lines).split() == (
        " ".join(parse_srt_blocks(raw_srt)[0].text_lines).split()
    )


def test_split_text_by_utf8_byte_limit_keeps_arabic_fragments_bounded() -> None:
    text = "هلا شلونك هاي تجربة عراقية"

    fragments = split_text_by_utf8_byte_limit(text, max_bytes=12)

    assert all(len(fragment.encode("utf-8")) <= 12 for fragment in fragments)
    assert " ".join(fragments).split() == text.split()


def test_split_text_by_utf8_byte_limit_rejects_limit_smaller_than_one_character() -> None:
    with pytest.raises(ValueError, match="one UTF-8 character"):
        split_text_by_utf8_byte_limit("ه", max_bytes=1)


def test_build_transcription_refinement_system_prompt_replaces_only_placeholder() -> None:
    raw_srt = "1\n00:00:00,000 --> 00:00:01,000\nهاي\n"

    prompt = build_transcription_refinement_system_prompt(raw_srt)

    assert prompt == IRAQI_ARABIC_TRANSCRIPTION_REFINEMENT_SYSTEM_PROMPT.replace(
        TRANSCRIPTION_REFINEMENT_PLACEHOLDER,
        raw_srt.strip(),
        1,
    )
    assert TRANSCRIPTION_REFINEMENT_PLACEHOLDER not in prompt


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
    assert "temperature" not in call
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["strict"] is True
    assert call["extra_body"] == {"provider": {"require_parameters": True}}
    assert "provider" not in call


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

        def transcribe_file_result(self, audio_path: Path, *, previous_transcript: str = "") -> FileTranscriptionResult:
            return FileTranscriptionResult(
                transcript="raw transcript",
                subtitle_cues=(SubtitleCue(0.0, 1.0, "raw transcript"),),
            )

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
