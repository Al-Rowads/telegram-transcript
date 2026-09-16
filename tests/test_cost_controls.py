from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, BadRequestError, OpenAI

from telegram_transcript import bot, transcriber as module
from telegram_transcript.config import ConfigError, Settings, load_settings
from telegram_transcript.model_catalog import GEMINI_FLASH_LITE_MODEL, RETIRED_TEXT_MODELS
from telegram_transcript.models import AudioChunk, FileTranscriptionResult, SubtitleCue, TranscriptWord
from telegram_transcript.transcriber import SrtBlock, TranscriptRefiner, TranscriptionError


class RecordingTranslations:
    """Transport double for checking request costs and cue delivery, not quality."""

    def __init__(self, responses: list[object] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.responses = iter(responses) if responses is not None else None

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.responses is not None:
            response = next(self.responses)
            if isinstance(response, Exception):
                raise response
            return response
        payload = json.loads(kwargs["messages"][1]["content"])
        content = json.dumps({"translations": [
            {"index": index, "translation": "سلام"} for index, _ in payload["cues"]
        ]})
        return completion(content)


def completion(content: str, *, finish_reason: str = "stop", usage: object = None) -> object:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)],
        usage=usage,
    )


def translator(completions: RecordingTranslations, **kwargs: object) -> TranscriptRefiner:
    return TranscriptRefiner(
        api_key="test-key", client=SimpleNamespace(chat=SimpleNamespace(completions=completions)), **kwargs,
    )


def cue(index: int, text: str = "هاي تجربة") -> SrtBlock:
    return SrtBlock(str(index), "00:00:00,000 --> 00:00:01,000", (text,))


def test_large_translation_has_bounded_requests_and_complete_cue_coverage() -> None:
    transport = RecordingTranslations()
    blocks = tuple(cue(index) for index in range(1, 97))
    translated = translator(transport).translate_srt_blocks(blocks)
    assert translated == ("سلام",) * 96
    assert len(transport.calls) == 2
    sent_ids = []
    for call in transport.calls:
        payload = json.loads(call["messages"][1]["content"])
        sent_ids.extend(index for index, _ in payload["cues"])
        assert "00:00" not in call["messages"][1]["content"]
        assert len(call["messages"][1]["content"].encode()) <= 12 * 1024
        assert 512 <= call["max_tokens"] <= 8192
    assert sent_ids == [block.index for block in blocks]


def test_oversized_cue_fragments_preserve_all_source_text_and_original_identity() -> None:
    transport = RecordingTranslations()
    text = "هاي تجربة " * 3000
    block = cue(7, text.strip())
    result = translator(transport).translate_srt_blocks((block,))
    assert len(result) == 1
    fragments = []
    for call in transport.calls:
        payload = json.loads(call["messages"][1]["content"])
        assert len(payload["cues"]) == 1
        index, fragment = payload["cues"][0]
        assert index == "7"
        fragments.append(fragment)
        assert 512 <= call["max_tokens"] <= 8192
        assert len(call["messages"][1]["content"].encode()) <= 12 * 1024
    assert len(fragments) > 1
    assert " ".join(fragments) == text.strip()
    assert module.render_translated_srt_block(block, result[0]).startswith("7\n" + block.timestamp)


def test_context_is_bounded_and_literal_translation_omits_it() -> None:
    previous = tuple((cue(i, "قبل " * 60), "ترجمه " * 60) for i in range(1, 7))
    following = tuple(cue(i, "بعد " * 60) for i in range(8, 13))
    raw = module.render_srt_blocks((cue(7),))
    payload = json.loads(module.build_refinement_input(raw, previous_context=previous, following_context=following))
    context = {key: payload.get(key, []) for key in ("previous", "following")}
    assert len(json.dumps(context, ensure_ascii=False, separators=(",", ":")).encode()) <= 1024
    assert all(len(values) <= 2 for values in context.values())
    assert payload["cues"] == [["7", "هاي تجربة"]]
    transport = RecordingTranslations()
    translator(transport, prompt_key="literal").translate_srt_blocks(
        (cue(7),), previous_context=previous, following_context=following,
    )
    assert json.loads(transport.calls[0]["messages"][1]["content"]) == {"cues": [["7", "هاي تجربة"]]}


@pytest.mark.parametrize("count, expected_calls", [(1, 3), (48, 2)])
def test_invalid_responses_never_create_a_retry_tree(count: int, expected_calls: int) -> None:
    transport = RecordingTranslations([completion("not json")] * 3)
    with pytest.raises(TranscriptionError, match="valid JSON"):
        translator(transport).translate_srt_blocks(tuple(cue(i) for i in range(count)))
    assert len(transport.calls) == expected_calls


def test_transport_retries_share_the_three_call_limit() -> None:
    failure = APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
    transport = RecordingTranslations([failure, completion("bad"), completion("bad")])
    with pytest.raises(TranscriptionError):
        translator(transport).translate_srt_blocks((cue(1), cue(2)))
    assert len(transport.calls) == 3


def test_permanent_api_error_is_not_retried() -> None:
    failure = BadRequestError(
        "unsupported model", response=httpx.Response(400, request=httpx.Request("POST", "https://example.invalid")),
        body=None,
    )
    transport = RecordingTranslations([failure])
    with pytest.raises(BadRequestError):
        translator(transport).translate_srt_blocks((cue(1),))
    assert len(transport.calls) == 1


def test_truncated_response_is_rejected_even_when_json_is_valid() -> None:
    response = completion('{"translations":[{"index":"1","translation":"سلام"}]}', finish_reason="length")
    transport = RecordingTranslations([response] * 3)
    with pytest.raises(TranscriptionError, match="truncated"):
        translator(transport).translate_srt_blocks((cue(1),))
    assert len(transport.calls) == 3


def test_usage_is_logged_before_validation_without_content_or_invented_counts(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="telegram_transcript.transcriber")
    response = completion("private-response-content", usage={
        "prompt_tokens": 120, "completion_tokens": 30,
        "completion_tokens_details": {"reasoning_tokens": 0}, "cost": 0.000024,
    })
    transport = RecordingTranslations([response] * 3)
    with pytest.raises(TranscriptionError):
        translator(transport).translate_srt_blocks((cue(1, "private-source-content"),))
    assert caplog.text.count("input_tokens=120 output_tokens=30 reasoning_tokens=0 cost_usd=2.4e-05") == 3
    assert "private-source-content" not in caplog.text
    assert "private-response-content" not in caplog.text
    module.log_completion_usage(completion("private"), model="model")
    assert "input_tokens=None output_tokens=None reasoning_tokens=None cost_usd=None" in caplog.text


def test_text_sdk_retries_are_disabled() -> None:
    client = module.create_openrouter_client("test-key")
    try:
        assert client.max_retries == 0
    finally:
        client.close()


def test_openrouter_sdk_serializes_cost_controls_and_exposes_reported_usage(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="telegram_transcript.transcriber")
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["reasoning"] == {"enabled": False}
        assert payload["provider"] == {"require_parameters": True}
        assert payload["max_tokens"] == 512
        assert payload["response_format"]["json_schema"]["strict"] is True
        return httpx.Response(200, json={
            "id": "test-completion", "object": "chat.completion", "created": 0,
            "model": GEMINI_FLASH_LITE_MODEL,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": '{"translations":[{"index":"1","translation":"سلام"}]}',
            }}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110, "cost": 0.000014},
        })

    with OpenAI(api_key="test", base_url=module.OPENROUTER_BASE_URL, max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        result = TranscriptRefiner(api_key="test", client=client).translate_srt_blocks((cue(1),))
    assert result == ("سلام",)
    assert len(requests) == 1
    assert "input_tokens=100 output_tokens=10 reasoning_tokens=None cost_usd=1.4e-05" in caplog.text


def test_gemini_truncation_is_not_misreported_as_silence(tmp_path: Path) -> None:
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"unused by injected transport")
    provider = module.GeminiSpeechToTextProvider(api_key="test", client=SimpleNamespace(
        chat=SimpleNamespace(completions=RecordingTranslations([completion("", finish_reason="length")])),
    ))
    with pytest.raises(TranscriptionError, match="truncated"):
        provider.transcribe_file_result(audio)


@pytest.mark.parametrize("model", sorted(RETIRED_TEXT_MODELS))
def test_known_expensive_environment_overrides_are_migrated(model: str) -> None:
    settings = load_settings({
        "TELEGRAM_BOT_TOKEN": "test", "TELEGRAM_API_ID": "123", "TELEGRAM_API_HASH": "test",
        "OPENROUTER_API_KEY": "test", "OPENROUTER_REFINE_MODEL": model,
        "OPENROUTER_TRANSCRIPTION_REFINEMENT_MODEL": model,
    }, load_dotenv_file=False)
    assert settings.openrouter_refine_model == GEMINI_FLASH_LITE_MODEL
    assert settings.openrouter_transcription_refinement_model == GEMINI_FLASH_LITE_MODEL


def test_default_provider_never_silently_switches_when_deepgram_is_unconfigured() -> None:
    with pytest.raises(ConfigError, match="DEEPGRAM_API_KEY"):
        bot.create_transcriber(Settings(telegram_bot_token="test", openrouter_api_key="test"))
    assert bot.parse_translation_model_command_arg("gpt") is None
    assert bot.parse_translation_model_command_arg("claude") is None


@pytest.mark.asyncio
async def test_default_low_confidence_transcription_only_calls_translation_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = RecordingTranslations()
    monkeypatch.setattr(module, "create_openrouter_client", lambda _: SimpleNamespace(
        chat=SimpleNamespace(completions=transport),
    ))
    pipeline = bot.create_transcriber(Settings(
        telegram_bot_token="test", deepgram_api_key="test", openrouter_api_key="test", refine=True,
    ))
    monkeypatch.setattr(pipeline.speech_to_text_provider, "transcribe_file_result", lambda *args, **kwargs:
        FileTranscriptionResult(
            transcript="هاي", subtitle_cues=(SubtitleCue(0, 1, "هاي"),),
            words=(TranscriptWord(0, 1, "هاي", 0.01),),
        ))
    audio = tmp_path / "audio.flac"
    audio.write_bytes(b"unused by injected provider")
    result = await pipeline.transcribe_chunks_async((AudioChunk(audio, duration_seconds=1),))
    assert result.final_transcript == "هاي"
    assert result.persian_transcript == "سلام"
    assert result.refined_transcript is None
    assert len(transport.calls) == 1
    assert transport.calls[0]["model"] == GEMINI_FLASH_LITE_MODEL


@pytest.mark.parametrize("retired", sorted(RETIRED_TEXT_MODELS))
def test_json_migration_runs_once_and_preserves_later_explicit_choices(tmp_path: Path, retired: str) -> None:
    store = bot.create_runtime_preferences_store(Settings(telegram_bot_token="test", runtime_state_path=tmp_path / "state.json"))
    original = replace(store.defaults, transcription_model="gemini", translation_enabled=True,
                       translation_model=retired, transcription_refinement_model="google/gemini-3.5-flash")
    payload = {"version": 3, **asdict(original)}
    payload.pop("audio_correction_enabled")
    payload.pop("transcription_refinement_enabled")
    store.path.write_text(json.dumps(payload))
    migrated = store.load()
    assert migrated.transcription_model == "deepgram"
    assert migrated.translation_enabled is True
    assert migrated.translation_model == GEMINI_FLASH_LITE_MODEL
    assert migrated.transcription_refinement_model == "google/gemini-3.5-flash"
    assert not migrated.transcription_refinement_enabled and not migrated.audio_correction_enabled
    selected = replace(migrated, transcription_model="gemini", transcription_refinement_enabled=True,
                       audio_correction_enabled=True)
    store.save(selected)
    assert store.load() == selected


@pytest.mark.parametrize("field", ["audio_correction_enabled", "transcription_refinement_enabled"])
@pytest.mark.parametrize("value", [None, 1, "true"])
def test_current_json_preferences_require_boolean_pass_flags(tmp_path: Path, field: str, value: object) -> None:
    store = bot.create_runtime_preferences_store(Settings(telegram_bot_token="test", runtime_state_path=tmp_path / "state.json"))
    payload = {"version": 4, **asdict(store.defaults), field: value}
    store.path.write_text(json.dumps(payload))
    with pytest.raises(ConfigError, match="enabled"):
        store.load()
