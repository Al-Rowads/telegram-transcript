from __future__ import annotations

import json

import pytest

from telegram_transcript.config import ConfigError
from telegram_transcript.runtime_state import RuntimePreferences, RuntimePreferencesStore


def make_store(tmp_path: object) -> RuntimePreferencesStore:
    path = tmp_path / "runtime.json"
    defaults = RuntimePreferences(1.0, "gemini", "openai/gpt-5.5", "normal")
    return RuntimePreferencesStore(
        path,
        defaults=defaults,
        transcription_models=frozenset({"gemini", "deepgram"}),
        translation_models=frozenset({"openai/gpt-5.5", "anthropic/claude-sonnet-4.6"}),
        translation_prompts=frozenset({"normal", "v2"}),
    )


def test_runtime_preferences_round_trip_and_use_defaults_when_missing(tmp_path: object) -> None:
    store = make_store(tmp_path)
    assert store.load() == store.defaults

    preferences = RuntimePreferences(1.2, "deepgram", "anthropic/claude-sonnet-4.6", "v2")
    store.save(preferences)

    assert store.load() == preferences
    assert json.loads(store.path.read_text(encoding="utf-8"))["version"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        '{"version": 2}',
        '{"version": 1, "audio_tempo": 1}',
        (
            '{"version": 1, "audio_tempo": 1, "transcription_model": "invalid", '
            '"translation_model": "openai/gpt-5.5", "translation_prompt": "normal"}'
        ),
    ],
)
def test_runtime_preferences_reject_invalid_state(tmp_path: object, payload: str) -> None:
    store = make_store(tmp_path)
    store.path.write_text(payload, encoding="utf-8")

    with pytest.raises(ConfigError):
        store.load()
