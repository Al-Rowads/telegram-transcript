from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from telegram_transcript.config import ConfigError
from telegram_transcript.runtime_state import RuntimePreferences, RuntimePreferencesStore


def make_store(tmp_path: object) -> RuntimePreferencesStore:
    path = tmp_path / "runtime.json"
    defaults = RuntimePreferences(
        audio_tempo=1.0,
        transcription_model="gemini",
        transcription_refinement_model="google/gemini-2.5-flash-lite",
        translation_enabled=False,
        translation_model="google/gemini-2.5-flash-lite",
        translation_prompt="natural",
    )
    return RuntimePreferencesStore(
        path,
        defaults=defaults,
        transcription_models=frozenset({"gemini", "deepgram", "whisper"}),
        transcription_refinement_models=frozenset(
            {"google/gemini-2.5-flash-lite", "google/gemini-3.5-flash"}
        ),
        translation_models=frozenset({"google/gemini-2.5-flash-lite", "qwen/qwen3-30b-a3b-instruct-2507"}),
        translation_prompts=frozenset({"natural", "literal", "normal", "v2"}),
    )


def test_runtime_preferences_round_trip_and_use_defaults_when_missing(tmp_path: object) -> None:
    store = make_store(tmp_path)
    assert store.load() == store.defaults

    preferences = RuntimePreferences(
        audio_tempo=1.2,
        transcription_model="whisper",
        transcription_refinement_model="google/gemini-3.5-flash",
        translation_enabled=True,
        translation_model="qwen/qwen3-30b-a3b-instruct-2507",
        translation_prompt="v2",
    )
    store.save(preferences)

    assert store.load() == preferences
    assert json.loads(store.path.read_text(encoding="utf-8"))["version"] == 4


@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize("legacy_field", ["translation_model", "transcription_refinement_model", "both"])
def test_runtime_preferences_migrate_saved_gpt_models(tmp_path: object, version: int, legacy_field: str) -> None:
    store = make_store(tmp_path)
    expected = replace(
        store.defaults,
        audio_tempo=0.8,
        transcription_model="deepgram",
        translation_prompt="literal",
        translation_model=(
            "qwen/qwen3-30b-a3b-instruct-2507"
            if legacy_field == "transcription_refinement_model"
            else "google/gemini-2.5-flash-lite"
        ),
        transcription_refinement_model=(
            "google/gemini-3.5-flash"
            if legacy_field == "translation_model" and version >= 2
            else "google/gemini-2.5-flash-lite"
        ),
    )
    payload = {"version": version, **asdict(expected), "transcription_model": "whisper"}
    for field in ("translation_model", "transcription_refinement_model"):
        if legacy_field in {field, "both"}:
            payload[field] = "openai/gpt-5.5"
    if version == 1:
        del payload["transcription_refinement_model"]
    if version < 3:
        del payload["translation_enabled"]
    store.path.write_text(json.dumps(payload), encoding="utf-8")

    assert store.load() == expected
    assert store.load() == expected
    assert json.loads(store.path.read_text(encoding="utf-8")) == {"version": 4, **asdict(expected)}
    store.save(store.load())
    assert json.loads(store.path.read_text(encoding="utf-8")) == {"version": 4, **asdict(expected)}
    assert store.load() == expected


def test_runtime_preferences_load_version_one_with_default_refinement_model(tmp_path: object) -> None:
    store = make_store(tmp_path)
    store.path.write_text(
        json.dumps(
            {
                "version": 1,
                "audio_tempo": 1.2,
                "transcription_model": "whisper",
                "translation_model": "qwen/qwen3-30b-a3b-instruct-2507",
                "translation_prompt": "v2",
            }
        ),
        encoding="utf-8",
    )

    preferences = store.load()

    assert preferences.transcription_refinement_model == "google/gemini-2.5-flash-lite"
    assert preferences.transcription_model == "deepgram"
    assert preferences.translation_enabled is False
    assert preferences.translation_model == "qwen/qwen3-30b-a3b-instruct-2507"


def test_runtime_preferences_load_version_two_with_default_translation_enabled(tmp_path: object) -> None:
    store = make_store(tmp_path)
    store.path.write_text(
        json.dumps(
            {
                "version": 2,
                "audio_tempo": 1.2,
                "transcription_model": "whisper",
                "transcription_refinement_model": "google/gemini-3.5-flash",
                "translation_model": "qwen/qwen3-30b-a3b-instruct-2507",
                "translation_prompt": "v2",
            }
        ),
        encoding="utf-8",
    )

    preferences = store.load()

    assert preferences.transcription_refinement_model == "google/gemini-3.5-flash"
    assert preferences.translation_enabled is False


def test_runtime_preferences_reject_invalid_refinement_model(tmp_path: object) -> None:
    store = make_store(tmp_path)
    store.path.write_text(
        json.dumps(
            {
                "version": 2,
                "audio_tempo": 1,
                "transcription_model": "gemini",
                "transcription_refinement_model": "invalid",
                "translation_model": "google/gemini-2.5-flash-lite",
                "translation_prompt": "natural",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="transcription refinement model"):
        store.load()


@pytest.mark.parametrize("translation_enabled", [None, 1, "true"])
def test_runtime_preferences_reject_invalid_translation_enabled(
    tmp_path: object,
    translation_enabled: object,
) -> None:
    store = make_store(tmp_path)
    store.path.write_text(
        json.dumps(
            {
                "version": 3,
                "audio_tempo": 1,
                "transcription_model": "gemini",
                "transcription_refinement_model": "google/gemini-2.5-flash-lite",
                "translation_enabled": translation_enabled,
                "translation_model": "google/gemini-2.5-flash-lite",
                "translation_prompt": "natural",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="translation enabled"):
        store.load()


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        '{"version": 2}',
        '{"version": 1, "audio_tempo": 1}',
        (
            '{"version": 1, "audio_tempo": 1, "transcription_model": "invalid", '
            '"translation_model": "google/gemini-2.5-flash-lite", "translation_prompt": "normal"}'
        ),
    ],
)
def test_runtime_preferences_reject_invalid_state(tmp_path: object, payload: str) -> None:
    store = make_store(tmp_path)
    store.path.write_text(payload, encoding="utf-8")

    with pytest.raises(ConfigError):
        store.load()
