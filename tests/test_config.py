from __future__ import annotations

import pytest

from telegram_transcript.config import (
    ConfigError,
    load_settings,
    mb_to_bytes,
    parse_bool,
    parse_user_ids,
)


BASE_ENV = {
    "TELEGRAM_BOT_TOKEN": "telegram-token",
    "TELEGRAM_API_ID": "12345",
    "TELEGRAM_API_HASH": "telegram-api-hash",
    "DEEPGRAM_API_KEY": "deepgram-key",
    "OPENAI_API_KEY": "openai-key",
    "OPENROUTER_API_KEY": "openrouter-key",
}


def test_load_settings_uses_defaults() -> None:
    settings = load_settings(BASE_ENV, load_dotenv_file=False)

    assert settings.telegram_bot_token == "telegram-token"
    assert settings.telegram_api_id == 12345
    assert settings.telegram_api_hash == "telegram-api-hash"
    assert settings.deepgram_api_key == "deepgram-key"
    assert settings.deepgram_transcribe_model == "nova-3"
    assert settings.deepgram_language == "ar-IQ"
    assert settings.deepgram_keyterms == ()
    assert settings.openai_api_key == "openai-key"
    assert settings.openrouter_api_key == "openrouter-key"
    assert settings.openrouter_refine_model == "openai/gpt-5.4-mini"
    assert settings.openrouter_transcription_refinement_model == "openai/gpt-5.4-mini"
    assert settings.refine is True
    assert settings.max_video_bytes == mb_to_bytes(2048)
    assert settings.audio_tempo == 1.0
    assert settings.max_concurrent_jobs == 1
    assert settings.telegram_request_timeout_seconds == 30.0
    assert settings.telegram_media_write_timeout_seconds == 120.0
    assert settings.runtime_state_path.as_posix() == "data/runtime-settings.json"
    assert settings.video_registry_path.as_posix() == "data/videos.sqlite3"
    assert settings.scoped_state_path.as_posix() == "data/bot-state.sqlite3"


def test_load_settings_parses_optional_values() -> None:
    settings = load_settings(
        {
            **BASE_ENV,
            "OPENROUTER_API_KEY": "custom-openrouter-key",
            "OPENROUTER_REFINE_MODEL": "custom-refine-model",
            "OPENROUTER_TRANSCRIPTION_REFINEMENT_MODEL": "custom-transcription-refinement-model",
            "DEEPGRAM_KEYTERMS": "اسم, مصطلح, اسم",
            "REFINE": "true",
            "ALLOWED_TELEGRAM_USER_IDS": "123, 456",
            "MAX_VIDEO_MB": "25.5",
            "AUDIO_TEMPO": "1",
            "MAX_CONCURRENT_JOBS": "3",
            "TELEGRAM_REQUEST_TIMEOUT_SECONDS": "45",
            "TELEGRAM_MEDIA_WRITE_TIMEOUT_SECONDS": "180",
            "RUNTIME_STATE_PATH": "~/telegram-state.json",
            "VIDEO_REGISTRY_PATH": "~/telegram-videos.sqlite3",
        },
        load_dotenv_file=False,
    )

    assert settings.deepgram_api_key == "deepgram-key"
    assert settings.openrouter_api_key == "custom-openrouter-key"
    assert settings.openrouter_refine_model == "custom-refine-model"
    assert settings.openrouter_transcription_refinement_model == "custom-transcription-refinement-model"
    assert settings.deepgram_keyterms == ("اسم", "مصطلح")
    assert settings.refine is True
    assert settings.allowed_telegram_user_ids == frozenset({123, 456})
    assert settings.max_video_mb == 25.5
    assert settings.audio_tempo == 1
    assert settings.max_concurrent_jobs == 3
    assert settings.telegram_request_timeout_seconds == 45.0
    assert settings.telegram_media_write_timeout_seconds == 180.0
    assert settings.runtime_state_path.name == "telegram-state.json"
    assert settings.video_registry_path.name == "telegram-videos.sqlite3"


def test_load_settings_rejects_more_than_one_hundred_deepgram_keyterms() -> None:
    with pytest.raises(ConfigError, match="at most 100"):
        load_settings(
            {**BASE_ENV, "DEEPGRAM_KEYTERMS": ",".join(f"term-{index}" for index in range(101))},
            load_dotenv_file=False,
        )


def test_load_settings_requires_credentials() -> None:
    with pytest.raises(ConfigError, match="TELEGRAM_BOT_TOKEN"):
        load_settings(
            {
                "TELEGRAM_API_ID": "12345",
                "TELEGRAM_API_HASH": "telegram-api-hash",
                "DEEPGRAM_API_KEY": "deepgram-key",
                "OPENROUTER_API_KEY": "openrouter-key",
            },
            load_dotenv_file=False,
        )

    with pytest.raises(ConfigError, match="TELEGRAM_API_ID"):
        load_settings(
            {
                "TELEGRAM_BOT_TOKEN": "telegram-token",
                "TELEGRAM_API_HASH": "telegram-api-hash",
                "DEEPGRAM_API_KEY": "deepgram-key",
                "OPENROUTER_API_KEY": "openrouter-key",
            },
            load_dotenv_file=False,
        )

    with pytest.raises(ConfigError, match="TELEGRAM_API_HASH"):
        load_settings(
            {
                "TELEGRAM_BOT_TOKEN": "telegram-token",
                "TELEGRAM_API_ID": "12345",
                "DEEPGRAM_API_KEY": "deepgram-key",
                "OPENROUTER_API_KEY": "openrouter-key",
            },
            load_dotenv_file=False,
        )


@pytest.mark.parametrize("api_id", ["nope", "0"])
def test_load_settings_rejects_invalid_telegram_api_id(api_id: str) -> None:
    with pytest.raises(ConfigError, match="TELEGRAM_API_ID"):
        load_settings({**BASE_ENV, "TELEGRAM_API_ID": api_id}, load_dotenv_file=False)


def test_deepgram_provider_is_optional() -> None:
    settings = load_settings(
        {
            "TELEGRAM_BOT_TOKEN": "telegram-token",
            "TELEGRAM_API_ID": "12345",
            "TELEGRAM_API_HASH": "telegram-api-hash",
            "OPENROUTER_API_KEY": "openrouter-key",
        },
        load_dotenv_file=False,
    )

    assert settings.deepgram_api_key == ""


def test_legacy_provider_setting_does_not_replace_deepgram() -> None:
    settings = load_settings(
        {
            "TELEGRAM_BOT_TOKEN": "telegram-token",
            "TELEGRAM_API_ID": "12345",
            "TELEGRAM_API_HASH": "telegram-api-hash",
            "SPEECH_TO_TEXT_PROVIDER": "openai",
            "OPENAI_API_KEY": "openai-key",
            "OPENROUTER_API_KEY": "openrouter-key",
        },
        load_dotenv_file=False,
    )

    assert settings.deepgram_api_key == ""


def test_openrouter_api_key_is_required() -> None:
    env = {key: value for key, value in BASE_ENV.items() if key != "OPENROUTER_API_KEY"}
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        load_settings(env, load_dotenv_file=False)


def test_openai_api_key_is_optional() -> None:
    env = {key: value for key, value in BASE_ENV.items() if key != "OPENAI_API_KEY"}
    assert load_settings(env, load_dotenv_file=False).openai_api_key == ""


def test_audio_tempo_must_stay_in_ffmpeg_range() -> None:
    with pytest.raises(ConfigError, match="AUDIO_TEMPO"):
        load_settings({**BASE_ENV, "AUDIO_TEMPO": "0.25"}, load_dotenv_file=False)


def test_max_video_limit_cannot_exceed_telegram_two_gib_limit() -> None:
    with pytest.raises(ConfigError, match="2048 MB"):
        load_settings({**BASE_ENV, "MAX_VIDEO_MB": "2049"}, load_dotenv_file=False)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TELEGRAM_REQUEST_TIMEOUT_SECONDS", "0"),
        ("TELEGRAM_MEDIA_WRITE_TIMEOUT_SECONDS", "not-a-number"),
    ],
)
def test_telegram_timeouts_must_be_positive_numbers(name: str, value: str) -> None:
    with pytest.raises(ConfigError, match=name):
        load_settings({**BASE_ENV, name: value}, load_dotenv_file=False)


def test_parse_user_ids_rejects_invalid_values() -> None:
    with pytest.raises(ConfigError, match="Invalid Telegram user ID"):
        parse_user_ids("123, nope")


def test_parse_bool_accepts_common_values() -> None:
    assert parse_bool("true", "REFINE", False) is True
    assert parse_bool("1", "REFINE", False) is True
    assert parse_bool("off", "REFINE", True) is False
    assert parse_bool(None, "REFINE", True) is True


def test_parse_bool_rejects_invalid_values() -> None:
    with pytest.raises(ConfigError, match="REFINE"):
        parse_bool("sometimes", "REFINE", True)
