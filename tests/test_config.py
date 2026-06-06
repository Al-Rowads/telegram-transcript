from __future__ import annotations

import pytest

from telegram_transcript.config import ConfigError, load_settings, mb_to_bytes, parse_user_ids


BASE_ENV = {
    "TELEGRAM_BOT_TOKEN": "telegram-token",
    "OPENAI_API_KEY": "openai-key",
}


def test_load_settings_uses_defaults() -> None:
    settings = load_settings(BASE_ENV, load_dotenv_file=False)

    assert settings.telegram_bot_token == "telegram-token"
    assert settings.openai_api_key == "openai-key"
    assert settings.openai_transcribe_model == "whisper-1"
    assert settings.openai_refine_model == "GPT5.4-mini"
    assert settings.max_video_bytes == mb_to_bytes(100)
    assert settings.max_openai_audio_bytes == mb_to_bytes(24)
    assert settings.max_concurrent_jobs == 1


def test_load_settings_parses_optional_values() -> None:
    settings = load_settings(
        {
            **BASE_ENV,
            "OPENAI_TRANSCRIBE_MODEL": "gpt-4o-transcribe",
            "OPENAI_REFINE_MODEL": "custom-refine-model",
            "ALLOWED_TELEGRAM_USER_IDS": "123, 456",
            "MAX_VIDEO_MB": "25.5",
            "MAX_OPENAI_AUDIO_MB": "12",
            "MAX_CONCURRENT_JOBS": "3",
        },
        load_dotenv_file=False,
    )

    assert settings.openai_transcribe_model == "gpt-4o-transcribe"
    assert settings.openai_refine_model == "custom-refine-model"
    assert settings.allowed_telegram_user_ids == frozenset({123, 456})
    assert settings.max_video_mb == 25.5
    assert settings.max_openai_audio_mb == 12
    assert settings.max_concurrent_jobs == 3


def test_load_settings_requires_credentials() -> None:
    with pytest.raises(ConfigError, match="TELEGRAM_BOT_TOKEN"):
        load_settings({"OPENAI_API_KEY": "openai-key"}, load_dotenv_file=False)


def test_openai_audio_limit_must_stay_below_25_mb() -> None:
    with pytest.raises(ConfigError, match="below OpenAI"):
        load_settings({**BASE_ENV, "MAX_OPENAI_AUDIO_MB": "25"}, load_dotenv_file=False)


def test_parse_user_ids_rejects_invalid_values() -> None:
    with pytest.raises(ConfigError, match="Invalid Telegram user ID"):
        parse_user_ids("123, nope")
