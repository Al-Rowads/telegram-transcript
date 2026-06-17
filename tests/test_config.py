from __future__ import annotations

import pytest

from telegram_transcript.config import ConfigError, load_settings, mb_to_bytes, parse_bool, parse_user_ids


BASE_ENV = {
    "TELEGRAM_BOT_TOKEN": "telegram-token",
    "TELEGRAM_API_ID": "12345",
    "TELEGRAM_API_HASH": "telegram-api-hash",
    "OPENAI_API_KEY": "openai-key",
}


def test_load_settings_uses_defaults() -> None:
    settings = load_settings(BASE_ENV, load_dotenv_file=False)

    assert settings.telegram_bot_token == "telegram-token"
    assert settings.telegram_api_id == 12345
    assert settings.telegram_api_hash == "telegram-api-hash"
    assert settings.openai_api_key == "openai-key"
    assert not hasattr(settings, "telegram_api_base_url")
    assert not hasattr(settings, "telegram_api_base_file_url")
    assert not hasattr(settings, "telegram_local_mode")
    assert settings.openai_transcribe_model == "gpt-4o-transcribe"
    assert settings.openai_refine_model == "gpt-5.4-mini"
    assert settings.refine is True
    assert settings.allowed_telegram_topic_id is None
    assert settings.max_video_bytes == mb_to_bytes(2048)
    assert settings.max_openai_audio_bytes == mb_to_bytes(24)
    assert settings.audio_tempo == 1.0
    assert settings.max_concurrent_jobs == 1


def test_load_settings_parses_optional_values() -> None:
    settings = load_settings(
        {
            **BASE_ENV,
            "OPENAI_TRANSCRIBE_MODEL": "gpt-4o-transcribe",
            "OPENAI_REFINE_MODEL": "custom-refine-model",
            "REFINE": "false",
            "ALLOWED_TELEGRAM_USER_IDS": "123, 456",
            "ALLOWED_TELEGRAM_TOPIC_ID": "789",
            "MAX_VIDEO_MB": "25.5",
            "MAX_OPENAI_AUDIO_MB": "12",
            "AUDIO_TEMPO": "1",
            "MAX_CONCURRENT_JOBS": "3",
        },
        load_dotenv_file=False,
    )

    assert settings.openai_transcribe_model == "gpt-4o-transcribe"
    assert settings.openai_refine_model == "custom-refine-model"
    assert settings.refine is False
    assert settings.allowed_telegram_user_ids == frozenset({123, 456})
    assert settings.allowed_telegram_topic_id == 789
    assert settings.max_video_mb == 25.5
    assert settings.max_openai_audio_mb == 12
    assert settings.audio_tempo == 1
    assert settings.max_concurrent_jobs == 3


def test_load_settings_requires_credentials() -> None:
    with pytest.raises(ConfigError, match="TELEGRAM_BOT_TOKEN"):
        load_settings(
            {
                "TELEGRAM_API_ID": "12345",
                "TELEGRAM_API_HASH": "telegram-api-hash",
                "OPENAI_API_KEY": "openai-key",
            },
            load_dotenv_file=False,
        )

    with pytest.raises(ConfigError, match="TELEGRAM_API_ID"):
        load_settings(
            {
                "TELEGRAM_BOT_TOKEN": "telegram-token",
                "TELEGRAM_API_HASH": "telegram-api-hash",
                "OPENAI_API_KEY": "openai-key",
            },
            load_dotenv_file=False,
        )

    with pytest.raises(ConfigError, match="TELEGRAM_API_HASH"):
        load_settings(
            {
                "TELEGRAM_BOT_TOKEN": "telegram-token",
                "TELEGRAM_API_ID": "12345",
                "OPENAI_API_KEY": "openai-key",
            },
            load_dotenv_file=False,
        )


@pytest.mark.parametrize("api_id", ["nope", "0"])
def test_load_settings_rejects_invalid_telegram_api_id(api_id: str) -> None:
    with pytest.raises(ConfigError, match="TELEGRAM_API_ID"):
        load_settings({**BASE_ENV, "TELEGRAM_API_ID": api_id}, load_dotenv_file=False)


def test_openai_audio_limit_must_stay_below_25_mb() -> None:
    with pytest.raises(ConfigError, match="below OpenAI"):
        load_settings({**BASE_ENV, "MAX_OPENAI_AUDIO_MB": "25"}, load_dotenv_file=False)


def test_max_video_limit_cannot_exceed_telegram_two_gib_limit() -> None:
    with pytest.raises(ConfigError, match="2048 MB"):
        load_settings({**BASE_ENV, "MAX_VIDEO_MB": "2049"}, load_dotenv_file=False)


def test_audio_tempo_must_stay_in_ffmpeg_range() -> None:
    with pytest.raises(ConfigError, match="AUDIO_TEMPO"):
        load_settings({**BASE_ENV, "AUDIO_TEMPO": "0.25"}, load_dotenv_file=False)


def test_parse_user_ids_rejects_invalid_values() -> None:
    with pytest.raises(ConfigError, match="Invalid Telegram user ID"):
        parse_user_ids("123, nope")


@pytest.mark.parametrize("topic_id", ["nope", "0"])
def test_load_settings_rejects_invalid_allowed_telegram_topic_id(topic_id: str) -> None:
    with pytest.raises(ConfigError, match="ALLOWED_TELEGRAM_TOPIC_ID"):
        load_settings({**BASE_ENV, "ALLOWED_TELEGRAM_TOPIC_ID": topic_id}, load_dotenv_file=False)


def test_parse_bool_accepts_common_values() -> None:
    assert parse_bool("true", "REFINE", False) is True
    assert parse_bool("1", "REFINE", False) is True
    assert parse_bool("off", "REFINE", True) is False
    assert parse_bool(None, "REFINE", True) is True


def test_parse_bool_rejects_invalid_values() -> None:
    with pytest.raises(ConfigError, match="REFINE"):
        parse_bool("sometimes", "REFINE", True)
