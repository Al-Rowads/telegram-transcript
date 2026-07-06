from __future__ import annotations

import pytest

from telegram_transcript.config import (
    ConfigError,
    load_settings,
    mb_to_bytes,
    parse_bool,
    parse_speech_to_text_provider,
    parse_user_ids,
)


BASE_ENV = {
    "TELEGRAM_BOT_TOKEN": "telegram-token",
    "DEEPGRAM_API_KEY": "deepgram-key",
}


def test_load_settings_uses_defaults() -> None:
    settings = load_settings(BASE_ENV, load_dotenv_file=False)

    assert settings.telegram_bot_token == "telegram-token"
    assert settings.speech_to_text_provider == "deepgram"
    assert settings.deepgram_api_key == "deepgram-key"
    assert settings.deepgram_transcribe_model == "nova-3"
    assert settings.deepgram_language == "ar"
    assert settings.openai_api_key == ""
    assert settings.openai_refine_model == "gpt-5.4-mini"
    assert settings.refine is False
    assert settings.max_video_bytes == mb_to_bytes(100)
    assert settings.max_audio_bytes == mb_to_bytes(24)
    assert settings.audio_tempo == 1.0
    assert settings.max_concurrent_jobs == 1


def test_load_settings_parses_optional_values() -> None:
    settings = load_settings(
        {
            "TELEGRAM_BOT_TOKEN": "telegram-token",
            "SPEECH_TO_TEXT_PROVIDER": "deepgram",
            "DEEPGRAM_API_KEY": "deepgram-key",
            "OPENAI_API_KEY": "openai-key",
            "OPENAI_REFINE_MODEL": "custom-refine-model",
            "REFINE": "true",
            "ALLOWED_TELEGRAM_USER_IDS": "123, 456",
            "MAX_VIDEO_MB": "25.5",
            "MAX_AUDIO_MB": "12",
            "AUDIO_TEMPO": "1",
            "MAX_CONCURRENT_JOBS": "3",
        },
        load_dotenv_file=False,
    )

    assert settings.speech_to_text_provider == "deepgram"
    assert settings.deepgram_api_key == "deepgram-key"
    assert settings.openai_api_key == "openai-key"
    assert settings.openai_refine_model == "custom-refine-model"
    assert settings.refine is True
    assert settings.allowed_telegram_user_ids == frozenset({123, 456})
    assert settings.max_video_mb == 25.5
    assert settings.max_audio_mb == 12
    assert settings.audio_tempo == 1
    assert settings.max_concurrent_jobs == 3


def test_load_settings_requires_credentials() -> None:
    with pytest.raises(ConfigError, match="TELEGRAM_BOT_TOKEN"):
        load_settings({"OPENAI_API_KEY": "openai-key"}, load_dotenv_file=False)


def test_deepgram_provider_requires_deepgram_api_key() -> None:
    with pytest.raises(ConfigError, match="DEEPGRAM_API_KEY"):
        load_settings({"TELEGRAM_BOT_TOKEN": "telegram-token"}, load_dotenv_file=False)


def test_openai_transcription_provider_is_not_supported() -> None:
    with pytest.raises(ConfigError, match="SPEECH_TO_TEXT_PROVIDER"):
        load_settings(
            {
                "TELEGRAM_BOT_TOKEN": "telegram-token",
                "SPEECH_TO_TEXT_PROVIDER": "openai",
                "OPENAI_API_KEY": "openai-key",
            },
            load_dotenv_file=False,
        )


def test_refinement_requires_openai_api_key() -> None:
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        load_settings({**BASE_ENV, "REFINE": "true"}, load_dotenv_file=False)


def test_audio_limit_must_stay_below_25_mb() -> None:
    with pytest.raises(ConfigError, match="below 25 MB"):
        load_settings({**BASE_ENV, "MAX_AUDIO_MB": "25"}, load_dotenv_file=False)


def test_audio_tempo_must_stay_in_ffmpeg_range() -> None:
    with pytest.raises(ConfigError, match="AUDIO_TEMPO"):
        load_settings({**BASE_ENV, "AUDIO_TEMPO": "0.25"}, load_dotenv_file=False)


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


def test_parse_speech_to_text_provider_rejects_invalid_values() -> None:
    with pytest.raises(ConfigError, match="SPEECH_TO_TEXT_PROVIDER"):
        parse_speech_to_text_provider("local")
