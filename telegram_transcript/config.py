from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from dotenv import load_dotenv

from telegram_transcript.ffmpeg import DEFAULT_AUDIO_TEMPO
from telegram_transcript.transcriber import (
    DEFAULT_DEEPGRAM_LANGUAGE,
    DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL,
    DEFAULT_REFINEMENT_MODEL,
)


class ConfigError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    deepgram_api_key: str = ""
    deepgram_transcribe_model: str = DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL
    deepgram_language: str = DEFAULT_DEEPGRAM_LANGUAGE
    openai_api_key: str = ""
    openai_refine_model: str = DEFAULT_REFINEMENT_MODEL
    refine: bool = False
    allowed_telegram_user_ids: frozenset[int] = frozenset()
    max_video_mb: float = 100.0
    audio_tempo: float = DEFAULT_AUDIO_TEMPO
    max_concurrent_jobs: int = 1

    @property
    def max_video_bytes(self) -> int:
        return mb_to_bytes(self.max_video_mb)


def load_settings(
    env: Mapping[str, str] | None = None,
    *,
    load_dotenv_file: bool = True,
) -> Settings:
    if load_dotenv_file:
        load_dotenv()

    source = env if env is not None else os.environ
    telegram_bot_token = require_value(source, "TELEGRAM_BOT_TOKEN")
    refine = parse_bool(source.get("REFINE"), "REFINE", False)

    deepgram_api_key = source.get("DEEPGRAM_API_KEY", "").strip()
    if not deepgram_api_key:
        raise ConfigError("DEEPGRAM_API_KEY is required.")
    deepgram_model = (
        source.get("DEEPGRAM_TRANSCRIBE_MODEL", DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL).strip()
        or DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL
    )
    deepgram_language = source.get("DEEPGRAM_LANGUAGE", DEFAULT_DEEPGRAM_LANGUAGE).strip() or DEFAULT_DEEPGRAM_LANGUAGE

    openai_api_key = source.get("OPENAI_API_KEY", "").strip()
    if refine and not openai_api_key:
        raise ConfigError("OPENAI_API_KEY is required when REFINE=true.")
    refine_model = source.get("OPENAI_REFINE_MODEL", DEFAULT_REFINEMENT_MODEL).strip() or DEFAULT_REFINEMENT_MODEL
    max_video_mb = parse_positive_float(source.get("MAX_VIDEO_MB"), "MAX_VIDEO_MB", 100.0)
    audio_tempo = parse_audio_tempo(source.get("AUDIO_TEMPO"))

    return Settings(
        telegram_bot_token=telegram_bot_token,
        deepgram_api_key=deepgram_api_key,
        deepgram_transcribe_model=deepgram_model,
        deepgram_language=deepgram_language,
        openai_api_key=openai_api_key,
        openai_refine_model=refine_model,
        refine=refine,
        allowed_telegram_user_ids=parse_user_ids(source.get("ALLOWED_TELEGRAM_USER_IDS")),
        max_video_mb=max_video_mb,
        audio_tempo=audio_tempo,
        max_concurrent_jobs=parse_positive_int(
            source.get("MAX_CONCURRENT_JOBS"),
            "MAX_CONCURRENT_JOBS",
            1,
        ),
    )


def require_value(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ConfigError(f"{name} is required.")
    return value


def parse_user_ids(raw: str | None) -> frozenset[int]:
    if not raw or not raw.strip():
        return frozenset()

    user_ids: set[int] = set()
    for part in raw.split(","):
        candidate = part.strip()
        if not candidate:
            continue
        try:
            user_ids.add(int(candidate))
        except ValueError as exc:
            raise ConfigError(f"Invalid Telegram user ID: {candidate}") from exc
    return frozenset(user_ids)


def parse_positive_float(raw: str | None, name: str, default: float) -> float:
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number.") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be greater than zero.")
    return value


def parse_positive_int(raw: str | None, name: str, default: int) -> int:
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer.") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be greater than zero.")
    return value


def parse_bool(raw: str | None, name: str, default: bool) -> bool:
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigError(f"{name} must be true or false.")


def parse_audio_tempo(raw: str | None) -> float:
    value = parse_positive_float(raw, "AUDIO_TEMPO", DEFAULT_AUDIO_TEMPO)
    if not 0.5 <= value <= 2.0:
        raise ConfigError("AUDIO_TEMPO must be between 0.5 and 2.0.")
    return value


def mb_to_bytes(value: float) -> int:
    return int(value * 1024 * 1024)
