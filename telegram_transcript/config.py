from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    openai_api_key: str
    openai_transcribe_model: str = "whisper-1"
    openai_refine_model: str = "GPT5.4-mini"
    allowed_telegram_user_ids: frozenset[int] = frozenset()
    max_video_mb: float = 100.0
    max_openai_audio_mb: float = 24.0
    max_concurrent_jobs: int = 1

    @property
    def max_video_bytes(self) -> int:
        return mb_to_bytes(self.max_video_mb)

    @property
    def max_openai_audio_bytes(self) -> int:
        return mb_to_bytes(self.max_openai_audio_mb)


def load_settings(
    env: Mapping[str, str] | None = None,
    *,
    load_dotenv_file: bool = True,
) -> Settings:
    if load_dotenv_file:
        load_dotenv()

    source = env if env is not None else os.environ
    telegram_bot_token = require_value(source, "TELEGRAM_BOT_TOKEN")
    openai_api_key = require_value(source, "OPENAI_API_KEY")
    model = source.get("OPENAI_TRANSCRIBE_MODEL", "whisper-1").strip() or "whisper-1"
    refine_model = source.get("OPENAI_REFINE_MODEL", "GPT5.4-mini").strip() or "GPT5.4-mini"
    max_video_mb = parse_positive_float(source.get("MAX_VIDEO_MB"), "MAX_VIDEO_MB", 100.0)
    max_openai_audio_mb = parse_positive_float(
        source.get("MAX_OPENAI_AUDIO_MB"),
        "MAX_OPENAI_AUDIO_MB",
        24.0,
    )
    if max_openai_audio_mb >= 25:
        raise ConfigError("MAX_OPENAI_AUDIO_MB must be below OpenAI's 25 MB upload limit.")

    return Settings(
        telegram_bot_token=telegram_bot_token,
        openai_api_key=openai_api_key,
        openai_transcribe_model=model,
        openai_refine_model=refine_model,
        allowed_telegram_user_ids=parse_user_ids(source.get("ALLOWED_TELEGRAM_USER_IDS")),
        max_video_mb=max_video_mb,
        max_openai_audio_mb=max_openai_audio_mb,
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


def mb_to_bytes(value: float) -> int:
    return int(value * 1024 * 1024)
