from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from telegram_transcript.ffmpeg import DEFAULT_AUDIO_TEMPO
from telegram_transcript.transcriber import (
    DEFAULT_DEEPGRAM_LANGUAGE,
    DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL,
    DEFAULT_REFINEMENT_MODEL,
    DEFAULT_TRANSCRIPTION_REFINEMENT_MODEL,
)

MAX_TELEGRAM_MEDIA_MB = 2048.0


class ConfigError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    telegram_api_id: int = 0
    telegram_api_hash: str = ""
    deepgram_api_key: str = ""
    deepgram_transcribe_model: str = DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL
    deepgram_language: str = DEFAULT_DEEPGRAM_LANGUAGE
    deepgram_keyterms: tuple[str, ...] = ()
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    openrouter_refine_model: str = DEFAULT_REFINEMENT_MODEL
    openrouter_transcription_refinement_model: str = DEFAULT_TRANSCRIPTION_REFINEMENT_MODEL
    refine: bool = False
    allowed_telegram_user_ids: frozenset[int] = frozenset()
    max_video_mb: float = MAX_TELEGRAM_MEDIA_MB
    audio_tempo: float = DEFAULT_AUDIO_TEMPO
    max_concurrent_jobs: int = 1
    runtime_state_path: Path = Path("data/runtime-settings.json")
    video_registry_path: Path = Path("data/videos.sqlite3")
    scoped_state_path: Path = Path("data/bot-state.sqlite3")

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
    telegram_api_id = parse_positive_int(require_value(source, "TELEGRAM_API_ID"), "TELEGRAM_API_ID", 0)
    telegram_api_hash = require_value(source, "TELEGRAM_API_HASH")
    refine = parse_bool(source.get("REFINE"), "REFINE", True)

    deepgram_api_key = source.get("DEEPGRAM_API_KEY", "").strip()
    deepgram_model = (
        source.get("DEEPGRAM_TRANSCRIBE_MODEL", DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL).strip()
        or DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL
    )
    deepgram_language = source.get("DEEPGRAM_LANGUAGE", DEFAULT_DEEPGRAM_LANGUAGE).strip() or DEFAULT_DEEPGRAM_LANGUAGE
    deepgram_keyterms = parse_comma_separated_values(source.get("DEEPGRAM_KEYTERMS"))
    if len(deepgram_keyterms) > 100:
        raise ConfigError("DEEPGRAM_KEYTERMS supports at most 100 unique terms.")

    openrouter_api_key = require_value(source, "OPENROUTER_API_KEY")
    openai_api_key = source.get("OPENAI_API_KEY", "").strip()
    refine_model = (
        source.get("OPENROUTER_REFINE_MODEL", DEFAULT_REFINEMENT_MODEL).strip()
        or DEFAULT_REFINEMENT_MODEL
    )
    transcription_refinement_model = (
        source.get(
            "OPENROUTER_TRANSCRIPTION_REFINEMENT_MODEL",
            DEFAULT_TRANSCRIPTION_REFINEMENT_MODEL,
        ).strip()
        or DEFAULT_TRANSCRIPTION_REFINEMENT_MODEL
    )
    max_video_mb = parse_positive_float(source.get("MAX_VIDEO_MB"), "MAX_VIDEO_MB", MAX_TELEGRAM_MEDIA_MB)
    if max_video_mb > MAX_TELEGRAM_MEDIA_MB:
        raise ConfigError("MAX_VIDEO_MB must be less than or equal to Telegram's 2048 MB file limit.")
    audio_tempo = parse_audio_tempo(source.get("AUDIO_TEMPO"))

    return Settings(
        telegram_bot_token=telegram_bot_token,
        telegram_api_id=telegram_api_id,
        telegram_api_hash=telegram_api_hash,
        deepgram_api_key=deepgram_api_key,
        deepgram_transcribe_model=deepgram_model,
        deepgram_language=deepgram_language,
        deepgram_keyterms=deepgram_keyterms,
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key,
        openrouter_refine_model=refine_model,
        openrouter_transcription_refinement_model=transcription_refinement_model,
        refine=refine,
        allowed_telegram_user_ids=parse_user_ids(source.get("ALLOWED_TELEGRAM_USER_IDS")),
        max_video_mb=max_video_mb,
        audio_tempo=audio_tempo,
        max_concurrent_jobs=parse_positive_int(
            source.get("MAX_CONCURRENT_JOBS"),
            "MAX_CONCURRENT_JOBS",
            1,
        ),
        runtime_state_path=Path(
            source.get("RUNTIME_STATE_PATH", "data/runtime-settings.json").strip()
            or "data/runtime-settings.json"
        ).expanduser(),
        video_registry_path=Path(
            source.get("VIDEO_REGISTRY_PATH", "data/videos.sqlite3").strip()
            or "data/videos.sqlite3"
        ).expanduser(),
        scoped_state_path=Path(
            source.get("SCOPED_STATE_PATH", "data/bot-state.sqlite3").strip()
            or "data/bot-state.sqlite3"
        ).expanduser(),
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


def parse_comma_separated_values(raw: str | None) -> tuple[str, ...]:
    if not raw or not raw.strip():
        return ()
    return tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))


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
