from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from telegram_transcript.config import ConfigError, parse_audio_tempo

RUNTIME_PREFERENCES_VERSION = 2
LEGACY_RUNTIME_PREFERENCES_VERSION = 1


class RuntimeStateError(RuntimeError):
    """Raised when runtime preferences cannot be saved."""


@dataclass(frozen=True)
class RuntimePreferences:
    audio_tempo: float
    transcription_model: str
    transcription_refinement_model: str
    translation_model: str
    translation_prompt: str


class RuntimePreferencesStore:
    def __init__(
        self,
        path: Path,
        *,
        defaults: RuntimePreferences,
        transcription_models: frozenset[str],
        transcription_refinement_models: frozenset[str],
        translation_models: frozenset[str],
        translation_prompts: frozenset[str],
    ) -> None:
        self.path = path
        self.defaults = defaults
        self.transcription_models = transcription_models
        self.transcription_refinement_models = transcription_refinement_models
        self.translation_models = translation_models
        self.translation_prompts = translation_prompts

    def load(self) -> RuntimePreferences:
        if not self.path.exists():
            return self.defaults
        try:
            payload: Any = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Unable to read runtime state from {self.path}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("version") not in {
            LEGACY_RUNTIME_PREFERENCES_VERSION,
            RUNTIME_PREFERENCES_VERSION,
        }:
            raise ConfigError(f"Unsupported runtime state format in {self.path}.")

        required_fields = {"audio_tempo", "transcription_model", "translation_model", "translation_prompt"}
        if not required_fields.issubset(payload):
            raise ConfigError(f"Runtime state in {self.path} is missing required settings.")
        refinement_model = (
            payload.get("transcription_refinement_model")
            if payload["version"] == RUNTIME_PREFERENCES_VERSION
            else self.defaults.transcription_refinement_model
        )
        if payload["version"] == RUNTIME_PREFERENCES_VERSION and refinement_model is None:
            raise ConfigError(f"Runtime state in {self.path} is missing required settings.")
        preferences = RuntimePreferences(
            audio_tempo=parse_audio_tempo(str(payload["audio_tempo"])),
            transcription_model=self._validate_choice(
                payload["transcription_model"], self.transcription_models, "transcription model"
            ),
            transcription_refinement_model=self._validate_choice(
                refinement_model,
                self.transcription_refinement_models,
                "transcription refinement model",
            ),
            translation_model=self._validate_choice(
                payload["translation_model"], self.translation_models, "translation model"
            ),
            translation_prompt=self._validate_choice(
                payload["translation_prompt"], self.translation_prompts, "translation prompt"
            ),
        )
        return preferences

    def save(self, preferences: RuntimePreferences) -> None:
        payload = {"version": RUNTIME_PREFERENCES_VERSION, **asdict(preferences)}
        temporary_path: Path | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.path.parent, prefix=f".{self.path.name}.", delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self.path)
        except OSError as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise RuntimeStateError(f"Unable to save runtime settings: {exc}") from exc

    @staticmethod
    def _validate_choice(value: object, choices: frozenset[str], label: str) -> str:
        if not isinstance(value, str) or value not in choices:
            raise ConfigError(f"Invalid persisted {label}: {value!r}.")
        return value
