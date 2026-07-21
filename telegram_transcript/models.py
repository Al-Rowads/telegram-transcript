from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioChunk:
    path: Path
    start_seconds: float = 0.0
    duration_seconds: float | None = None
    time_scale: float = 1.0
    owned_start_seconds: float | None = None
    owned_end_seconds: float | None = None


@dataclass(frozen=True)
class TranscriptWord:
    start_seconds: float
    end_seconds: float
    text: str
    confidence: float | None = None

    def shifted(self, offset_seconds: float) -> TranscriptWord:
        return TranscriptWord(
            start_seconds=self.start_seconds + offset_seconds,
            end_seconds=self.end_seconds + offset_seconds,
            text=self.text,
            confidence=self.confidence,
        )

    def scaled(self, factor: float) -> TranscriptWord:
        return TranscriptWord(
            start_seconds=self.start_seconds * factor,
            end_seconds=self.end_seconds * factor,
            text=self.text,
            confidence=self.confidence,
        )


@dataclass(frozen=True)
class SubtitleCue:
    start_seconds: float
    end_seconds: float
    text: str

    def shifted(self, offset_seconds: float) -> SubtitleCue:
        return SubtitleCue(
            start_seconds=self.start_seconds + offset_seconds,
            end_seconds=self.end_seconds + offset_seconds,
            text=self.text,
        )

    def scaled(self, factor: float) -> SubtitleCue:
        return SubtitleCue(
            start_seconds=self.start_seconds * factor,
            end_seconds=self.end_seconds * factor,
            text=self.text,
        )


@dataclass(frozen=True)
class FileTranscriptionResult:
    transcript: str
    subtitle_cues: tuple[SubtitleCue, ...] = ()
    words: tuple[TranscriptWord, ...] = ()


@dataclass(frozen=True)
class TranscriptionResult:
    raw_transcript: str
    refined_transcript: str | None = None
    cleaned_transcript: str | None = None
    persian_transcript: str | None = None
    subtitle_cues: tuple[SubtitleCue, ...] = ()
    translated_srt: str | None = None
    line_translated_transcript: str | None = None
    words: tuple[TranscriptWord, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def final_transcript(self) -> str:
        """Return the authoritative, audio-grounded transcript.

        ``refined_transcript`` remains available as a compatibility alias for
        callers that need the derived reading version, but it is never the
        source of truth.
        """
        return self.raw_transcript

    @property
    def reading_transcript(self) -> str:
        return self.cleaned_transcript or self.refined_transcript or self.raw_transcript
