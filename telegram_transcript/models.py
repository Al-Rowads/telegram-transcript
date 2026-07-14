from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioChunk:
    path: Path
    start_seconds: float = 0.0


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


@dataclass(frozen=True)
class FileTranscriptionResult:
    transcript: str
    subtitle_cues: tuple[SubtitleCue, ...] = ()
    words: tuple[TranscriptWord, ...] = ()


@dataclass(frozen=True)
class TranscriptionResult:
    raw_transcript: str
    refined_transcript: str | None = None
    subtitle_cues: tuple[SubtitleCue, ...] = ()
    translated_srt: str | None = None
    line_translated_transcript: str | None = None
    words: tuple[TranscriptWord, ...] = ()

    @property
    def final_transcript(self) -> str:
        return self.refined_transcript if self.refined_transcript is not None else self.raw_transcript
