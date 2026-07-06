from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from openai import OpenAI

from telegram_transcript.models import AudioChunk, FileTranscriptionResult, SubtitleCue, TranscriptionResult

DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL = "nova-3"
DEFAULT_DEEPGRAM_LANGUAGE = "ar"
DEFAULT_REFINEMENT_MODEL = "gpt-5.4"
RAW_TRANSCRIPT_START = "<srt_file>"
RAW_TRANSCRIPT_END = "</srt_file>"
ProgressCallback = Callable[[str, Mapping[str, object]], Awaitable[None]]

SRT_TRANSLATION_SYSTEM_PROMPT = """You are an expert Arabic-to-Persian subtitle translator.

You will receive a complete SRT subtitle file. Your job is to add one Persian translation line to each subtitle cue while preserving the SRT file structure exactly.

Rules:
- Do not change cue numbers.
- Do not change timestamps.
- Do not change blank lines between cues.
- Do not change, correct, normalize, or translate the original Arabic subtitle text.
- Add exactly one Persian translation line after the original Arabic text in every cue.
- Wrap only the Persian translation line in this exact tag format: <font color="green">Persian translation</font>.
- Keep names, numbers, brands, and technical terms accurate.
- Output only the final SRT text.

Example output cue:
1103
01:07:29,610 --> 01:07:30,810
{{Arabic text}}
<font color="green">{{Persian translation for that line}}</font>"""

SRT_TRANSLATION_REQUEST = (
    "Add Persian translations to this SRT file. Preserve every cue number, timestamp, "
    "blank line, and original Arabic subtitle line exactly. Output only valid SRT."
)
BAGHDADI_ARABIC_REFINEMENT_SYSTEM_PROMPT = SRT_TRANSLATION_SYSTEM_PROMPT
BAGHDADI_ARABIC_REFINEMENT_REQUEST = SRT_TRANSLATION_REQUEST
GREEN_FONT_RE = re.compile(r'^<font\s+color=["\']?green["\']?>\s*(.*?)\s*</font>$', re.IGNORECASE)


class TranscriptionError(RuntimeError):
    """Raised when a speech-to-text or refinement response is unusable."""


@dataclass(frozen=True)
class SrtBlock:
    index: str
    timestamp: str
    text_lines: tuple[str, ...]


class SpeechToTextProvider(Protocol):
    provider_name: str
    model: str

    def transcribe_file(self, audio_path: Path, *, previous_transcript: str = "") -> str:
        ...


class DeepgramSpeechToTextProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL,
        language: str = DEFAULT_DEEPGRAM_LANGUAGE,
        client: Any | None = None,
    ) -> None:
        self.provider_name = "deepgram"
        self.model = model
        self.language = language
        self.client = client if client is not None else create_deepgram_client(api_key)

    def transcribe_file(self, audio_path: Path, *, previous_transcript: str = "") -> str:
        return self.transcribe_file_result(audio_path, previous_transcript=previous_transcript).transcript

    def transcribe_file_result(self, audio_path: Path, *, previous_transcript: str = "") -> FileTranscriptionResult:
        del previous_transcript
        with audio_path.open("rb") as audio_file:
            response = self.client.listen.v1.media.transcribe_file(
                request=audio_file.read(),
                model=self.model,
                language=self.language,
                smart_format=True,
                punctuate=True,
                paragraphs=True,
                utterances=True,
            )
        return extract_deepgram_file_transcription_result(response)


class TranscriptRefiner:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_REFINEMENT_MODEL,
        system_prompt: str = SRT_TRANSLATION_SYSTEM_PROMPT,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.client = client if client is not None else OpenAI(api_key=api_key)

    def refine_transcript(self, transcript: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=self.system_prompt,
            input=build_refinement_input(transcript),
            temperature=0,
        )
        refined = extract_response_text(response).strip()
        if not refined:
            raise TranscriptionError("OpenAI refinement returned empty text.")
        return refined


class SpeechTranscriber:
    def __init__(
        self,
        *,
        speech_to_text_provider: SpeechToTextProvider,
        refiner: TranscriptRefiner | None = None,
    ) -> None:
        self.speech_to_text_provider = speech_to_text_provider
        self.refiner = refiner

    @property
    def provider_name(self) -> str:
        return self.speech_to_text_provider.provider_name

    @property
    def model(self) -> str:
        return self.speech_to_text_provider.model

    @property
    def refine(self) -> bool:
        return self.refiner is not None

    @property
    def refinement_model(self) -> str | None:
        return self.refiner.model if self.refiner is not None else None

    def transcribe_file(self, audio_path: Path, *, previous_transcript: str = "") -> str:
        return self.speech_to_text_provider.transcribe_file(
            audio_path,
            previous_transcript=previous_transcript,
        )

    def transcribe_file_result(self, audio_path: Path, *, previous_transcript: str = "") -> FileTranscriptionResult:
        transcribe_file_result = getattr(self.speech_to_text_provider, "transcribe_file_result", None)
        if callable(transcribe_file_result):
            return transcribe_file_result(audio_path, previous_transcript=previous_transcript)
        return FileTranscriptionResult(
            transcript=self.transcribe_file(audio_path, previous_transcript=previous_transcript)
        )

    def transcribe_chunks(self, chunks: Sequence[Path]) -> str:
        return self.transcribe_chunks_raw(chunks)

    def transcribe_chunks_raw(self, chunks: Sequence[Path]) -> str:
        transcripts = []
        previous_transcript = ""
        for chunk in chunks:
            text = self.transcribe_file(chunk, previous_transcript=previous_transcript).strip()
            if text:
                transcripts.append(text)
                previous_transcript = text
        return "\n\n".join(transcripts)

    def refine_transcript(self, transcript: str) -> str:
        if self.refiner is None:
            return transcript
        return self.refiner.refine_transcript(transcript)

    async def transcribe_chunks_async(
        self,
        chunks: Sequence[Path | AudioChunk],
        progress_callback: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        total = len(chunks)
        transcripts = []
        subtitle_cues: list[SubtitleCue] = []
        previous_transcript = ""
        for index, chunk in enumerate(chunks, start=1):
            audio_chunk = normalize_audio_chunk(chunk)
            if progress_callback is not None:
                await progress_callback(
                    "transcribing_chunk",
                    {
                        "index": index,
                        "total": total,
                        "chunk_bytes": audio_chunk.path.stat().st_size,
                        "provider": self.provider_name,
                        "model": self.model,
                    },
                )
            file_result = (
                await asyncio.to_thread(
                    self.transcribe_file_result,
                    audio_chunk.path,
                    previous_transcript=previous_transcript,
                )
            )
            text = file_result.transcript.strip()
            if text:
                transcripts.append(text)
                previous_transcript = text
            subtitle_cues.extend(cue.shifted(audio_chunk.start_seconds) for cue in file_result.subtitle_cues)
            if progress_callback is not None:
                await progress_callback(
                    "chunk_transcribed",
                    {
                        "index": index,
                        "total": total,
                        "raw_chars": len(text),
                    },
                )

        transcript = "\n\n".join(transcripts)
        if not transcript.strip() or self.refiner is None or not subtitle_cues:
            return TranscriptionResult(raw_transcript=transcript, subtitle_cues=tuple(subtitle_cues))

        raw_srt = render_srt(subtitle_cues)
        if progress_callback is not None:
            await progress_callback(
                "refining_transcript",
                {
                    "raw_chars": len(raw_srt),
                    "model": self.refiner.model,
                },
            )
        translated_srt = await asyncio.to_thread(self.refine_transcript, raw_srt)
        translated_srt = validate_translated_srt(raw_srt=raw_srt, translated_srt=translated_srt)
        if progress_callback is not None:
            await progress_callback(
                "refinement_complete",
                {
                    "translated_srt_chars": len(translated_srt),
                },
            )
        return TranscriptionResult(
            raw_transcript=transcript,
            subtitle_cues=tuple(subtitle_cues),
            translated_srt=translated_srt,
        )


def create_deepgram_client(api_key: str) -> Any:
    try:
        from deepgram import DeepgramClient
    except ImportError as exc:
        raise TranscriptionError("deepgram-sdk is required for transcription.") from exc
    return DeepgramClient(api_key=api_key)


def extract_deepgram_transcript_text(response: Any) -> str:
    paragraphs = get_nested_response_value(
        response,
        ("results", "channels", 0, "alternatives", 0, "paragraphs", "paragraphs"),
    )
    if isinstance(paragraphs, Sequence) and not isinstance(paragraphs, (str, bytes)):
        paragraph_texts = [
            text.strip()
            for paragraph in paragraphs
            if isinstance((text := get_response_field(paragraph, "text")), str) and text.strip()
        ]
        if paragraph_texts:
            return "\n\n".join(paragraph_texts)

    transcript = get_nested_response_value(
        response,
        ("results", "channels", 0, "alternatives", 0, "transcript"),
    )
    if isinstance(transcript, str):
        return transcript
    raise TranscriptionError("Deepgram transcription response did not include transcript.")


def extract_deepgram_file_transcription_result(response: Any) -> FileTranscriptionResult:
    return FileTranscriptionResult(
        transcript=extract_deepgram_transcript_text(response),
        subtitle_cues=extract_deepgram_subtitle_cues(response),
    )


def extract_deepgram_subtitle_cues(response: Any) -> tuple[SubtitleCue, ...]:
    utterance_cues = extract_deepgram_utterance_cues(response)
    if utterance_cues:
        return utterance_cues
    return extract_deepgram_word_cues(response)


def extract_deepgram_utterance_cues(response: Any) -> tuple[SubtitleCue, ...]:
    utterances = get_nested_response_value(response, ("results", "utterances"))
    if not isinstance(utterances, Sequence) or isinstance(utterances, (str, bytes)):
        return ()

    cues = []
    for utterance in utterances:
        cue = build_subtitle_cue(
            start=get_response_field(utterance, "start"),
            end=get_response_field(utterance, "end"),
            text=get_response_field(utterance, "transcript"),
        )
        if cue is not None:
            cues.append(cue)
    return tuple(cues)


def extract_deepgram_word_cues(response: Any) -> tuple[SubtitleCue, ...]:
    words = get_nested_response_value(response, ("results", "channels", 0, "alternatives", 0, "words"))
    if not isinstance(words, Sequence) or isinstance(words, (str, bytes)):
        return ()

    cues = []
    for word in words:
        text = get_response_field(word, "punctuated_word") or get_response_field(word, "word")
        cue = build_subtitle_cue(
            start=get_response_field(word, "start"),
            end=get_response_field(word, "end"),
            text=text,
        )
        if cue is not None:
            cues.append(cue)
    return tuple(cues)


def get_nested_response_value(response: Any, path: Sequence[str | int]) -> Any:
    current = response
    for part in path:
        if isinstance(part, int):
            if not isinstance(current, Sequence) or isinstance(current, (str, bytes)):
                return None
            try:
                current = current[part]
            except IndexError:
                return None
            continue

        if isinstance(current, Mapping):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            return None
    return current


def get_response_field(response: Any, name: str) -> Any:
    if isinstance(response, Mapping):
        return response.get(name)
    return getattr(response, name, None)


def build_subtitle_cue(*, start: Any, end: Any, text: Any) -> SubtitleCue | None:
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return None
    if end <= start:
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    return SubtitleCue(start_seconds=float(start), end_seconds=float(end), text=text.strip())


def normalize_audio_chunk(chunk: Path | AudioChunk) -> AudioChunk:
    if isinstance(chunk, AudioChunk):
        return chunk
    return AudioChunk(path=chunk)


def format_srt_timestamp(seconds: float) -> str:
    total_milliseconds = max(0, round(seconds * 1000))
    milliseconds = total_milliseconds % 1000
    total_seconds = total_milliseconds // 1000
    seconds_part = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d},{milliseconds:03d}"


def render_srt(cues: Sequence[SubtitleCue]) -> str:
    blocks = []
    for index, cue in enumerate(cues, start=1):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_srt_timestamp(cue.start_seconds)} --> {format_srt_timestamp(cue.end_seconds)}",
                    cue.text,
                ]
            )
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def parse_srt_blocks(srt: str) -> tuple[SrtBlock, ...]:
    normalized = srt.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return ()

    blocks = []
    for raw_block in re.split(r"\n{2,}", normalized):
        lines = raw_block.split("\n")
        if len(lines) < 3:
            raise TranscriptionError("SRT block is missing text.")
        index = lines[0].strip()
        timestamp = lines[1].strip()
        text_lines = tuple(lines[2:])
        if not index:
            raise TranscriptionError("SRT block is missing cue number.")
        if "-->" not in timestamp:
            raise TranscriptionError("SRT block is missing timestamp.")
        if not all(line.strip() for line in text_lines):
            raise TranscriptionError("SRT block contains an empty subtitle line.")
        blocks.append(SrtBlock(index=index, timestamp=timestamp, text_lines=text_lines))
    return tuple(blocks)


def validate_translated_srt(*, raw_srt: str, translated_srt: str) -> str:
    raw_blocks = parse_srt_blocks(raw_srt)
    translated_blocks = parse_srt_blocks(translated_srt)
    if len(raw_blocks) != len(translated_blocks):
        raise TranscriptionError("OpenAI translated SRT changed the cue count.")

    normalized_blocks = []
    for raw_block, translated_block in zip(raw_blocks, translated_blocks, strict=True):
        if raw_block.index != translated_block.index:
            raise TranscriptionError("OpenAI translated SRT changed a cue number.")
        if raw_block.timestamp != translated_block.timestamp:
            raise TranscriptionError("OpenAI translated SRT changed a timestamp.")

        expected_line_count = len(raw_block.text_lines) + 1
        if len(translated_block.text_lines) != expected_line_count:
            raise TranscriptionError("OpenAI translated SRT must add exactly one Persian line per cue.")

        original_lines = translated_block.text_lines[: len(raw_block.text_lines)]
        if original_lines != raw_block.text_lines:
            raise TranscriptionError("OpenAI translated SRT changed the original Arabic text.")

        persian_line = translated_block.text_lines[-1].strip()
        match = GREEN_FONT_RE.fullmatch(persian_line)
        if match is not None:
            persian_text = match.group(1).strip()
        elif persian_line.lower().startswith("<font"):
            raise TranscriptionError("OpenAI translated SRT used unsupported color markup.")
        else:
            persian_text = persian_line
        if not persian_text:
            raise TranscriptionError("OpenAI translated SRT included an empty Persian translation.")

        normalized_blocks.append(
            "\n".join(
                [
                    raw_block.index,
                    raw_block.timestamp,
                    *raw_block.text_lines,
                    f'<font color="green">{persian_text}</font>',
                ]
            )
        )
    return "\n\n".join(normalized_blocks) + ("\n" if normalized_blocks else "")


def extract_response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        output_text = response.get("output_text")
        if isinstance(output_text, str):
            return output_text

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text

    raise TranscriptionError("OpenAI refinement response did not include text.")


def build_refinement_input(transcript: str) -> str:
    return f"{SRT_TRANSLATION_REQUEST}\n\n{RAW_TRANSCRIPT_START}\n{transcript}\n{RAW_TRANSCRIPT_END}"
