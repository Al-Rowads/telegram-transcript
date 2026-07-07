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
DEFAULT_OPENAI_TRANSCRIPTION_MODEL = "gpt-4o-transcribe-diarize"
DEFAULT_GEMINI_TRANSCRIPTION_MODEL = "gemini-3.5-flash"
DEFAULT_REFINEMENT_MODEL = "gpt-5.4"
MAX_SRT_TRANSLATION_CHUNK_BYTES = 8 * 1024
RAW_TRANSCRIPT_START = "<srt_file>"
RAW_TRANSCRIPT_END = "</srt_file>"
ProgressCallback = Callable[[str, Mapping[str, object]], Awaitable[None]]

SRT_TRANSLATION_SYSTEM_PROMPT = """You are an expert Arabic-to-Persian subtitle translator.

You will receive a complete SRT subtitle file. Your job is to add Persian translations to the subtitle cues while keeping the output useful as subtitles.

Rules:
- Keep cue numbers, timestamps, blank lines, and original Arabic subtitle text when practical.
- Add Persian translation text after the related Arabic subtitle text.
- Use as many Persian lines as needed for a natural translation.
- Wrap Persian translation lines in this tag format when practical: <font color="green">Persian translation</font>.
- Keep names, numbers, brands, and technical terms accurate.
- Output only the translated subtitle text.

Example output cue:
1103
01:07:29,610 --> 01:07:30,810
{{Arabic text}}
<font color="green">{{Persian translation for that cue}}</font>"""

SRT_TRANSLATION_REQUEST = (
    "Add Persian translations to this SRT file. Keep cue numbers, timestamps, "
    "blank lines, and original Arabic subtitle lines when practical. Output only the translated subtitle text."
)

BAGHDADI_ARABIC_REFINEMENT_SYSTEM_PROMPT = SRT_TRANSLATION_SYSTEM_PROMPT
BAGHDADI_ARABIC_REFINEMENT_REQUEST = SRT_TRANSLATION_REQUEST
GREEN_FONT_RE = re.compile(r'^<font\s+color=["\']?green["\']?>\s*(.*?)\s*</font>$', re.IGNORECASE)
SRT_CUE_NUMBER_RE = re.compile(r"^\d+$")
SRT_TIMESTAMP_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}(?:\s+.*)?$"
)
SRT_TIMESTAMP_CAPTURE_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(?P<end>\d{2}:\d{2}:\d{2},\d{3})(?:\s+.*)?$"
)
SRT_RESPONSE_FENCE_RE = re.compile(r"^```(?:srt|text)?\s*\n(?P<body>.*)\n```$", re.IGNORECASE | re.DOTALL)

SRT_TRANSCRIPTION_PROMPT = """Transcribe this audio as valid SRT subtitles.

Rules:
- Output only SRT content.
- Include sequential cue numbers.
- Use timestamps in HH:MM:SS,mmm --> HH:MM:SS,mmm format.
- Put only spoken text in cue text lines.
- Do not include Markdown fences, summaries, notes, or any text outside the SRT file."""


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


class OpenAISpeechToTextProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
        client: Any | None = None,
    ) -> None:
        self.provider_name = "openai"
        self.model = model
        self.client = client if client is not None else OpenAI(api_key=api_key)

    def transcribe_file(self, audio_path: Path, *, previous_transcript: str = "") -> str:
        return self.transcribe_file_result(audio_path, previous_transcript=previous_transcript).transcript

    def transcribe_file_result(self, audio_path: Path, *, previous_transcript: str = "") -> FileTranscriptionResult:
        del previous_transcript
        with audio_path.open("rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                file=audio_file,
                model=self.model,
                response_format="diarized_json",
                chunking_strategy="auto",
                temperature=0,
            )
        return extract_openai_diarized_file_transcription_result(response)


class GeminiSpeechToTextProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_GEMINI_TRANSCRIPTION_MODEL,
        client: Any | None = None,
        prompt: str = SRT_TRANSCRIPTION_PROMPT,
    ) -> None:
        self.provider_name = "gemini"
        self.model = model
        self.client = client if client is not None else create_gemini_client(api_key)
        self.prompt = prompt

    def transcribe_file(self, audio_path: Path, *, previous_transcript: str = "") -> str:
        return self.transcribe_file_result(audio_path, previous_transcript=previous_transcript).transcript

    def transcribe_file_result(self, audio_path: Path, *, previous_transcript: str = "") -> FileTranscriptionResult:
        del previous_transcript
        uploaded_file = self.client.files.upload(file=str(audio_path))
        response = self.client.models.generate_content(
            model=self.model,
            contents=[self.prompt, uploaded_file],
            config=create_gemini_generate_content_config(temperature=0),
        )
        srt = extract_response_text(response).strip()
        return parse_srt_file_transcription_result(srt, provider_name="Gemini")


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
        return extract_response_text(response).strip()


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
        raw_srt_chunks = split_srt_by_byte_limit(raw_srt, MAX_SRT_TRANSLATION_CHUNK_BYTES)
        translated_srt_chunks = []
        for index, raw_srt_chunk in enumerate(raw_srt_chunks, start=1):
            if progress_callback is not None:
                await progress_callback(
                    "refining_transcript",
                    {
                        "index": index,
                        "total": len(raw_srt_chunks),
                        "raw_chars": len(raw_srt_chunk),
                        "raw_bytes": len(raw_srt_chunk.encode("utf-8")),
                        "model": self.refiner.model,
                    },
                )
            translated_srt_chunk = await asyncio.to_thread(self.refine_transcript, raw_srt_chunk)
            translated_srt_chunks.append(translated_srt_chunk)
        translated_srt = join_translated_srt_chunks(translated_srt_chunks)
        line_translated_transcript = render_line_translated_transcript_from_srt(translated_srt)
        if progress_callback is not None:
            await progress_callback(
                "refinement_complete",
                {
                    "translated_srt_chars": len(translated_srt),
                    "line_translated_transcript_chars": len(line_translated_transcript),
                },
            )
        return TranscriptionResult(
            raw_transcript=transcript,
            subtitle_cues=tuple(subtitle_cues),
            translated_srt=translated_srt,
            line_translated_transcript=line_translated_transcript,
        )


def create_deepgram_client(api_key: str) -> Any:
    try:
        from deepgram import DeepgramClient
    except ImportError as exc:
        raise TranscriptionError("deepgram-sdk is required for transcription.") from exc
    return DeepgramClient(api_key=api_key)


def create_gemini_client(api_key: str) -> Any:
    try:
        from google import genai
    except ImportError as exc:
        raise TranscriptionError("google-genai is required for Gemini transcription.") from exc
    return genai.Client(api_key=api_key)


def create_gemini_generate_content_config(*, temperature: int) -> Any:
    try:
        from google.genai import types
    except ImportError:
        return {"temperature": temperature}
    return types.GenerateContentConfig(temperature=temperature)


def extract_openai_diarized_file_transcription_result(response: Any) -> FileTranscriptionResult:
    segments = get_response_field(response, "segments")
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)) or not segments:
        raise TranscriptionError("OpenAI diarized transcription response did not include segments.")

    cues = []
    for segment in segments:
        cue = build_subtitle_cue(
            start=get_response_field(segment, "start"),
            end=get_response_field(segment, "end"),
            text=format_diarized_segment_text(
                speaker=get_response_field(segment, "speaker"),
                text=get_response_field(segment, "text"),
            ),
        )
        if cue is not None:
            cues.append(cue)

    if not cues:
        raise TranscriptionError("OpenAI diarized transcription response did not include usable segments.")
    srt = render_srt(cues)
    return FileTranscriptionResult(
        transcript=render_line_translated_transcript_from_srt(srt).strip(),
        subtitle_cues=tuple(cues),
    )


def format_diarized_segment_text(*, speaker: Any, text: Any) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    stripped_text = text.strip()
    if isinstance(speaker, str) and speaker.strip():
        return f"{speaker.strip()}: {stripped_text}"
    return stripped_text


def parse_srt_file_transcription_result(srt: str, *, provider_name: str) -> FileTranscriptionResult:
    cleaned_srt = clean_srt_response_text(srt)
    blocks = parse_srt_blocks(cleaned_srt)
    if not blocks:
        raise TranscriptionError(f"{provider_name} transcription response did not include SRT cues.")

    cues = []
    for block in blocks:
        cue = build_subtitle_cue_from_srt_block(block)
        if cue is None:
            raise TranscriptionError(f"{provider_name} transcription response included an invalid SRT cue.")
        cues.append(cue)
    return FileTranscriptionResult(
        transcript=render_line_translated_transcript_from_srt(render_srt_blocks(blocks)).strip(),
        subtitle_cues=tuple(cues),
    )


def clean_srt_response_text(srt: str) -> str:
    cleaned = srt.strip()
    match = SRT_RESPONSE_FENCE_RE.fullmatch(cleaned)
    if match is not None:
        return match.group("body").strip()
    return cleaned


def build_subtitle_cue_from_srt_block(block: SrtBlock) -> SubtitleCue | None:
    match = SRT_TIMESTAMP_CAPTURE_RE.fullmatch(block.timestamp)
    if match is None:
        return None
    return build_subtitle_cue(
        start=parse_srt_timestamp(match.group("start")),
        end=parse_srt_timestamp(match.group("end")),
        text="\n".join(line.strip() for line in block.text_lines if line.strip()),
    )


def parse_srt_timestamp(timestamp: str) -> float:
    hours, minutes, seconds_with_milliseconds = timestamp.split(":", 2)
    seconds, milliseconds = seconds_with_milliseconds.split(",", 1)
    return (
        int(hours) * 60 * 60
        + int(minutes) * 60
        + int(seconds)
        + int(milliseconds) / 1000
    )


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


def render_srt_blocks(blocks: Sequence[SrtBlock]) -> str:
    rendered_blocks = [
        "\n".join([block.index, block.timestamp, *block.text_lines])
        for block in blocks
    ]
    return "\n\n".join(rendered_blocks) + ("\n" if rendered_blocks else "")


def split_srt_by_byte_limit(srt: str, max_bytes: int = MAX_SRT_TRANSLATION_CHUNK_BYTES) -> tuple[str, ...]:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be greater than zero.")

    chunks: list[str] = []
    current_blocks: list[SrtBlock] = []
    for block in parse_srt_blocks(srt):
        candidate_blocks = [*current_blocks, block]
        candidate = render_srt_blocks(candidate_blocks)
        if current_blocks and len(candidate.encode("utf-8")) > max_bytes:
            chunks.append(render_srt_blocks(current_blocks))
            current_blocks = [block]
            continue
        current_blocks = candidate_blocks

    if current_blocks:
        chunks.append(render_srt_blocks(current_blocks))
    return tuple(chunks)


def assemble_srt_chunks(chunks: Sequence[str]) -> str:
    blocks: list[SrtBlock] = []
    for chunk in chunks:
        blocks.extend(parse_srt_blocks(chunk))
    return render_srt_blocks(blocks)


def join_translated_srt_chunks(chunks: Sequence[str]) -> str:
    translated_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return "\n\n".join(translated_chunks) + ("\n" if translated_chunks else "")


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


def render_line_translated_transcript_from_srt(translated_srt: str) -> str:
    text_lines = []
    for line in translated_srt.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped_line = line.strip()
        if SRT_CUE_NUMBER_RE.fullmatch(stripped_line) or SRT_TIMESTAMP_RE.fullmatch(stripped_line):
            continue
        match = GREEN_FONT_RE.fullmatch(stripped_line)
        text_lines.append(match.group(1).strip() if match is not None else stripped_line)

    while text_lines and not text_lines[0]:
        text_lines.pop(0)
    while text_lines and not text_lines[-1]:
        text_lines.pop()
    return "\n".join(text_lines) + ("\n" if text_lines else "")


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

    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text

    if isinstance(response, dict):
        text = response.get("text")
        if isinstance(text, str):
            return text

    raise TranscriptionError("OpenAI refinement response did not include text.")


def build_refinement_input(transcript: str) -> str:
    return f"{SRT_TRANSLATION_REQUEST}\n\n{RAW_TRANSCRIPT_START}\n{transcript}\n{RAW_TRANSCRIPT_END}"
