from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from openai import OpenAI

from telegram_transcript.models import AudioChunk, FileTranscriptionResult, SubtitleCue, TranscriptionResult

DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL = "nova-3"
DEFAULT_DEEPGRAM_LANGUAGE = "ar"
DEFAULT_REFINEMENT_MODEL = "gpt-5.4-mini"
RAW_TRANSCRIPT_START = "<raw_asr_transcript>"
RAW_TRANSCRIPT_END = "</raw_asr_transcript>"
ProgressCallback = Callable[[str, Mapping[str, object]], Awaitable[None]]

BAGHDADI_ARABIC_REFINEMENT_SYSTEM_PROMPT = """You are an expert Arabic transcript editor specializing in accurate Baghdadi Iraqi Arabic.

You will receive an imperfect ASR transcript from a video. Your job is to clean, refine, and sanity-check it while preserving what the speaker actually intended.

Rules:
- Do not translate to Modern Standard Arabic.
- Do not normalize Iraqi/Baghdadi speech into formal Arabic.
- Do not change the speaker's intended meaning.
- Do not add new ideas, explanations, or information.
- Correct only high-confidence transcription mistakes.
- Clean up broken words, repeated false starts, awkward phrasing, and unclear slang only when the intended meaning is obvious from context.
- Keep the wording natural, conversational, and authentic to Baghdad Arabic.
- Preserve Iraqi/Baghdadi slang and expressions when appropriate.
- Add proper punctuation and paragraph breaks for readability.
- Keep names, brands, numbers, and technical terms accurate.
- Preserve foreign names in English when known, such as Alain de Botton.
- If a word or phrase is uncertain, choose the most likely Baghdadi Arabic version only when confidence is high.
- If confidence is low or the audio meaning cannot be recovered from context, write [غير واضح].
- Output only the final cleaned Baghdadi Iraqi Arabic transcript.

The final result must be accurate first, then polished. It must sound like a native Baghdad speaker naturally said it."""

BAGHDADI_ARABIC_REFINEMENT_REQUEST = (
    "Refine and sanity-check this ASR transcript as natural Baghdadi Iraqi Arabic. "
    "Keep the meaning exactly the same. Output only the cleaned transcript."
)


class TranscriptionError(RuntimeError):
    """Raised when a speech-to-text or refinement response is unusable."""


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
        system_prompt: str = BAGHDADI_ARABIC_REFINEMENT_SYSTEM_PROMPT,
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
        transcript = self.transcribe_chunks_raw(chunks)
        if not transcript.strip() or self.refiner is None:
            return transcript
        return self.refine_transcript(transcript)

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
        if not transcript.strip() or self.refiner is None:
            return TranscriptionResult(raw_transcript=transcript, subtitle_cues=tuple(subtitle_cues))

        if progress_callback is not None:
            await progress_callback(
                "refining_transcript",
                {
                    "raw_chars": len(transcript),
                    "model": self.refiner.model,
                },
            )
        refined = await asyncio.to_thread(self.refine_transcript, transcript)
        if progress_callback is not None:
            await progress_callback(
                "refinement_complete",
                {
                    "cleaned_chars": len(refined),
                },
            )
        return TranscriptionResult(
            raw_transcript=transcript,
            refined_transcript=refined,
            subtitle_cues=tuple(subtitle_cues),
        )


def create_deepgram_client(api_key: str) -> Any:
    try:
        from deepgram import DeepgramClient
    except ImportError as exc:
        raise TranscriptionError("deepgram-sdk is required when SPEECH_TO_TEXT_PROVIDER=deepgram.") from exc
    return DeepgramClient(api_key=api_key)


def extract_deepgram_transcript_text(response: Any) -> str:
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
    return f"{BAGHDADI_ARABIC_REFINEMENT_REQUEST}\n\n{RAW_TRANSCRIPT_START}\n{transcript}\n{RAW_TRANSCRIPT_END}"
