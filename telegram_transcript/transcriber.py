from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from openai import OpenAI

DEFAULT_TRANSCRIPTION_MODEL = "gpt-4o-transcribe"
DEFAULT_REFINEMENT_MODEL = "gpt-5.4-mini"
TRANSCRIPTION_CONTEXT_CHARS = 800
RAW_TRANSCRIPT_START = "<raw_asr_transcript>"
RAW_TRANSCRIPT_END = "</raw_asr_transcript>"
ProgressCallback = Callable[[str, Mapping[str, object]], Awaitable[None]]

IRAQI_ARABIC_TRANSCRIPTION_PROMPT = """النص الصوتي باللهجة العراقية/البغدادية.
فرّغ الكلام بكتابة عراقية طبيعية وواضحة، بدون ترجمة وبدون فصحى إلا إذا المتحدث يستخدمها.
حافظ على المعنى، الأسماء، الأرقام، والعبارات الأجنبية المعروفة مثل Alain de Botton.
استخدم ترقيم بسيط وقسّم الكلام بشكل مفهوم."""

IRAQI_ARABIC_SYSTEM_PROMPT = IRAQI_ARABIC_TRANSCRIPTION_PROMPT

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
    """Raised when OpenAI returns an unusable transcription response."""


class OpenAITranscriber:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_TRANSCRIPTION_MODEL,
        prompt: str = IRAQI_ARABIC_SYSTEM_PROMPT,
        refinement_model: str = DEFAULT_REFINEMENT_MODEL,
        refinement_system_prompt: str = BAGHDADI_ARABIC_REFINEMENT_SYSTEM_PROMPT,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.prompt = prompt
        self.refinement_model = refinement_model
        self.refinement_system_prompt = refinement_system_prompt
        self.client = client if client is not None else OpenAI(api_key=api_key)

    def transcribe_file(self, audio_path: Path, *, previous_transcript: str = "") -> str:
        with audio_path.open("rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                prompt=self.build_transcription_prompt(previous_transcript),
            )
        return extract_transcript_text(response)

    def build_transcription_prompt(self, previous_transcript: str = "") -> str:
        previous_tail = previous_transcript.strip()[-TRANSCRIPTION_CONTEXT_CHARS:]
        if not previous_tail:
            return self.prompt
        return (
            f"{self.prompt}\n\n"
            "سياق آخر مقطع سابق للاستمرارية فقط، لا تكرره إلا إذا كان مسموعاً في هذا المقطع:\n"
            f"{previous_tail}"
        )

    def transcribe_chunks(self, chunks: Sequence[Path]) -> str:
        transcript = self.transcribe_chunks_raw(chunks)
        if not transcript.strip():
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
        response = self.client.responses.create(
            model=self.refinement_model,
            instructions=self.refinement_system_prompt,
            input=build_refinement_input(transcript),
            temperature=0,
        )
        refined = extract_response_text(response).strip()
        if not refined:
            raise TranscriptionError("OpenAI refinement returned empty text.")
        return refined

    async def transcribe_chunks_async(
        self,
        chunks: Sequence[Path],
        progress_callback: ProgressCallback | None = None,
    ) -> str:
        total = len(chunks)
        transcripts = []
        previous_transcript = ""
        for index, chunk in enumerate(chunks, start=1):
            if progress_callback is not None:
                await progress_callback(
                    "transcribing_chunk",
                    {
                        "index": index,
                        "total": total,
                        "chunk_bytes": chunk.stat().st_size,
                        "model": self.model,
                    },
                )
            text = (
                await asyncio.to_thread(
                    self.transcribe_file,
                    chunk,
                    previous_transcript=previous_transcript,
                )
            ).strip()
            if text:
                transcripts.append(text)
                previous_transcript = text
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
        if not transcript.strip():
            return transcript

        if progress_callback is not None:
            await progress_callback(
                "refining_transcript",
                {
                    "raw_chars": len(transcript),
                    "model": self.refinement_model,
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
        return refined


def extract_transcript_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict) and isinstance(response.get("text"), str):
        return response["text"]

    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text

    raise TranscriptionError("OpenAI transcription response did not include text.")


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
