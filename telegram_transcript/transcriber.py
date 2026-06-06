from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from openai import OpenAI


class TranscriptionError(RuntimeError):
    """Raised when OpenAI returns an unusable transcription response."""


class OpenAITranscriber:
    def __init__(self, *, api_key: str, model: str, client: Any | None = None) -> None:
        self.model = model
        self.client = client if client is not None else OpenAI(api_key=api_key)

    def transcribe_file(self, audio_path: Path) -> str:
        with audio_path.open("rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
            )
        return extract_transcript_text(response)

    def transcribe_chunks(self, chunks: Sequence[Path]) -> str:
        transcripts = []
        for chunk in chunks:
            text = self.transcribe_file(chunk).strip()
            if text:
                transcripts.append(text)
        return "\n\n".join(transcripts)

    async def transcribe_chunks_async(self, chunks: Sequence[Path]) -> str:
        return await asyncio.to_thread(self.transcribe_chunks, chunks)


def extract_transcript_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict) and isinstance(response.get("text"), str):
        return response["text"]

    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text

    raise TranscriptionError("OpenAI transcription response did not include text.")
