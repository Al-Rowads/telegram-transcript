from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from openai import OpenAI

DEFAULT_REFINEMENT_MODEL = "GPT5.4-mini"

IRAQI_ARABIC_SYSTEM_PROMPT = """النص الصوتي باللهجة العراقية. اكتب التفريغ بصياغة عراقية واضحة ومهذبة بدون ترجمة للإنجليزية.
صحّح أخطاء التعرف الشائعة، خصوصاً الكلمات المتشابهة صوتياً، وحافظ على المعنى الأصلي بدون إضافة أفكار جديدة.
استخدم كتابة عراقية طبيعية مثل: هاي، شنو، ليش، جاي، يكول، احجي، بيها، مالته، علمود، هواي.
اكتب أسماء الأشخاص والكتّاب الأجانب بالإنجليزية إذا كانت معروفة، مثل: Alain de Botton.
ضع علامات ترقيم مناسبة وقسّم الكلام إلى أسطر قصيرة عند تغيّر الفكرة.
لا تستخدم الفصحى إلا إذا كان المتحدث يستخدمها فعلاً.
المطلوب: تفريغ عراقي واضح، مرتب، ومفهوم، مع الحفاظ على لهجة المتحدث."""

BAGHDADI_ARABIC_REFINEMENT_SYSTEM_PROMPT = """You are an expert Arabic transcript editor specializing in Baghdadi Iraqi Arabic.

You will receive a raw Whisper transcript from a video. Your job is to clean, refine, and sanity-check the transcript while keeping it in natural Baghdad Arabic.

Rules:
- Do not translate to Modern Standard Arabic.
- Do not change the speaker’s intended meaning.
- Do not add new ideas, explanations, or information.
- Fix obvious Whisper transcription mistakes.
- Clean up broken words, repeated false starts, awkward phrasing, and unclear slang when the intended meaning is obvious.
- Keep the wording natural, conversational, and authentic to Baghdad Arabic.
- Preserve Iraqi/Baghdadi slang and expressions when appropriate.
- Add proper punctuation and paragraph breaks for readability.
- If a word or phrase is unclear, choose the most likely Baghdadi Arabic version based on context.
- If something is impossible to confidently understand, mark it as [غير واضح].
- Keep names, brands, numbers, and technical terms accurate.
- Output only the refined Baghdadi Arabic transcript unless asked otherwise.

The final result must sound like a native Baghdad speaker naturally said it."""

BAGHDADI_ARABIC_REFINEMENT_REQUEST = (
    "Refine and sanity-check this Whisper transcript as natural Baghdadi Iraqi Arabic. "
    "Keep the meaning the same and output only the cleaned transcript:"
)


class TranscriptionError(RuntimeError):
    """Raised when OpenAI returns an unusable transcription response."""


class OpenAITranscriber:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
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

    def transcribe_file(self, audio_path: Path) -> str:
        with audio_path.open("rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                prompt=self.prompt,
            )
        return extract_transcript_text(response)

    def transcribe_chunks(self, chunks: Sequence[Path]) -> str:
        transcript = self.transcribe_chunks_raw(chunks)
        if not transcript.strip():
            return transcript
        return self.refine_transcript(transcript)

    def transcribe_chunks_raw(self, chunks: Sequence[Path]) -> str:
        transcripts = []
        for chunk in chunks:
            text = self.transcribe_file(chunk).strip()
            if text:
                transcripts.append(text)
        return "\n\n".join(transcripts)

    def refine_transcript(self, transcript: str) -> str:
        response = self.client.responses.create(
            model=self.refinement_model,
            instructions=self.refinement_system_prompt,
            input=f"{BAGHDADI_ARABIC_REFINEMENT_REQUEST}\n\n{transcript}",
        )
        return extract_response_text(response)

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
