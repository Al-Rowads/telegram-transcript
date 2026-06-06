from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from openai import OpenAI

IRAQI_ARABIC_SYSTEM_PROMPT = """النص الصوتي باللهجة العراقية. اكتب التفريغ بصياغة عراقية واضحة ومهذبة بدون ترجمة للإنجليزية.
صحّح أخطاء التعرف الشائعة، خصوصاً الكلمات المتشابهة صوتياً، وحافظ على المعنى الأصلي بدون إضافة أفكار جديدة.
استخدم كتابة عراقية طبيعية مثل: هاي، شنو، ليش، جاي، يكول، احجي، بيها، مالته، علمود، هواي.
اكتب أسماء الأشخاص والكتّاب الأجانب بالإنجليزية إذا كانت معروفة، مثل: Alain de Botton.
ضع علامات ترقيم مناسبة وقسّم الكلام إلى أسطر قصيرة عند تغيّر الفكرة.
لا تستخدم الفصحى إلا إذا كان المتحدث يستخدمها فعلاً.
المطلوب: تفريغ عراقي واضح، مرتب، ومفهوم، مع الحفاظ على لهجة المتحدث."""


class TranscriptionError(RuntimeError):
    """Raised when OpenAI returns an unusable transcription response."""


class OpenAITranscriber:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        prompt: str = IRAQI_ARABIC_SYSTEM_PROMPT,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.prompt = prompt
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
