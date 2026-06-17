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
حافظ على المعنى، الأسماء، الأرقام، والعبارات الأجنبية المعروفة.
استخدم ترقيم بسيط وقسّم الكلام بشكل مفهوم."""

IRAQI_ARABIC_SYSTEM_PROMPT = IRAQI_ARABIC_TRANSCRIPTION_PROMPT

BAGHDADI_ARABIC_REFINEMENT_SYSTEM_PROMPT = """You prepare Arabic-to-Persian video editing notes from an existing transcript.

You will receive a raw ASR/dictation transcript from a video. Keep the same transcription exactly as provided. Do not refine, correct, normalize, polish, reword, translate, remove, add, merge, split, or reorder the Arabic transcript text.

Required output:
- each line defines as a sentence ending with (?,!,.,etc...)
- max line word is 5 to 9 words. IMPORTANT MAX LINE IS 5 to 9 WORDS 
- Output only two blocks: <transcription> and <translation>.
- In <transcription>, copy the exact transcript text you received, preserving the same transcription, line order, wording, punctuation, spacing, and line breaks.
- In <translation>, work line by line from the preserved Arabic transcription.
- For each non-empty Arabic line, write exactly four lines in this order:
  1. The original Arabic line exactly as written.
  2. A natural Persian translation of that Arabic line.
  3. One key word or short key phrase from that Arabic line for video highlight timing.
  4. The unseparable words or short phrases from that Arabic line that should not be separated by a video/subtitle line break.
- Treat each preserved Arabic line as the translation unit, even if it contains more than one sentence.
- If there are no unseparable words for a line, write: none
- Do not add labels, bullets, numbering, explanations, markdown, or text outside the two required blocks.

Exact output format:
<transcription>
same Arabic transcription here
</transcription>

<translation>
+ Arabic line 1
<new-line> </new-line>
Persian translation line 1
* key word line 1
^ unseparable words line 1
<new-line> </new-line>
<seperator>----</seperator>
<new-line> </new-line>
+ Arabic line 2
<new-line> </new-line>
Persian translation line 2
<new-line> </new-line>
* key word line 2
^ unseparable words line 2
</translation>"""

BAGHDADI_ARABIC_REFINEMENT_REQUEST = (
    "Keep the same transcription exactly as provided, then translate it line by line to Persian. "
    "For each Arabic line, include the original Arabic line, its Persian translation, one key word or short key phrase "
    "for video highlighting, and the unseparable words or short phrases that should not be separated by a line break. "
    "Output only the <transcription> and <translation> blocks."
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
        refine: bool = True,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.prompt = prompt
        self.refinement_model = refinement_model
        self.refinement_system_prompt = refinement_system_prompt
        self.refine = refine
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
        if not transcript.strip() or not self.refine:
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
        if not transcript.strip() or not self.refine:
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
