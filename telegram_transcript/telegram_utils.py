from __future__ import annotations

import textwrap

TELEGRAM_TEXT_LIMIT = 3900
TRANSCRIPT_WRAP_WIDTH = 100


def format_transcript_for_delivery(text: str, width: int = TRANSCRIPT_WRAP_WIDTH) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return ""

    lines: list[str] = []
    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue

        wrapped = textwrap.wrap(
            line,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines.extend(wrapped or [""])

    return "\n".join(lines).strip()


def split_text_for_telegram(text: str, limit: int = TELEGRAM_TEXT_LIMIT) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    if len(normalized) <= limit:
        return [normalized]

    chunks: list[str] = []
    current = ""

    for line in normalized.splitlines(keepends=True):
        while len(line) > limit:
            cut = line.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            piece = line[:cut].strip()
            if current:
                chunks.append(current.rstrip())
                current = ""
            if piece:
                chunks.append(piece)
            line = line[cut:].lstrip()

        if len(current) + len(line) > limit:
            if current.strip():
                chunks.append(current.rstrip())
            current = ""
        current += line

    if current.strip():
        chunks.append(current.rstrip())
    return chunks


def should_send_as_text(text: str, limit: int = TELEGRAM_TEXT_LIMIT) -> bool:
    return len(text.strip()) <= limit
