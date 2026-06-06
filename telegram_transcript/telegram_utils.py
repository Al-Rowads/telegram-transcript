from __future__ import annotations

TELEGRAM_TEXT_LIMIT = 3900


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
