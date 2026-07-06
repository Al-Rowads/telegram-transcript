from __future__ import annotations

from pathlib import Path
from typing import Any

from telegram_transcript.config import Settings


class TelegramDownloadError(RuntimeError):
    """Raised when Telegram media cannot be downloaded through MTProto."""


class TelegramMediaDownloader:
    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        self.settings = settings
        self.client = client

    async def start(self) -> None:
        if self.client is None:
            try:
                from telethon import TelegramClient
                from telethon.sessions import MemorySession
            except ImportError as exc:
                raise TelegramDownloadError("telethon is required for Telegram media downloads.") from exc

            self.client = TelegramClient(
                MemorySession(),
                self.settings.telegram_api_id,
                self.settings.telegram_api_hash,
            )
        await self.client.start(bot_token=self.settings.telegram_bot_token)

    async def close(self) -> None:
        if self.client is not None:
            await self.client.disconnect()

    async def download_message_media(self, chat_id: int, message_id: int, target_path: Path) -> Path:
        if self.client is None:
            raise TelegramDownloadError("Telegram media downloader has not been started.")

        message = await self.client.get_messages(chat_id, ids=message_id)
        if message is None:
            raise TelegramDownloadError("Telegram message was not found for media download.")

        downloaded = await self.client.download_media(message, file=str(target_path))
        if downloaded is None:
            raise TelegramDownloadError("Telegram message did not include downloadable media.")
        if not target_path.exists() or target_path.stat().st_size == 0:
            raise TelegramDownloadError("Telegram media download did not produce a file.")
        return target_path
