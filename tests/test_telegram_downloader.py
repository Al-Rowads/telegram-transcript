from __future__ import annotations

from pathlib import Path

import pytest

from telegram_transcript.config import Settings
from telegram_transcript.telegram_downloader import TelegramDownloadError, TelegramMediaDownloader


class FakeTelethonClient:
    def __init__(self, *, message: object | None = object(), media_bytes: bytes | None = b"media") -> None:
        self.message = message
        self.media_bytes = media_bytes
        self.started_with: str | None = None
        self.disconnected = False
        self.message_requests: list[tuple[int, int]] = []
        self.downloads: list[tuple[object, Path]] = []

    async def start(self, *, bot_token: str) -> None:
        self.started_with = bot_token

    async def disconnect(self) -> None:
        self.disconnected = True

    async def get_messages(self, chat_id: int, *, ids: int) -> object | None:
        self.message_requests.append((chat_id, ids))
        return self.message

    async def download_media(self, message: object, *, file: str) -> str | None:
        target_path = Path(file)
        self.downloads.append((message, target_path))
        if self.media_bytes is None:
            return None
        target_path.write_bytes(self.media_bytes)
        return file


def make_settings() -> Settings:
    return Settings(
        telegram_bot_token="bot-token",
        telegram_api_id=12345,
        telegram_api_hash="telegram-api-hash",
    )


@pytest.mark.asyncio
async def test_telegram_media_downloader_starts_and_closes_client() -> None:
    client = FakeTelethonClient()
    downloader = TelegramMediaDownloader(make_settings(), client=client)

    await downloader.start()
    await downloader.close()

    assert client.started_with == "bot-token"
    assert client.disconnected is True


@pytest.mark.asyncio
async def test_telegram_media_downloader_downloads_message_media(tmp_path: Path) -> None:
    message = object()
    client = FakeTelethonClient(message=message, media_bytes=b"video")
    downloader = TelegramMediaDownloader(make_settings(), client=client)
    await downloader.start()

    target_path = tmp_path / "source.mp4"
    result = await downloader.download_message_media(100, 42, target_path)

    assert result == target_path
    assert target_path.read_bytes() == b"video"
    assert client.message_requests == [(100, 42)]
    assert client.downloads == [(message, target_path)]


@pytest.mark.asyncio
async def test_telegram_media_downloader_rejects_missing_message(tmp_path: Path) -> None:
    client = FakeTelethonClient(message=None)
    downloader = TelegramMediaDownloader(make_settings(), client=client)
    await downloader.start()

    with pytest.raises(TelegramDownloadError, match="not found"):
        await downloader.download_message_media(100, 42, tmp_path / "source.mp4")


@pytest.mark.asyncio
async def test_telegram_media_downloader_rejects_missing_media(tmp_path: Path) -> None:
    client = FakeTelethonClient(media_bytes=None)
    downloader = TelegramMediaDownloader(make_settings(), client=client)
    await downloader.start()

    with pytest.raises(TelegramDownloadError, match="downloadable media"):
        await downloader.download_message_media(100, 42, tmp_path / "source.mp4")


@pytest.mark.asyncio
async def test_telegram_media_downloader_requires_start(tmp_path: Path) -> None:
    downloader = TelegramMediaDownloader(make_settings())

    with pytest.raises(TelegramDownloadError, match="not been started"):
        await downloader.download_message_media(100, 42, tmp_path / "source.mp4")
