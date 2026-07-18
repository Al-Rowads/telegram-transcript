from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest

from telegram_transcript import training_resources as resource_module
from telegram_transcript.training_resources import (
    IRAQI_ARABIC_TRAINING_RESOURCES,
    IraqiArabicTrainingResourceManager,
    TrainingResource,
    TrainingResourceError,
)


def make_resource(
    source: str,
    relative_path: str,
    url: str,
    payload: bytes,
    *,
    encoding: str = "utf-8",
) -> TrainingResource:
    digest = hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()
    return TrainingResource(
        source=source,
        relative_path=Path(relative_path),
        url=url,
        size=len(payload),
        git_blob_sha1=digest,
        encoding=encoding,
    )


@pytest.mark.asyncio
async def test_existing_valid_resources_are_loaded_without_network(tmp_path: Path) -> None:
    resources = (
        make_resource("IANLP", "ianlp/sample.json", "https://example.test/ianlp", "شلونك".encode()),
        make_resource("IA2D", "ia2d/tweets.txt", "https://example.test/ia2d", "هواية".encode()),
    )
    for resource, payload in zip(resources, ("شلونك".encode(), "هواية".encode()), strict=True):
        path = tmp_path / resource.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    class NetworkMustNotBeUsed:
        def stream(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("network should not be used for valid resources")

    context = await IraqiArabicTrainingResourceManager(
        tmp_path,
        resources=resources,
        client=NetworkMustNotBeUsed(),
    ).ensure_available()

    assert "SOURCE: IANLP" in context
    assert "شلونك" in context
    assert "SOURCE: IA2D" in context
    assert "هواية" in context


@pytest.mark.asyncio
async def test_missing_resources_are_downloaded_and_atomically_validated(tmp_path: Path) -> None:
    payloads = {
        "https://example.test/ianlp": "عراقي".encode(),
        "https://example.test/ia2d": "بغدادي".encode(),
    }
    resources = tuple(
        make_resource(source, path, url, payloads[url])
        for source, path, url in (
            ("IANLP", "ianlp/sample.json", "https://example.test/ianlp"),
            ("IA2D", "ia2d/tweets.txt", "https://example.test/ia2d"),
        )
    )
    requests: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(200, content=payloads[str(request.url)])

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        context = await IraqiArabicTrainingResourceManager(
            tmp_path,
            resources=resources,
            client=client,
        ).ensure_available()

    assert requests == ["https://example.test/ianlp", "https://example.test/ia2d"]
    assert (tmp_path / "ianlp/sample.json").read_bytes() == payloads[requests[0]]
    assert (tmp_path / "ia2d/tweets.txt").read_bytes() == payloads[requests[1]]
    assert "عراقي" in context
    assert "بغدادي" in context
    assert not list(tmp_path.rglob(".*.*"))


@pytest.mark.asyncio
async def test_invalid_download_keeps_existing_file_and_removes_temporary_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = "صحيح".encode()
    resource = make_resource("IA2D", "ia2d/tweets.txt", "https://example.test/ia2d", expected)
    target = tmp_path / resource.relative_path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old-corrupt-content")
    monkeypatch.setattr(resource_module, "DOWNLOAD_RETRY_DELAYS_SECONDS", ())

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"wrong"))
    ) as client:
        manager = IraqiArabicTrainingResourceManager(
            tmp_path,
            resources=(resource,),
            client=client,
        )
        with pytest.raises(TrainingResourceError, match="integrity validation"):
            await manager.ensure_available()

    assert target.read_bytes() == b"old-corrupt-content"
    assert not list(target.parent.glob(f".{target.name}.*"))


def test_production_manifest_excludes_modern_arabic_newspaper_sources() -> None:
    paths_and_urls = "\n".join(
        f"{resource.relative_path}\n{resource.url}"
        for resource in IRAQI_ARABIC_TRAINING_RESOURCES
    )

    assert "Modern_Arabic_E-Newspapers" not in paths_and_urls
    assert any("huggingface.co" in resource.url for resource in IRAQI_ARABIC_TRAINING_RESOURCES)
    assert any("raw.githubusercontent.com" in resource.url for resource in IRAQI_ARABIC_TRAINING_RESOURCES)


@pytest.mark.asyncio
async def test_legacy_iso_8859_6_resource_is_decoded_to_unicode_context(tmp_path: Path) -> None:
    arabic_text = "قيل لـ رجل عاقل"
    payload = arabic_text.encode("iso-8859-6")
    resource = make_resource(
        "IA2D",
        "ia2d/Tweets_Raw_Data.txt",
        "https://example.test/tweets",
        payload,
        encoding="iso-8859-6",
    )
    path = tmp_path / resource.relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)

    context = await IraqiArabicTrainingResourceManager(
        tmp_path,
        resources=(resource,),
    ).ensure_available()

    assert arabic_text in context


@pytest.mark.asyncio
async def test_resource_with_bytes_invalid_for_declared_encoding_is_rejected(tmp_path: Path) -> None:
    payload = b"\xe2\xea\xe4"
    resource = make_resource(
        "IA2D",
        "ia2d/invalid.txt",
        "https://example.test/invalid",
        payload,
    )
    path = tmp_path / resource.relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)

    with pytest.raises(TrainingResourceError, match="Unable to read Iraqi Arabic training resource"):
        await IraqiArabicTrainingResourceManager(
            tmp_path,
            resources=(resource,),
        ).ensure_available()


def test_production_manifest_declares_legacy_tweets_encoding() -> None:
    raw_tweets = next(
        resource
        for resource in IRAQI_ARABIC_TRAINING_RESOURCES
        if resource.relative_path.name == "Tweets_Raw_Data.txt"
    )

    assert raw_tweets.encoding == "iso-8859-6"
    assert all(
        resource.encoding == "utf-8"
        for resource in IRAQI_ARABIC_TRAINING_RESOURCES
        if resource is not raw_tweets
    )
