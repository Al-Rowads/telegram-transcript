from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

IANLP_REVISION = "206b0eb862808fadb3faea2ea7f12010e7897b66"
IA2D_REVISION = "7ac618dd6663d9a52acac17acd0259cdb0eff398"
MAX_IRAQI_ARABIC_CONTEXT_BYTES = 2 * 1024 * 1024
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_RETRY_DELAYS_SECONDS = (1.0, 2.0)


class TrainingResourceError(RuntimeError):
    """Raised when pinned Iraqi Arabic training resources cannot be prepared."""


@dataclass(frozen=True)
class TrainingResource:
    source: str
    relative_path: Path
    url: str
    size: int
    git_blob_sha1: str
    encoding: str = "utf-8"


IRAQI_ARABIC_TRAINING_RESOURCES = (
    TrainingResource(
        source="Iraqi Arabic NLP Toolkit (IANLP)",
        relative_path=Path("ianlp/sample_dataset.json"),
        url=(
            "https://huggingface.co/datasets/hussainhadi/"
            f"Iraqi-Arabic-NLP-Toolkit-IANLP/resolve/{IANLP_REVISION}/sample_dataset.json"
        ),
        size=9_242,
        git_blob_sha1="558a79ee1c3622645925abdc93376ad375b8359f",
    ),
    TrainingResource(
        source="Iraqi Arabic Dialect Dataset (IA2D)",
        relative_path=Path("ia2d/Annotated_Dataset_After_Preprocessing.txt"),
        url=(
            "https://raw.githubusercontent.com/ebady/Iraqi-Arabic-Dialect-Dataset/"
            f"{IA2D_REVISION}/Data/Annotated_Dataset_After_Preprocessing.txt"
        ),
        size=180_212,
        git_blob_sha1="1c08e865ddc12f0f86267646f9cfa39d0a8d79f8",
    ),
    TrainingResource(
        source="Iraqi Arabic Dialect Dataset (IA2D)",
        relative_path=Path("ia2d/Annotated_Tweets_Before_Preprocessing.txt"),
        url=(
            "https://raw.githubusercontent.com/ebady/Iraqi-Arabic-Dialect-Dataset/"
            f"{IA2D_REVISION}/Data/Annotated_Tweets_Before_Preprocessing.txt"
        ),
        size=347_751,
        git_blob_sha1="332733e29a8451e01cf96970ae9b9a2139042c51",
    ),
    TrainingResource(
        source="Iraqi Arabic Dialect Dataset (IA2D)",
        relative_path=Path("ia2d/Iraqi_Dialect_Facebook.txt"),
        url=(
            "https://raw.githubusercontent.com/ebady/Iraqi-Arabic-Dialect-Dataset/"
            f"{IA2D_REVISION}/Data/Elongated_Words/Dictionary_Raw_Data/"
            "Iraqi_Dialect_Facebook/fbds1.txt"
        ),
        size=92_358,
        git_blob_sha1="d264634ffdfe5d41fd3be281a151ff908196f3ab",
    ),
    TrainingResource(
        source="Iraqi Arabic Dialect Dataset (IA2D)",
        relative_path=Path("ia2d/ElongatedWordDictionary.txt"),
        url=(
            "https://raw.githubusercontent.com/ebady/Iraqi-Arabic-Dialect-Dataset/"
            f"{IA2D_REVISION}/Data/Elongated_Words/ElongatedWordDictionary.txt"
        ),
        size=83_920,
        git_blob_sha1="a4ccd93ac94c46e3d060ba2bc504324cbbaca943",
    ),
    TrainingResource(
        source="Iraqi Arabic Dialect Dataset (IA2D)",
        relative_path=Path("ia2d/Tweets_Raw_Data.txt"),
        url=(
            "https://raw.githubusercontent.com/ebady/Iraqi-Arabic-Dialect-Dataset/"
            f"{IA2D_REVISION}/Data/Tweets_Raw_Data.txt"
        ),
        size=210_900,
        git_blob_sha1="ce8e6901faf88540df090a61ca7b2de13c8af1a7",
        # The upstream file uses legacy Mac Arabic; Iraqi letters such as گ use
        # byte values that are undefined in ISO-8859-6.
        encoding="mac_arabic",
    ),
)


class IraqiArabicTrainingResourceManager:
    def __init__(
        self,
        root: Path,
        *,
        resources: tuple[TrainingResource, ...] = IRAQI_ARABIC_TRAINING_RESOURCES,
        client: Any | None = None,
    ) -> None:
        self.root = root
        self.resources = resources
        self.client = client

    async def ensure_available(self) -> str:
        invalid_resources = [
            resource
            for resource in self.resources
            if not resource_file_is_valid(self.root / resource.relative_path, resource)
        ]
        if invalid_resources:
            await self._download_resources(tuple(invalid_resources))
        return build_iraqi_arabic_training_context(self.root, self.resources)

    async def _download_resources(self, resources: tuple[TrainingResource, ...]) -> None:
        if self.client is not None:
            for resource in resources:
                await self._download_with_retries(self.client, resource)
            return

        timeout = httpx.Timeout(60.0, connect=10.0)
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            for resource in resources:
                await self._download_with_retries(client, resource)

    async def _download_with_retries(self, client: Any, resource: TrainingResource) -> None:
        last_error: BaseException | None = None
        for attempt in range(DOWNLOAD_ATTEMPTS):
            try:
                await self._download_resource(client, resource)
                return
            except (OSError, httpx.HTTPError, TrainingResourceError) as exc:
                last_error = exc
                if attempt < len(DOWNLOAD_RETRY_DELAYS_SECONDS):
                    await asyncio.sleep(DOWNLOAD_RETRY_DELAYS_SECONDS[attempt])
        raise TrainingResourceError(
            f"Unable to prepare Iraqi Arabic training resource {resource.relative_path}: {last_error}"
        ) from last_error

    async def _download_resource(self, client: Any, resource: TrainingResource) -> None:
        target_path = self.root / resource.relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb",
                dir=target_path.parent,
                prefix=f".{target_path.name}.",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                downloaded_bytes = 0
                digest = hashlib.sha1(f"blob {resource.size}\0".encode("ascii"))
                async with client.stream("GET", resource.url) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        downloaded_bytes += len(chunk)
                        if downloaded_bytes > resource.size:
                            raise TrainingResourceError(
                                f"Downloaded resource exceeded its pinned size: {resource.relative_path}"
                            )
                        digest.update(chunk)
                        temporary_file.write(chunk)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            if downloaded_bytes != resource.size or digest.hexdigest() != resource.git_blob_sha1:
                raise TrainingResourceError(
                    f"Downloaded resource failed integrity validation: {resource.relative_path}"
                )
            os.replace(temporary_path, target_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def resource_file_is_valid(path: Path, resource: TrainingResource) -> bool:
    try:
        if path.stat().st_size != resource.size:
            return False
        digest = hashlib.sha1(f"blob {resource.size}\0".encode("ascii"))
        with path.open("rb") as resource_file:
            for chunk in iter(lambda: resource_file.read(64 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == resource.git_blob_sha1
    except OSError:
        return False


def build_iraqi_arabic_training_context(
    root: Path,
    resources: tuple[TrainingResource, ...] = IRAQI_ARABIC_TRAINING_RESOURCES,
) -> str:
    sections = []
    for resource in resources:
        path = root / resource.relative_path
        if not resource_file_is_valid(path, resource):
            raise TrainingResourceError(f"Iraqi Arabic training resource is missing or invalid: {path}")
        try:
            content = path.read_bytes().decode(resource.encoding).strip()
        except (OSError, UnicodeDecodeError, LookupError) as exc:
            raise TrainingResourceError(f"Unable to read Iraqi Arabic training resource {path}: {exc}") from exc
        sections.append(
            f"SOURCE: {resource.source}\nFILE: {resource.relative_path.as_posix()}\n{content}"
        )

    context = "\n\n".join(sections)
    if len(context.encode("utf-8")) > MAX_IRAQI_ARABIC_CONTEXT_BYTES:
        raise TrainingResourceError("Iraqi Arabic training context exceeded its safety limit.")
    return context
