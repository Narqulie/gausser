"""Read-only discovery of a scene's published image dataset."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from .scene import ScenePaths


class DatasetChangedError(ValueError):
    """A continuation was requested for a superseded dataset generation."""


@dataclass(frozen=True)
class DatasetImage:
    """A display-safe reference to one canonical scene image."""

    relative_path: str
    name: str
    size_bytes: int


@dataclass(frozen=True)
class DatasetImagePage:
    images: tuple[DatasetImage, ...]
    total_count: int
    next_offset: int | None
    revision: str


def list_dataset_images(
    paths: ScenePaths,
    *,
    offset: int = 0,
    limit: int = 200,
    revision: str | None = None,
) -> DatasetImagePage:
    """List a bounded page of canonical direct ``images/`` children.

    Image bytes remain in the scene; the native client resolves only these
    engine-derived relative paths while its project security scope is active.
    """
    if offset < 0:
        raise ValueError("offset must not be negative")
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")

    files = paths.image_files()
    current_revision = _dataset_revision(files)
    if revision is not None and revision != current_revision:
        raise DatasetChangedError("dataset changed; reload the first page")
    page_files = files[offset:offset + limit]
    images = tuple(
        DatasetImage(
            relative_path=(Path("images") / image.name).as_posix(),
            name=image.name,
            size_bytes=image.stat().st_size,
        )
        for image in page_files
    )
    next_offset = offset + len(images)
    return DatasetImagePage(images, len(files), next_offset if next_offset < len(files) else None, current_revision)


def _dataset_revision(files: list[Path]) -> str:
    """Identify one published dataset generation without reading image pixels."""
    digest = hashlib.blake2b(digest_size=16)
    for image in files:
        stat = image.stat()
        digest.update(image.name.encode("utf-8"))
        digest.update(b"\\0")
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
        digest.update(b"\\0")
    return digest.hexdigest()
