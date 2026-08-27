"""Read-only discovery of a scene's published image dataset."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from .scene import SceneError, ScenePaths


class DatasetChangedError(ValueError):
    """A continuation was requested for a superseded dataset generation."""


@dataclass(frozen=True)
class DatasetImage:
    """A display-safe source image and its optional published SAM mask."""

    relative_path: str
    name: str
    size_bytes: int
    mask_relative_path: str | None


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

    # Dataset publication atomically replaces directories. A read which overlaps
    # that brief switch must never assemble file details from two generations.
    for _ in range(3):
        try:
            files = paths.image_files()
            current_revision = _dataset_revision(paths, files)
            if revision is not None and revision != current_revision:
                raise DatasetChangedError("dataset changed; reload the first page")
            page_files = files[offset:offset + limit]
            images = tuple(
                DatasetImage(
                    relative_path=(Path("images") / image.name).as_posix(),
                    name=image.name,
                    size_bytes=image.stat().st_size,
                    mask_relative_path=_mask_relative_path(paths, image),
                )
                for image in page_files
            )
            # Re-scan the complete generation after building the page.
            if _dataset_revision(paths, paths.image_files()) != current_revision:
                continue
        except FileNotFoundError:
            continue
        except SceneError:
            # Only publication's short absent-directory interval is transient;
            # preserve genuine scene-layout diagnostics.
            if not paths.images.is_dir():
                continue
            raise
        next_offset = offset + len(images)
        return DatasetImagePage(images, len(files), next_offset if next_offset < len(files) else None, current_revision)
    raise DatasetChangedError("dataset changed while reading; reload the first page")


def _mask_relative_path(paths: ScenePaths, image: Path) -> str | None:
    mask = paths.brush_masks / f"{image.stem}.png"
    if not mask.is_file():
        return None
    try:
        relative = paths.relative_path(mask)
    except Exception:
        return None
    return relative.as_posix() if relative.parent == Path("masks") else None


def _dataset_revision(paths: ScenePaths, files: list[Path]) -> str:
    """Identify source and mask bytes without reading image pixels."""
    digest = hashlib.blake2b(digest_size=16)
    for image in files:
        stat = image.stat()
        digest.update(image.name.encode("utf-8"))
        digest.update(b"\\0")
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
        digest.update(b"\\0")
        mask = paths.brush_masks / f"{image.stem}.png"
        if mask.is_file():
            mask_stat = mask.stat()
            digest.update(mask.name.encode("utf-8"))
            digest.update(b"\\0")
            digest.update(f"{mask_stat.st_size}:{mask_stat.st_mtime_ns}".encode("ascii"))
            digest.update(b"\\0")
    return digest.hexdigest()
