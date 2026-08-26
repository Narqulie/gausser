"""Transactional photo-directory ingestion for initialized Gausser scenes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import shutil

from .manifest import fingerprint_inputs, record_stage
from .publish import replace_directory
from .scene import SceneError, ScenePaths
from .ui import list_images


class IngestPolicy(str, Enum):
    """How imported scene images retain their relationship to the source."""

    COPY = "copy"
    LINK = "link"


@dataclass(frozen=True)
class IngestResult:
    """The source and scene-owned images published by one ingestion."""

    source: Path
    policy: IngestPolicy
    images: tuple[Path, ...]


def ingest_photo_directory(
    paths: ScenePaths,
    source: Path | str,
    *,
    policy: IngestPolicy | str,
) -> IngestResult:
    """Import supported direct images from *source* into an empty initialized scene.

    The staged image set is published only after all source files have been
    materialized and validated.  If manifest publication fails, ``images/`` is
    restored to its required empty state.
    """
    _require_initialized_scene(paths)
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_dir():
        raise SceneError(f"Photo source is not a directory: {source_path}")
    if any(paths.images.iterdir()):
        raise SceneError(f"Scene image directory must be empty before ingestion: {paths.images}")

    selected = list_images(source_path)
    if not selected:
        raise SceneError(f"Photo source contains no supported images: {source_path}")
    selected_fingerprints = fingerprint_inputs(selected)
    selected_policy = _parse_policy(policy)

    attempt = paths.staging_attempt("ingest", paths.new_run_id())
    staged_images = attempt / "images"
    published = False
    try:
        staged_images.mkdir(parents=True)
        for image in selected:
            destination = staged_images / image.name
            if selected_policy is IngestPolicy.COPY:
                shutil.copy2(image, destination)
            else:
                destination.symlink_to(image)

        staged_paths = ScenePaths.from_root(attempt)
        staged_files = staged_paths.image_files()
        if len(staged_files) != len(selected):
            raise SceneError("Staged source images did not match the selected source set")

        replace_directory(staged_images, paths.images)
        published = True
        record_stage(
            paths,
            "ingest",
            {
                "source": str(source_path),
                "policy": selected_policy.value,
                "inputs": selected_fingerprints,
                "outputs": {
                    "images": str(paths.relative_path(paths.images)),
                    "count": len(staged_files),
                },
            },
        )
        return IngestResult(source_path, selected_policy, tuple(paths.image_files()))
    except Exception:
        if published:
            _restore_empty_images(paths.images)
        raise
    finally:
        shutil.rmtree(attempt, ignore_errors=True)


def _require_initialized_scene(paths: ScenePaths) -> None:
    required_directories = (paths.root, paths.images, paths.runs, paths.metadata, paths.staging)
    if not all(directory.is_dir() for directory in required_directories) or not paths.manifest.is_file():
        raise SceneError(f"Scene is not initialized: {paths.root}")


def _parse_policy(policy: IngestPolicy | str) -> IngestPolicy:
    try:
        return IngestPolicy(policy)
    except ValueError as exc:
        supported = ", ".join(item.value for item in IngestPolicy)
        raise SceneError(f"Unsupported ingest policy {policy!r}; expected one of: {supported}") from exc


def _restore_empty_images(images: Path) -> None:
    """Restore the only pre-ingestion images state allowed by this operation."""
    shutil.rmtree(images)
    images.mkdir()
