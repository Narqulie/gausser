"""Transactional photo-directory ingestion for initialized Gausser scenes."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import shutil
from typing import Iterator

from .manifest import fingerprint_inputs, record_stage
from .publish import replace_directory
from .scene import SceneError, ScenePaths
from .ui import is_image_file, list_images


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


@dataclass(frozen=True)
class DatasetMutationResult:
    """A published source-set edit and the retired dependent artifacts."""

    added: tuple[str, ...]
    removed: tuple[str, ...]
    archive: Path


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


def mutate_photo_dataset(
    paths: ScenePaths,
    *,
    additions: list[Path | str],
    removals: list[str],
    policy: IngestPolicy | str = IngestPolicy.COPY,
) -> DatasetMutationResult:
    """Atomically edit a published source set and invalidate derived outputs.

    The authoritative ``images/`` directory is replaced as one complete flat
    generation. Masks, reconstruction state, and trainer/LOD runs were derived
    from the prior generation, so they are moved under ``.gausser/invalidated``
    rather than being silently reused or permanently deleted.
    """
    _require_initialized_scene(paths)
    selected_policy = _parse_policy(policy)
    current = {image.name: image for image in paths.image_files()}
    removal_names = _validate_removals(current, removals)
    added_sources = _validate_additions(additions)
    _validate_addition_names(current, removal_names, added_sources)
    if not added_sources and not removal_names:
        raise SceneError("Select at least one photo to add or remove")

    mutation_id = paths.new_run_id()
    attempt = paths.staging_attempt("dataset", mutation_id)
    staged_images = attempt / "images"
    archive = paths.metadata / "invalidated" / f"dataset-{mutation_id}"
    try:
        staged_images.mkdir(parents=True)
        for name, image in current.items():
            if name not in removal_names:
                _materialize_existing(image, staged_images / name)
        for source in added_sources:
            destination = staged_images / source.name
            _materialize_source(source, destination, selected_policy)

        staged_paths = ScenePaths.from_root(attempt)
        staged_files = staged_paths.image_files()
        expected_count = len(current) - len(removal_names) + len(added_sources)
        if len(staged_files) != expected_count:
            raise SceneError("Staged source images did not match the requested edit")

        staged_runs = attempt / "runs"
        staged_runs.mkdir()
        with _publish_dataset_generation(
            paths,
            staged_images=staged_images,
            staged_runs=staged_runs,
            archive=archive,
        ):
            record_stage(
                paths,
                "dataset",
                {
                    "added": [source.name for source in added_sources],
                    "removed": sorted(removal_names),
                    "policy": selected_policy.value,
                    "outputs": {"images": str(paths.relative_path(paths.images)), "count": len(staged_files)},
                    "invalidated_archive": str(paths.relative_path(archive)),
                },
            )
        return DatasetMutationResult(
            tuple(source.name for source in added_sources),
            tuple(sorted(removal_names)),
            archive,
        )
    finally:
        shutil.rmtree(attempt, ignore_errors=True)


def _validate_removals(current: dict[str, Path], removals: list[str]) -> set[str]:
    names: set[str] = set()
    for name in removals:
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise SceneError(f"Removal must be a direct filename: {name!r}")
        if name not in current:
            raise SceneError(f"Image is not in this scene: {name}")
        names.add(name)
    return names


def _validate_additions(additions: list[Path | str]) -> list[Path]:
    selected: list[Path] = []
    for value in additions:
        source = Path(value).expanduser().resolve()
        if not source.is_file() or not is_image_file(source):
            raise SceneError(f"Addition is not a supported image file: {source}")
        selected.append(source)
    if len({source.name for source in selected}) != len(selected):
        raise SceneError("Added photos must have unique filenames")
    return selected


def _validate_addition_names(current: dict[str, Path], removals: set[str], additions: list[Path]) -> None:
    names = {name for name in current if name not in removals}
    stems = {Path(name).stem for name in names}
    for source in additions:
        if source.name in names:
            raise SceneError(f"An image named {source.name!r} already exists in this scene")
        if source.stem in stems:
            raise SceneError(f"Duplicate image stem {source.stem!r}; Brush masks would collide")
        names.add(source.name)
        stems.add(source.stem)


def _materialize_existing(source: Path, destination: Path) -> None:
    """Stage existing scene ownership without following a linked source."""
    if source.is_symlink():
        destination.symlink_to(os.readlink(source))
    else:
        os.link(source, destination)


def _materialize_source(source: Path, destination: Path, policy: IngestPolicy) -> None:
    if policy is IngestPolicy.COPY:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source)


@contextmanager
def _publish_dataset_generation(
    paths: ScenePaths,
    *,
    staged_images: Path,
    staged_runs: Path,
    archive: Path,
) -> Iterator[None]:
    """Publish source images and retire every output tied to the old generation.

    Archive moves and replacements share one rollback boundary with manifest
    writing. This prevents a new source set from coexisting with a reconstruction
    or training export whose inputs no longer exist.
    """
    if archive.exists():
        raise SceneError(f"Dataset invalidation archive already exists: {archive}")
    archive.mkdir(parents=True)
    replacements = ((paths.images, staged_images), (paths.runs, staged_runs))
    retired = (
        (paths.brush_masks, archive / "masks"),
        (paths.colmap, archive / "colmap"),
        (paths.database, archive / "database.db"),
        (paths.sparse, archive / "sparse"),
        (paths.runs, archive / "runs"),
        (paths.images, archive / "images"),
    )
    moved: list[tuple[Path, Path]] = []
    published: list[tuple[Path, Path]] = []
    try:
        for source, destination in retired:
            if source.exists() or source.is_symlink():
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.replace(destination)
                moved.append((source, destination))
        for destination, staged in replacements:
            staged.replace(destination)
            published.append((destination, staged))
        yield
    except Exception:
        for destination, staged in reversed(published):
            if destination.exists() or destination.is_symlink():
                destination.replace(staged)
        for source, destination in reversed(moved):
            if destination.exists() or destination.is_symlink():
                destination.replace(source)
        raise
    else:
        # The archive is deliberately durable provenance, not a temporary backup.
        pass
    finally:
        invalidated_root = archive.parent
        if archive.exists() and not any(archive.rglob("*")):
            shutil.rmtree(archive)
        if invalidated_root.exists() and not any(invalidated_root.iterdir()):
            invalidated_root.rmdir()


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
