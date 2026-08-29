"""Transactional photo-directory ingestion for initialized Gausser scenes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import shutil
from typing import Sequence

from .manifest import fingerprint_inputs
from .publish import publish_stage
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
    preserve_attempt = False
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

        try:
            publish_stage(
                paths,
                "ingest",
                generation="images",
                replacements=((staged_images, paths.images),),
                data={
                    "source": str(source_path),
                    "policy": selected_policy.value,
                    "inputs": selected_fingerprints,
                    "outputs": {
                        "images": str(paths.relative_path(paths.images)),
                        "count": len(staged_files),
                    },
                },
            )
        except Exception:
            preserve_attempt = True
            raise
        return IngestResult(source_path, selected_policy, tuple(paths.image_files()))
    finally:
        if not preserve_attempt:
            shutil.rmtree(attempt, ignore_errors=True)


def mutate_photo_dataset(
    paths: ScenePaths,
    *,
    additions: Sequence[Path | str],
    removals: Sequence[str],
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
    preserve_attempt = False
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

        try:
            archive = publish_stage(
                paths,
                "dataset",
                generation="images",
                replacements=((staged_images, paths.images),),
                data={
                    "added": [source.name for source in added_sources],
                    "removed": sorted(removal_names),
                    "policy": selected_policy.value,
                    "outputs": {"images": str(paths.relative_path(paths.images)), "count": len(staged_files)},
                },
            )
        except Exception:
            preserve_attempt = True
            raise
        if archive is None:
            raise RuntimeError("dataset replacement did not retire the previous image generation")
        return DatasetMutationResult(
            tuple(source.name for source in added_sources),
            tuple(sorted(removal_names)),
            archive,
        )
    finally:
        if not preserve_attempt:
            shutil.rmtree(attempt, ignore_errors=True)


def _validate_removals(current: dict[str, Path], removals: Sequence[str]) -> set[str]:
    names: set[str] = set()
    for name in removals:
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise SceneError(f"Removal must be a direct filename: {name!r}")
        if name not in current:
            raise SceneError(f"Image is not in this scene: {name}")
        names.add(name)
    return names


def _validate_additions(additions: Sequence[Path | str]) -> list[Path]:
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
