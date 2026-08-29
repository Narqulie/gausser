"""Exception-safe publication of scene-owned artifacts and stage metadata."""

from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Sequence

from .manifest import transition_stage
from .scene import SceneError, ScenePaths
from .workflow import Generation, generation_policy, superseded_artifacts


def publish_stage(
    paths: ScenePaths,
    stage: str,
    *,
    generation: Generation | None,
    replacements: Sequence[tuple[Path, Path]],
    data: dict,
) -> Path | None:
    """Publish files plus manifest metadata as one exception-safe stage commit.

    Existing canonical artifacts are moved into a scene-local invalidation
    archive. If a rename or manifest transition fails, every artifact and the
    original manifest bytes are restored; staged outputs remain available for
    diagnosis.
    """
    normalized = tuple((source.resolve(), destination.resolve()) for source, destination in replacements)
    _validate_replacements(paths, generation, normalized)

    mutation_id = paths.new_run_id()
    archive = paths.metadata / "invalidated" / f"{stage}-{mutation_id}"
    rollback = paths.staging / f".publish-{stage}-{mutation_id}"
    if archive.exists() or rollback.exists():
        raise SceneError(f"Publication transaction already exists for {stage}: {mutation_id}")

    policy = generation_policy(generation) if generation is not None else None
    retired = list(superseded_artifacts(paths, generation)) if generation is not None else []
    for _source, destination in normalized:
        if not any(destination == owned or destination.is_relative_to(owned) for owned in retired):
            retired.append(destination)

    manifest_before = paths.manifest.read_bytes() if paths.manifest.exists() else None
    moved_originals: list[tuple[Path, Path]] = []
    moved_replacements: list[tuple[Path, Path]] = []
    archived_data = False
    created_runs = False

    try:
        for destination in retired:
            if not destination.exists() and not destination.is_symlink():
                continue
            relative = paths.relative_path(destination)
            meaningful = _artifact_has_data(destination)
            storage = (archive if meaningful else rollback) / relative
            storage.parent.mkdir(parents=True, exist_ok=True)
            destination.replace(storage)
            moved_originals.append((storage, destination))
            archived_data = archived_data or meaningful

        for source, destination in normalized:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
            moved_replacements.append((destination, source))

        if generation is not None and not paths.runs.exists():
            paths.runs.mkdir(parents=True)
            created_runs = True

        stage_data = dict(data)
        if archived_data:
            stage_data["invalidated_archive"] = str(paths.relative_path(archive))
        transition_stage(
            paths,
            stage,
            stage_data,
            invalidated_stages=policy.invalidated_stages if policy is not None else (),
        )
    except BaseException:
        for destination, source in reversed(moved_replacements):
            if destination.exists() or destination.is_symlink():
                source.parent.mkdir(parents=True, exist_ok=True)
                destination.replace(source)
        if created_runs:
            try:
                paths.runs.rmdir()
            except OSError:
                pass
        for storage, destination in reversed(moved_originals):
            if destination.is_dir() and not any(destination.iterdir()):
                destination.rmdir()
            destination.parent.mkdir(parents=True, exist_ok=True)
            storage.replace(destination)
        _restore_manifest(paths, manifest_before)
        _remove_transaction_tree(archive)
        _remove_transaction_tree(rollback)
        raise

    _remove_transaction_tree(rollback)
    if not archived_data:
        _remove_transaction_tree(archive)
        return None
    return archive


def _validate_replacements(
    paths: ScenePaths,
    generation: Generation | None,
    replacements: tuple[tuple[Path, Path], ...],
) -> None:
    if not replacements:
        raise SceneError("Stage publication requires at least one staged output")

    destinations: set[str] = set()
    for source, destination in replacements:
        if not source.exists() and not source.is_symlink():
            raise SceneError(f"Staged output does not exist: {source}")
        try:
            source.relative_to(paths.staging.resolve())
        except ValueError as exc:
            raise SceneError(f"Staged output is outside scene staging: {source}") from exc
        relative = str(paths.relative_path(destination))
        if relative in destinations:
            raise SceneError(f"Duplicate publication destination: {destination}")
        destinations.add(relative)
        if source == destination:
            raise SceneError(f"Staged output is already canonical: {source}")

    if generation is not None:
        required = generation_policy(generation).required_destinations
        missing = required - destinations
        if missing:
            raise SceneError(
                f"{generation} publication is missing canonical output(s): {', '.join(sorted(missing))}"
            )


def _artifact_has_data(path: Path) -> bool:
    if path.is_symlink() or path.is_file():
        return True
    return any(path.iterdir())


def _restore_manifest(paths: ScenePaths, contents: bytes | None) -> None:
    if contents is None:
        paths.manifest.unlink(missing_ok=True)
        return
    paths.metadata.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", dir=paths.metadata, prefix="manifest-restore-", suffix=".tmp", delete=False) as stream:
        stream.write(contents)
        temporary = Path(stream.name)
    temporary.replace(paths.manifest)


def _remove_transaction_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    parent = path.parent
    try:
        parent.rmdir()
    except OSError:
        pass
