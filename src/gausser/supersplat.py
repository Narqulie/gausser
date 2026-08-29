"""Transactional import of an explicitly edited SuperSplat export."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil

from .manifest import fingerprint_file
from .publish import publish_stage
from .scene import ScenePaths

_SUPPORTED_SUFFIXES = frozenset({".ply", ".sog", ".spz", ".splat"})


@dataclass(frozen=True)
class SuperSplatImport:
    run_id: str
    export: Path
    provenance: Path


def reimport_edited_export(
    paths: ScenePaths,
    source_export: Path | str,
    edited_export: Path | str,
    *,
    run_id: str | None = None,
) -> SuperSplatImport:
    """Publish one edited external export in a fresh, provenance-bearing run.

    Both the original source artifact and final copied export remain distinct.
    The external editor can therefore never overwrite a trainer-owned result.
    """
    source = _scene_owned_export(paths, source_export)
    edited = _external_export(edited_export)
    selected_run_id = run_id or paths.new_run_id()
    _validate_run_id(selected_run_id)
    destination = paths.runs / "supersplat" / selected_run_id
    if destination.exists():
        raise ValueError(f"SuperSplat destination already exists: {destination}")

    attempt = paths.staging_attempt("supersplat", selected_run_id)
    staged_run = attempt / "run"
    staged_export = staged_run / "exports" / f"edited{edited.suffix.lower()}"
    staged_export.parent.mkdir(parents=True)
    shutil.copy2(edited, staged_export)
    provenance = staged_run / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "editor": "SuperSplat",
                "source_artifact": str(paths.relative_path(source)),
                "source_sha256": fingerprint_file(source),
                "edited_source_filename": edited.name,
                "edited_sha256": fingerprint_file(edited),
                "export": str(Path("exports") / staged_export.name),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    published_export = destination / "exports" / staged_export.name
    published_provenance = destination / "provenance.json"
    publish_stage(
        paths,
        "supersplat",
        generation=None,
        replacements=((staged_run, destination),),
        data={
            "source": str(paths.relative_path(source)),
            "run_id": selected_run_id,
            "outputs": {
                "export": str(paths.relative_path(published_export)),
                "provenance": str(paths.relative_path(published_provenance)),
            },
        },
    )
    shutil.rmtree(attempt, ignore_errors=True)
    return SuperSplatImport(selected_run_id, published_export, published_provenance)


def _validate_run_id(value: str) -> None:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"Invalid run id: {value!r}")


def _scene_owned_export(paths: ScenePaths, value: Path | str) -> Path:
    candidate = Path(value).expanduser()
    source = candidate.resolve() if candidate.is_absolute() else (paths.root / candidate).resolve()
    try:
        relative = paths.relative_path(source)
    except Exception as exc:
        raise ValueError(f"source must be a scene-owned export: {source}") from exc
    if not source.is_file() or not relative.parts or relative.parts[0] != "runs" or source.suffix.lower() not in _SUPPORTED_SUFFIXES:
        raise ValueError(f"source must be a scene-owned export: {source}")
    return source


def _external_export(value: Path | str) -> Path:
    edited = Path(value).expanduser().resolve()
    if not edited.is_file() or edited.suffix.lower() not in _SUPPORTED_SUFFIXES:
        raise ValueError(f"edited export must be a regular PLY, SOG, SPZ, or SPLAT file: {edited}")
    return edited
