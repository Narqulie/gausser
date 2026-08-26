"""Atomic, scene-local metadata for completed workflow stages."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .scene import ScenePaths


def fingerprint_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_inputs(paths: list[Path]) -> dict[str, str]:
    return {path.name: fingerprint_file(path) for path in paths}


def initialize_manifest(paths: ScenePaths) -> None:
    """Publish the initial manifest for a newly created scene."""
    if paths.manifest.exists():
        raise ValueError(f"scene manifest already exists: {paths.manifest}")
    paths.metadata.mkdir(parents=True, exist_ok=True)
    _write_manifest(paths, {"version": 1, "stages": {}})


def record_stage(paths: ScenePaths, stage: str, data: dict[str, Any]) -> None:
    document = _load_manifest(paths)
    document["stages"][stage] = {
        "completed_at": datetime.now(UTC).isoformat(),
        **data,
    }
    _write_manifest(paths, document)


def record_run_started(paths: ScenePaths, run_id: str, stage: str, settings: dict[str, Any] | None) -> None:
    """Atomically persist an immutable resolved-settings snapshot before launch."""
    document = _load_manifest(paths)
    runs = document.setdefault("runs", {})
    if not isinstance(runs, dict) or run_id in runs:
        raise ValueError(f"invalid or duplicate manifest run: {run_id}")
    runs[run_id] = {
        "stage": stage,
        "started_at": datetime.now(UTC).isoformat(),
        "settings": settings,
        "state": "running",
    }
    _write_manifest(paths, document)


def record_run_finished(paths: ScenePaths, run_id: str, state: str, result: dict[str, Any] | None = None) -> None:
    """Atomically finalize a previously recorded run without changing its snapshot."""
    document = _load_manifest(paths)
    runs = document.setdefault("runs", {})
    entry = runs.get(run_id) if isinstance(runs, dict) else None
    if not isinstance(entry, dict):
        raise ValueError(f"unknown manifest run: {run_id}")
    entry["state"] = state
    entry["finished_at"] = datetime.now(UTC).isoformat()
    if result is not None:
        entry["result"] = result
    _write_manifest(paths, document)


def _load_manifest(paths: ScenePaths) -> dict[str, Any]:
    paths.metadata.mkdir(parents=True, exist_ok=True)
    document: dict[str, Any] = {"version": 1, "stages": {}}
    if not paths.manifest.exists():
        return document
    try:
        loaded = json.loads(paths.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid scene manifest: {paths.manifest}") from exc
    if not isinstance(loaded, dict) or loaded.get("version") != 1 or not isinstance(loaded.get("stages"), dict):
        raise ValueError(f"unsupported scene manifest: {paths.manifest}")
    return loaded


def _write_manifest(paths: ScenePaths, document: dict[str, Any]) -> None:
    with NamedTemporaryFile("w", encoding="utf-8", dir=paths.metadata, prefix="manifest-", suffix=".tmp", delete=False) as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temp_path = Path(stream.name)
    temp_path.replace(paths.manifest)
