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
    paths.metadata.mkdir(parents=True, exist_ok=True)
    document: dict[str, Any] = {"version": 1, "stages": {}}
    if paths.manifest.exists():
        try:
            loaded = json.loads(paths.manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid scene manifest: {paths.manifest}") from exc
        if not isinstance(loaded, dict) or loaded.get("version") != 1 or not isinstance(loaded.get("stages"), dict):
            raise ValueError(f"unsupported scene manifest: {paths.manifest}")
        document = loaded

    document["stages"][stage] = {
        "completed_at": datetime.now(UTC).isoformat(),
        **data,
    }
    _write_manifest(paths, document)


def _write_manifest(paths: ScenePaths, document: dict[str, Any]) -> None:
    with NamedTemporaryFile("w", encoding="utf-8", dir=paths.metadata, prefix="manifest-", suffix=".tmp", delete=False) as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temp_path = Path(stream.name)
    temp_path.replace(paths.manifest)
