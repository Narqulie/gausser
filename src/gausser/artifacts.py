"""Discovery of scene-owned deliverables for native and headless clients."""

from __future__ import annotations

from dataclasses import dataclass

from .scene import SceneError, ScenePaths


_VIEWABLE_SUFFIXES = frozenset({".ply", ".sog", ".spz", ".splat"})


@dataclass(frozen=True)
class Artifact:
    relative_path: str
    kind: str
    run_id: str
    format: str
    size_bytes: int


def list_artifacts(paths: ScenePaths) -> list[Artifact]:
    """Return direct, scene-owned training exports and published LOD metadata."""
    if not paths.runs.is_dir():
        return []

    artifacts: list[Artifact] = []
    for candidate in paths.runs.rglob("*"):
        if not candidate.is_file():
            continue
        suffix = candidate.suffix.casefold()
        is_lod_metadata = candidate.name == "lod-meta.json"
        if suffix not in _VIEWABLE_SUFFIXES and not is_lod_metadata:
            continue
        try:
            relative = paths.relative_path(candidate)
        except SceneError:
            # A symlinked export outside this scene must never become a trusted artifact.
            continue
        parts = relative.parts
        if len(parts) < 3 or parts[0] != "runs":
            continue
        kind, run_id = parts[1], parts[2]
        artifacts.append(
            Artifact(
                relative_path=relative.as_posix(),
                kind=kind,
                run_id=run_id,
                format="lod" if is_lod_metadata else suffix.removeprefix("."),
                size_bytes=candidate.stat().st_size,
            )
        )
    return sorted(artifacts, key=lambda artifact: artifact.relative_path)
