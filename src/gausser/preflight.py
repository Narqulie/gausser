"""Read-only scene and tool readiness checks shared by CLI and TUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from .colmap_model import read_model_stats
from .scene import SceneError, ScenePaths


@dataclass(frozen=True)
class Readiness:
    stage: str
    state: str
    detail: str


def inspect_scene(paths: ScenePaths) -> list[Readiness]:
    try:
        images = paths.image_files()
        image_state = Readiness("images", "ready", f"{len(images)} images")
    except SceneError as exc:
        images = []
        image_state = Readiness("images", "unavailable", str(exc))

    model = read_model_stats(paths.sparse / "0")
    colmap_state = (
        Readiness("colmap", "completed", f"{model.images} registered images, {model.points} points")
        if model else Readiness("colmap", "ready" if images else "unavailable", "No valid sparse/0 model")
    )
    return [
        image_state,
        Readiness("sammer", "ready" if images else "unavailable", "SAM3 required"),
        colmap_state,
        Readiness("brush", "ready" if model else "unavailable", "Brush requires a valid COLMAP model"),
    ]


def inspect_tools() -> list[Readiness]:
    tools = (("sammer", "sam3"), ("colmap", "colmap"), ("brush", "brush"))
    return [
        Readiness(stage, "ready" if shutil.which(binary) else "unavailable", binary)
        for stage, binary in tools
    ]
