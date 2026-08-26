"""Read-only scene and tool readiness checks shared by CLI and TUI."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
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
    sharpframes_ready = not paths.images.exists() or not any(paths.images.iterdir())
    sharpframes_detail = (
        "Provide a video or image-directory source"
        if sharpframes_ready else "Images already exist; explicitly replace them from the Sharp Frames CLI"
    )
    msplat_ready = bool(images) and model is not None and model.format == "binary"
    msplat_detail = "Experimental unmasked training; valid binary COLMAP model required"
    if model is not None and model.format != "binary":
        msplat_detail = "Msplat upstream requires binary COLMAP output"
    return [
        image_state,
        Readiness("sharpframes", "ready" if sharpframes_ready else "unavailable", sharpframes_detail),
        Readiness("sammer", "ready" if images else "unavailable", "SAM3 required"),
        colmap_state,
        Readiness("brush", "ready" if model else "unavailable", "Brush requires a valid COLMAP model"),
        Readiness("msplat", "ready" if msplat_ready else "unavailable", msplat_detail),
        Readiness("lod", "ready", "Provide an explicit scene-owned PLY or SOG export"),
    ]


def inspect_tools() -> list[Readiness]:
    tools = (
        ("sharpframes", "sharp-frames"), ("sammer", "sam3"), ("colmap", "colmap"), ("brush", "brush"),
        ("msplat", "msplatter"), ("lod", "splat-transform"),
    )
    readiness = [
        Readiness(stage, "ready" if shutil.which(binary) else "unavailable", binary)
        for stage, binary in tools
    ]
    if importlib.util.find_spec("msplat") is None:
        return [
            Readiness(item.stage, "unavailable", "Install with `pip install msplat`") if item.stage == "msplat" else item
            for item in readiness
        ]
    return readiness
