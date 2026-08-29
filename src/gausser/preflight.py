"""Read-only scene and tool readiness checks shared by interactive clients."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import importlib
from pathlib import Path
import subprocess
from typing import Any, Literal, Mapping

from .artifacts import Artifact, list_artifacts
from .colmap_model import read_model_stats
from .process import command_environment, resolve_command
from .scene import SceneError, ScenePaths


RequiredAction = Literal["none", "force", "existing_policy"]


@dataclass(frozen=True)
class Readiness:
    stage: str
    state: str
    detail: str
    required_action: RequiredAction = "none"


def _sammer_readiness(paths: ScenePaths, images: Sequence[Path]) -> Readiness:
    if not images:
        return Readiness("sammer", "unavailable", "SAM3 requires scene images")

    expected_brush = {f"{image.stem}.png" for image in images}
    expected_colmap = {f"{image.name}.png" for image in images}
    existing_brush = (
        {path.name for path in paths.brush_masks.glob("*.png")}
        if paths.brush_masks.is_dir()
        else set()
    )
    existing_colmap = (
        {path.name for path in paths.colmap_masks.glob("*.png")}
        if paths.colmap_masks.is_dir()
        else set()
    )
    if existing_brush == expected_brush and existing_colmap == expected_colmap:
        return Readiness(
            "sammer",
            "completed",
            f"Complete masks for {len(images)} images; rerunning replaces them",
            required_action="force",
        )
    if existing_brush or existing_colmap:
        return Readiness(
            "sammer",
            "ready",
            "Existing masks are incomplete or do not match images; replacement is required",
            required_action="force",
        )
    return Readiness("sammer", "ready", "No masks published")


def inspect_scene(paths: ScenePaths, artifacts: Sequence[Artifact] | None = None) -> list[Readiness]:
    try:
        images = paths.image_files()
        image_state = Readiness(
            "images",
            "ready" if images else "unavailable",
            f"{len(images)} images",
        )
    except SceneError as exc:
        images = []
        image_state = Readiness("images", "unavailable", str(exc))

    model = read_model_stats(paths.sparse / "0")
    colmap_artifacts_exist = paths.database.exists() or paths.sparse.exists()
    if model:
        colmap_state = Readiness(
            "colmap",
            "completed",
            f"{model.images} registered images, {model.points} points",
            required_action="existing_policy",
        )
    elif images and colmap_artifacts_exist:
        colmap_state = Readiness(
            "colmap",
            "ready",
            "Existing COLMAP artifacts are incomplete or invalid; resume or replacement is required",
            required_action="existing_policy",
        )
    else:
        colmap_state = Readiness(
            "colmap",
            "ready" if images else "unavailable",
            "No valid sparse/0 model",
        )
    sharpframes_ready = not images
    sharpframes_state = Readiness(
        "sharpframes",
        "ready" if sharpframes_ready else "completed",
        (
            "Provide a video or image-directory source"
            if sharpframes_ready
            else f"{len(images)} scene images exist; selecting new frames replaces them"
        ),
        required_action="none" if sharpframes_ready else "force",
    )
    sammer_state = _sammer_readiness(paths, images)
    msplat_ready = bool(images) and model is not None and model.format == "binary"
    msplat_detail = "Experimental unmasked training; valid binary COLMAP model required"
    if model is not None and model.format != "binary":
        msplat_detail = "Msplat upstream requires binary COLMAP output"

    scene_artifacts = list_artifacts(paths) if artifacts is None else artifacts
    lod_publications = [
        artifact for artifact in scene_artifacts if artifact.kind == "lod" and artifact.format == "lod"
    ]
    lod_sources = [
        artifact
        for artifact in scene_artifacts
        if artifact.kind != "lod" and artifact.format in {"ply", "sog"}
    ]
    if lod_publications:
        lod_state = Readiness("lod", "completed", f"{len(lod_publications)} published LOD run(s)")
    elif lod_sources:
        preview = lod_sources[0].relative_path
        lod_state = Readiness("lod", "ready", f"Scene-owned source available: {preview}")
    else:
        lod_state = Readiness("lod", "unavailable", "No scene-owned PLY or SOG export")
    return [
        image_state,
        sharpframes_state,
        sammer_state,
        colmap_state,
        Readiness("brush", "ready" if model else "unavailable", "Brush requires a valid COLMAP model"),
        Readiness("msplat", "ready" if msplat_ready else "unavailable", msplat_detail),
        lod_state,
    ]


_STAGE_TITLES = {
    "sharpframes": "Sharp Frames",
    "sammer": "Sammer",
    "colmap": "COLMAP",
    "brush": "Brusher",
    "msplat": "Msplatter",
    "lod": "LOD Builder",
}


def validate_stage_readiness(
    paths: ScenePaths,
    stage: str,
    values: Mapping[str, Any],
) -> Readiness:
    """Validate a resolved run against scene state while its mutation lock is held."""
    statuses = {status.stage: status for status in inspect_scene(paths)}
    try:
        status = statuses[stage]
        title = _STAGE_TITLES[stage]
    except KeyError as exc:
        raise ValueError(f"unknown stage: {stage}") from exc
    if status.state not in {"ready", "completed"}:
        raise ValueError(f"{title} is unavailable: {status.detail}")
    if status.required_action == "force" and values.get("force") is not True:
        raise ValueError(f"{title} requires explicit replacement confirmation")
    if status.required_action == "existing_policy" and values.get("existing") not in {
        "resume",
        "replace",
    }:
        raise ValueError("COLMAP requires an explicit resume or replace policy")
    return status


def _command_runtime_status(
    stage: str,
    command: str | None,
    *,
    success_detail: str,
    missing_detail: str,
) -> Readiness:
    """Validate a Python-backed stage in the command environment that will run it."""
    if not command:
        return Readiness(stage, "unavailable", missing_detail)
    try:
        result = subprocess.run(
            [command, "--check-runtime"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            env=command_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Readiness(stage, "unavailable", f"Runtime check failed: {exc}")
    if result.returncode == 0:
        return Readiness(stage, "ready", success_detail)
    detail = (result.stderr or result.stdout).strip().splitlines()
    return Readiness(
        stage,
        "unavailable",
        detail[-1] if detail else "Runtime check failed",
    )


def _current_sammer_runtime_status() -> Readiness:
    """Validate SAM3 in the interpreter already running the direct CLI."""
    try:
        from sammer.engine import validate_sam3_runtime

        validate_sam3_runtime()
    except (ImportError, OSError, RuntimeError) as exc:
        detail = str(exc).strip().splitlines()
        return Readiness(
            "sammer",
            "unavailable",
            detail[-1] if detail else "SAM3 runtime import failed",
        )
    return Readiness("sammer", "ready", "Current interpreter has a compatible SAM3 runtime")


def _current_msplat_runtime_status() -> Readiness:
    """Validate Msplat in the interpreter already running the direct CLI."""
    try:
        runtime = importlib.import_module("msplat")
        version = getattr(runtime, "__version__", None)
        if version != "1.1.4":
            raise RuntimeError(f"Msplat 1.1.4 is required; found {version or 'unknown'}")
    except (ImportError, OSError, RuntimeError) as exc:
        detail = str(exc).strip().splitlines()
        return Readiness(
            "msplat",
            "unavailable",
            detail[-1] if detail else "Msplat runtime import failed",
        )
    return Readiness("msplat", "ready", "Current interpreter has Msplat 1.1.4")


def _commands_status(stage: str, commands: Sequence[str]) -> Readiness:
    missing = [command for command in commands if resolve_command(command) is None]
    return Readiness(
        stage,
        "unavailable" if missing else "ready",
        (
            f"Missing required command(s): {', '.join(missing)}"
            if missing
            else " + ".join(commands)
        ),
    )


def inspect_direct_tool(stage: str) -> Readiness:
    """Validate only dependencies used by the already-running direct stage CLI."""
    if stage == "sharpframes":
        return _commands_status(stage, ("sharp-frames", "ffmpeg", "ffprobe"))
    if stage == "sammer":
        return _current_sammer_runtime_status()
    if stage == "colmap":
        return _commands_status(stage, ("colmap",))
    if stage == "brush":
        return _commands_status(stage, ("brush",))
    if stage == "msplat":
        return _current_msplat_runtime_status()
    if stage == "lod":
        return _commands_status(stage, ("splat-transform",))
    raise ValueError(f"unknown stage: {stage}")


def inspect_launch_tool(stage: str) -> Readiness:
    """Validate the stage command and dependencies a launcher will spawn."""
    if stage == "sharpframes":
        return _commands_status(
            stage, ("sharpframes", "sharp-frames", "ffmpeg", "ffprobe")
        )
    if stage == "sammer":
        return _command_runtime_status(
            stage,
            resolve_command("sammer"),
            success_detail="sammer + compatible SAM3 runtime",
            missing_detail="Install the sammer command and SAM3 runtime",
        )
    if stage == "colmap":
        return _commands_status(stage, ("colmapper", "colmap"))
    if stage == "brush":
        return _commands_status(stage, ("brusher", "brush"))
    if stage == "msplat":
        return _command_runtime_status(
            stage,
            resolve_command("msplatter"),
            success_detail="msplatter + compatible Msplat runtime",
            missing_detail="Install the msplatter command and Msplat runtime",
        )
    if stage == "lod":
        return _commands_status(stage, ("lodder", "splat-transform"))
    raise ValueError(f"unknown stage: {stage}")


def inspect_tools() -> list[Readiness]:
    """Report every stage runtime used by launcher and engine clients."""
    return [
        inspect_launch_tool(stage)
        for stage in ("sharpframes", "sammer", "colmap", "brush", "msplat", "lod")
    ]
