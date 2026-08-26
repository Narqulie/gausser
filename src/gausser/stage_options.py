"""Versioned, engine-owned native Studio stage option specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class StageOptionError(ValueError):
    """A typed stage setting is invalid before an external tool is launched."""


@dataclass(frozen=True)
class ResolvedStageOptions:
    """The complete settings snapshot and CLI arguments for one Studio stage run."""

    stage: str
    specification_version: int
    preset: str
    values: dict[str, Any]
    arguments: tuple[str, ...]


_COLMAP_VIDEO_DEFAULTS: dict[str, Any] = {
    "matcher": "sequential",
    "use_gpu": True,
    "existing": "error",
    "mask_path": None,
    "vocab_tree": None,
}

_COLMAP_PHOTO_DEFAULTS: dict[str, Any] = {
    **_COLMAP_VIDEO_DEFAULTS,
    "matcher": "exhaustive",
}


_SHARPFRAMES_DEFAULTS: dict[str, Any] = {
    "fps": 10,
    "num_frames": 300,
    "min_buffer": 3,
    "format": "jpg",
    "selection_method": "best-n",
    "batch_size": 5,
    "batch_buffer": 2,
    "outlier_window_size": 15,
    "outlier_sensitivity": 60,
    "width": 0,
}


def catalog() -> dict[str, dict[str, Any]]:
    """Return JSON-compatible definitions for Studio stages with typed options.

    The first migrated stage is Sharp Frames. Definitions are intentionally
    authoritative Python data so the native app can render them without owning
    option defaults or command construction.
    """
    return {
        "colmap": _colmap_definition(),
        "sharpframes": {
            "version": 1,
            "title": "Sharp Frames",
            "description": "Select sharp frames from one video, a video directory, or an image directory.",
            "preset": "upstream-defaults",
            "groups": [
                {"id": "source", "title": "Source", "description": "Selected video or image source."},
                {"id": "selection", "title": "Selection", "description": "Frame sampling and selection."},
                {"id": "advanced", "title": "Advanced", "description": "Algorithm-specific controls.", "advanced": True},
            ],
            "presets": [{"recommended": True, 
                "id": "upstream-defaults",
                "title": "Upstream defaults",
                "description": "Sharp Frames 0.4 defaults; no Gausser-specific quality claim.",
                "values": _SHARPFRAMES_DEFAULTS,
            }],
            "fields": [
                _field("source", "Source", "path", None, required=True, read_only=True, group="source",
                       help="Video file, directory of videos, or image directory."),
                _field("fps", "Frames per second", "integer", 10, minimum=1, group="selection",
                       help="Sampling rate for video input."),
                _field("num_frames", "Selected frames", "integer", 300, minimum=1, group="selection",
                       help="Number of output images to retain."),
                _field("min_buffer", "Minimum buffer", "integer", 3, minimum=0, group="selection",
                       help="Minimum source frames between selected images."),
                _field("format", "Image format", "enum", "jpg", choices=["jpg", "png"], group="selection",
                       help="Published scene image format."),
                _field("selection_method", "Selection method", "enum", "best-n", group="selection",
                       choices=["best-n", "batched", "outlier-removal"],
                       help="Sharp Frames selection algorithm."),
                _field("batch_size", "Batch size", "integer", 5, minimum=1, group="advanced", advanced=True,
                       help="Items selected per batch when using batched selection."),
                _field("batch_buffer", "Batch buffer", "integer", 2, minimum=0, group="advanced", advanced=True,
                       help="Items skipped between batched selections."),
                _field("outlier_window_size", "Outlier window", "integer", 15, minimum=5, maximum=31, group="advanced", advanced=True,
                       help="Neighbourhood size for outlier-removal selection."),
                _field("outlier_sensitivity", "Outlier sensitivity", "integer", 60, minimum=0, maximum=100, group="advanced", advanced=True,
                       help="Higher values remove more outliers."),
                _field("width", "Output width", "integer", 0, minimum=0, group="advanced", advanced=True,
                       help="Resize output to this width; 0 preserves source resolution."),
            ],
        },
    }


def resolve_options(stage: str, options: Mapping[str, object]) -> ResolvedStageOptions:
    """Validate Studio options and produce deterministic CLI arguments."""
    if stage == "colmap":
        return _resolve_colmap(options)
    if stage == "sharpframes":
        return _resolve_sharpframes(options)
    raise StageOptionError(f"typed options are not available for {stage}")


def _colmap_definition() -> dict[str, Any]:
    return {
        "version": 1,
        "title": "COLMAP Reconstruction",
        "description": "Build a scene-owned camera reconstruction from the published images.",
        "preset": "video-sequence",
        "groups": [
            {"id": "matching", "title": "Matching", "description": "Choose the relationship between source images."},
            {"id": "artifacts", "title": "Existing reconstruction", "description": "Choose how existing scene-owned COLMAP artifacts are handled.", "advanced": True},
            {"id": "advanced", "title": "Advanced", "description": "Optional masks and large photo-collection support.", "advanced": True},
        ],
        "presets": [
            {"recommended": True, "id": "video-sequence", "title": "Video sequence", "description": "Sequential matching for ordered frames or a camera walkaround.", "values": _COLMAP_VIDEO_DEFAULTS},
            {"id": "photo-set", "title": "Small photo set", "description": "Exhaustive matching for an unordered, modest-sized photo set.", "values": _COLMAP_PHOTO_DEFAULTS},
        ],
        "warnings": [
            {"level": "warning", "message": "Exhaustive matching grows quadratically. Large image sets should use sequential video matching or a supplied vocabulary tree."},
            {"level": "warning", "message": "macOS COLMAP installations normally have no CUDA support; Automatic GPU uses the tool's platform-safe policy."},
        ],
        "fields": [
            _field("matcher", "Matcher", "enum", "sequential", choices=["sequential", "exhaustive", "vocab_tree"], group="matching", help="Sequential is appropriate for ordered frames; exhaustive is for small unordered photo sets."),
            _field("use_gpu", "GPU acceleration", "boolean", True, group="matching", help="Automatic platform policy; disable to force CPU execution."),
            _field("existing", "Existing artifacts", "enum", "error", choices=["error", "resume", "replace"], group="artifacts", advanced=True, help="Error is safest. Resume only after inspecting partial reconstruction state."),
            _field("mask_path", "Feature mask directory", "path", None, group="advanced", advanced=True, help="Optional explicit COLMAP mask directory. Leave empty to auto-detect scene masks."),
            _field("vocab_tree", "Vocabulary tree", "path", None, group="advanced", advanced=True, visible_when={"key": "matcher", "equals": "vocab_tree"}, help="Required when using vocabulary-tree matching."),
        ],
    }


def _resolve_colmap(options: Mapping[str, object]) -> ResolvedStageOptions:
    allowed = set(_COLMAP_VIDEO_DEFAULTS)
    unexpected = sorted(set(options) - allowed)
    if unexpected:
        raise StageOptionError(f"unsupported colmap options: {', '.join(unexpected)}")
    values: dict[str, Any] = dict(_COLMAP_VIDEO_DEFAULTS)
    values.update(options)
    _enum(values, "matcher", ("sequential", "exhaustive", "vocab_tree"))
    _enum(values, "existing", ("error", "resume", "replace"))
    if not isinstance(values["use_gpu"], bool):
        raise StageOptionError("use_gpu must be a boolean")
    for key in ("mask_path", "vocab_tree"):
        value = values[key]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise StageOptionError(f"{key} must be a non-empty path or null")
    if values["matcher"] == "vocab_tree" and values["vocab_tree"] is None:
        raise StageOptionError("vocab_tree is required when matcher is vocab_tree")

    arguments: list[str] = ["--matcher", values["matcher"], "--existing", values["existing"]]
    if not values["use_gpu"]:
        arguments.append("--no-gpu")
    if values["mask_path"] is not None:
        arguments.extend(["--mask-path", values["mask_path"]])
    if values["vocab_tree"] is not None:
        arguments.extend(["--vocab-tree", values["vocab_tree"]])
    preset = (
        "video-sequence" if values == _COLMAP_VIDEO_DEFAULTS
        else "photo-set" if values == _COLMAP_PHOTO_DEFAULTS
        else "custom"
    )
    return ResolvedStageOptions("colmap", 1, preset, values, tuple(arguments))


def _resolve_sharpframes(options: Mapping[str, object]) -> ResolvedStageOptions:
    allowed = {"source", *_SHARPFRAMES_DEFAULTS}
    unexpected = sorted(set(options) - allowed)
    if unexpected:
        raise StageOptionError(f"unsupported sharpframes options: {', '.join(unexpected)}")

    source = options.get("source")
    if not isinstance(source, str) or not source.strip():
        raise StageOptionError("source must be a non-empty path")

    values: dict[str, Any] = {"source": source, **_SHARPFRAMES_DEFAULTS}
    values.update(options)
    _positive_integer(values, "fps", minimum=1)
    _positive_integer(values, "num_frames", minimum=1)
    _positive_integer(values, "min_buffer", minimum=0)
    _positive_integer(values, "batch_size", minimum=1)
    _positive_integer(values, "batch_buffer", minimum=0)
    _positive_integer(values, "outlier_window_size", minimum=5, maximum=31)
    _positive_integer(values, "outlier_sensitivity", minimum=0, maximum=100)
    _positive_integer(values, "width", minimum=0)
    _enum(values, "format", ("jpg", "png"))
    _enum(values, "selection_method", ("best-n", "batched", "outlier-removal"))

    arguments = (
        source,
        "--fps", str(values["fps"]),
        "--num-frames", str(values["num_frames"]),
        "--min-buffer", str(values["min_buffer"]),
        "--format", values["format"],
        "--selection-method", values["selection_method"],
        "--batch-size", str(values["batch_size"]),
        "--batch-buffer", str(values["batch_buffer"]),
        "--outlier-window-size", str(values["outlier_window_size"]),
        "--outlier-sensitivity", str(values["outlier_sensitivity"]),
        "--width", str(values["width"]),
    )
    preset = "upstream-defaults" if all(values[key] == value for key, value in _SHARPFRAMES_DEFAULTS.items()) else "custom"
    return ResolvedStageOptions("sharpframes", 1, preset, values, arguments)


def _field(key: str, label: str, kind: str, default: object, **properties: object) -> dict[str, object]:
    return {"key": key, "label": label, "type": kind, "default": default, **properties}


def _positive_integer(values: dict[str, Any], name: str, *, minimum: int, maximum: int | None = None) -> None:
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise StageOptionError(f"{name} must be an integer")
    if value < minimum:
        raise StageOptionError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise StageOptionError(f"{name} must be at most {maximum}")


def _enum(values: dict[str, Any], name: str, choices: tuple[str, ...]) -> None:
    value = values[name]
    if value not in choices:
        raise StageOptionError(f"{name} must be one of: {', '.join(choices)}")
