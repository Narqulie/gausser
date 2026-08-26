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


_LOD_DEFAULTS: dict[str, Any] = {
    "source": "",
    "chunk_count": 512,
    "chunk_extent": 16.0,
    "lod_percentages": "100,50,25",
}


_MSPLAT_DEFAULTS: dict[str, Any] = {
    "iterations": 30000,
    "num_downscales": 0,
    "downscale_factor": 4.0,
    "no_eval": False,
    "acknowledge_unmasked": False,
}


_BRUSH_DEFAULTS: dict[str, Any] = {
    "total_train_iters": 30000,
    "export_every": 5000,
    "eval_every": 1000,
    "refine_every": 200,
    "max_splats": 1500000,
    "max_resolution": None,
    "eval_split_every": None,
}


_SAMMER_DEFAULTS: dict[str, Any] = {
    "prompt": "",
    "score_threshold": 0.5,
    "all_instances": False,
    "device": "auto",
    "checkpoint": None,
    "force": False,
    "dilation": 0,
    "feather": 0,
}


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
        "sammer": _sammer_definition(),
        "colmap": _colmap_definition(),
        "brush": _brush_definition(),
        "msplat": _msplat_definition(),
        "lod": _lod_definition(),
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
    if stage == "lod":
        return _resolve_lod(options)
    if stage == "msplat":
        return _resolve_msplat(options)
    if stage == "brush":
        return _resolve_brush(options)
    if stage == "sammer":
        return _resolve_sammer(options)
    if stage == "colmap":
        return _resolve_colmap(options)
    if stage == "sharpframes":
        return _resolve_sharpframes(options)
    raise StageOptionError(f"typed options are not available for {stage}")


def _lod_definition() -> dict[str, Any]:
    return {
        "version": 1,
        "title": "Streamed LOD",
        "description": "Create a validated PlayCanvas streamed LOD tree from one exact scene-owned PLY or SOG export.",
        "preset": "web-streaming",
        "groups": [
            {"id": "source", "title": "Source export", "description": "The selected training export is never modified."},
            {"id": "streaming", "title": "Streaming layout", "description": "Chunk and spatial partition sizing."},
        ],
        "presets": [
            {"recommended": True, "id": "web-streaming", "title": "Web streaming", "description": "Validated 100% / 50% / 25% tree with splat-transform's standard 512K chunks.", "values": _LOD_DEFAULTS},
        ],
        "fields": [
            _field("source", "Source export", "path", "", required=True, read_only=True, group="source", help="Explicit PLY or SOG artifact selected from this scene."),
            _field("lod_percentages", "LOD percentages", "text", "100,50,25", group="streaming", help="Strictly descending retained-splat percentages beginning with 100."),
            _field("chunk_count", "Chunk size (K splats)", "integer", 512, minimum=1, group="streaming", help="Approximate splats per streamed chunk in thousands."),
            _field("chunk_extent", "Chunk extent (metres)", "decimal", 16.0, minimum=0, group="streaming", help="Approximate spatial extent of a chunk."),
        ],
    }


def _resolve_lod(options: Mapping[str, object]) -> ResolvedStageOptions:
    unexpected = sorted(set(options) - set(_LOD_DEFAULTS))
    if unexpected:
        raise StageOptionError(f"unsupported lod options: {', '.join(unexpected)}")
    values: dict[str, Any] = dict(_LOD_DEFAULTS)
    values.update(options)
    if not isinstance(values["source"], str) or not values["source"].strip():
        raise StageOptionError("source must be a non-empty scene-owned export path")
    source_path = values["source"].replace("\\", "/")
    if source_path.startswith("/") or ".." in source_path.split("/") or not source_path.lower().endswith((".ply", ".sog")):
        raise StageOptionError("source must be a relative scene-owned .ply or .sog export path")
    _positive_integer(values, "chunk_count", minimum=1)
    extent = values["chunk_extent"]
    if isinstance(extent, bool) or not isinstance(extent, (int, float)) or extent <= 0:
        raise StageOptionError("chunk_extent must be greater than 0")
    raw = values["lod_percentages"]
    if not isinstance(raw, str):
        raise StageOptionError("lod_percentages must be comma-separated integers")
    try:
        percentages = [int(value) for value in raw.split(",")]
    except ValueError as exc:
        raise StageOptionError("lod_percentages must be comma-separated integers") from exc
    if len(percentages) < 2 or percentages[0] != 100 or any(not 0 < value <= 100 for value in percentages) or any(later >= earlier for earlier, later in zip(percentages, percentages[1:])):
        raise StageOptionError("lod_percentages must be strictly descending values in 1..100 beginning with 100")
    arguments = (values["source"], "--chunk-count", str(values["chunk_count"]), "--chunk-extent", str(values["chunk_extent"]), "--lod-percentages", values["lod_percentages"])
    default_values = {**_LOD_DEFAULTS, "source": values["source"]}
    preset = "web-streaming" if values == default_values else "custom"
    return ResolvedStageOptions("lod", 1, preset, values, arguments)


def _msplat_definition() -> dict[str, Any]:
    return {
        "version": 1,
        "title": "Msplat Training",
        "description": "Experimental unmasked Msplat training from a binary COLMAP model.",
        "preset": "local-memory-safe",
        "groups": [
            {"id": "training", "title": "Training", "description": "Iteration count and resolution scale."},
            {"id": "safety", "title": "Masking acknowledgement", "description": "Msplat cannot consume SAM masks."},
            {"id": "advanced", "title": "Advanced", "description": "Msplat-specific downscale and evaluation options.", "advanced": True},
        ],
        "presets": [
            {"recommended": True, "id": "local-memory-safe", "title": "Local memory safe", "description": "The verified 425-camera local configuration: factor-4 source downscale and 30k iterations.", "values": _MSPLAT_DEFAULTS},
            {"id": "validation-run", "title": "Validation run", "description": "Shorter factor-4 run for verifying reconstruction and environment wiring.", "values": {**_MSPLAT_DEFAULTS, "iterations": 7000}},
        ],
        "warnings": [
            {"level": "warning", "message": "Msplat has no mask-input support and no maximum-splats control. It cannot replace masked Brush training silently."},
            {"level": "warning", "message": "Full-resolution 425-camera training exceeded the tested 36 GB local memory envelope. Keep factor 4 unless the target machine has been evaluated."},
        ],
        "fields": [
            _field("iterations", "Training iterations", "integer", 30000, minimum=1, group="training", help="Total Msplat optimization steps."),
            _field("downscale_factor", "Image downscale factor", "decimal", 4.0, minimum=1, group="training", help="Load images at 1/N resolution. Factor 4 is the tested local-safe setting."),
            _field("acknowledge_unmasked", "I understand SAM masks are not used", "boolean", False, group="safety", required=True, help="Required explicit acknowledgement before unmasked training can start."),
            _field("num_downscales", "Internal downscale passes", "integer", 0, minimum=0, group="advanced", advanced=True, help="Additional upstream Msplat resolution passes."),
            _field("no_eval", "Skip final evaluation", "boolean", False, group="advanced", advanced=True, help="Avoid the final metrics pass when only an export is needed."),
        ],
    }


def _resolve_msplat(options: Mapping[str, object]) -> ResolvedStageOptions:
    unexpected = sorted(set(options) - set(_MSPLAT_DEFAULTS))
    if unexpected:
        raise StageOptionError(f"unsupported msplat options: {', '.join(unexpected)}")
    values: dict[str, Any] = dict(_MSPLAT_DEFAULTS)
    values.update(options)
    _positive_integer(values, "iterations", minimum=1)
    _positive_integer(values, "num_downscales", minimum=0)
    factor = values["downscale_factor"]
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or factor < 1:
        raise StageOptionError("downscale_factor must be at least 1")
    for key in ("no_eval", "acknowledge_unmasked"):
        if not isinstance(values[key], bool):
            raise StageOptionError(f"{key} must be a boolean")
    if not values["acknowledge_unmasked"]:
        raise StageOptionError("acknowledge_unmasked must be true because Msplat does not use SAM masks")
    arguments: list[str] = ["--iterations", str(values["iterations"]), "--num-downscales", str(values["num_downscales"]), "--downscale-factor", str(values["downscale_factor"])]
    if values["no_eval"]:
        arguments.append("--no-eval")
    standard = {**_MSPLAT_DEFAULTS, "acknowledge_unmasked": True}
    validation = {**standard, "iterations": 7000}
    preset = "local-memory-safe" if values == standard else "validation-run" if values == validation else "custom"
    return ResolvedStageOptions("msplat", 1, preset, values, tuple(arguments))


def _brush_definition() -> dict[str, Any]:
    return {
        "version": 1,
        "title": "Brush Training",
        "description": "Train a masked, scene-owned Gaussian splat export from the validated COLMAP model.",
        "preset": "apple-silicon-safe",
        "groups": [
            {"id": "training", "title": "Training", "description": "Training duration and artifact cadence."},
            {"id": "density", "title": "Density and memory", "description": "Controls that bound Gaussian growth and image memory."},
            {"id": "advanced", "title": "Advanced", "description": "Evaluation and refinement cadence.", "advanced": True},
        ],
        "presets": [
            {"recommended": True, "id": "apple-silicon-safe", "title": "Apple Silicon safe", "description": "30k iterations with a 1.5M-splat ceiling to avoid the observed unbounded-density Metal stall.", "values": _BRUSH_DEFAULTS},
        ],
        "warnings": [
            {"level": "warning", "message": "Brush 0.3.0 may stall after density growth on Metal when splat count is unbounded. Keep a finite max-splats value unless you accept that upstream risk."},
        ],
        "fields": [
            _field("total_train_iters", "Training iterations", "integer", 30000, minimum=1, group="training", help="Total optimization steps."),
            _field("export_every", "Export every", "integer", 5000, minimum=1, group="training", help="Write a PLY checkpoint at this interval; it must be smaller than total iterations."),
            _field("max_splats", "Maximum splats", "integer", 1500000, minimum=1, group="density", help="Finite growth ceiling; 1.5M is the tested Apple-Silicon-safe setting."),
            _field("max_resolution", "Maximum image resolution", "integer", None, minimum=1, group="density", help="Optional longest-side image cap; leave blank to use source resolution."),
            _field("eval_every", "Evaluate every", "integer", 1000, minimum=1, group="advanced", advanced=True, help="Run held-out evaluation at this interval."),
            _field("refine_every", "Refine every", "integer", 200, minimum=1, group="advanced", advanced=True, help="Densification/refinement interval."),
            _field("eval_split_every", "Hold out every Nth image", "integer", None, minimum=2, group="advanced", advanced=True, help="Optional evaluation split; leave blank to use Brush defaults."),
        ],
    }


def _resolve_brush(options: Mapping[str, object]) -> ResolvedStageOptions:
    unexpected = sorted(set(options) - set(_BRUSH_DEFAULTS))
    if unexpected:
        raise StageOptionError(f"unsupported brush options: {', '.join(unexpected)}")
    values: dict[str, Any] = dict(_BRUSH_DEFAULTS)
    values.update(options)
    for key in ("total_train_iters", "export_every", "eval_every", "refine_every", "max_splats"):
        _positive_integer(values, key, minimum=1)
    if values["export_every"] >= values["total_train_iters"]:
        raise StageOptionError("export_every must be less than total_train_iters")
    for key, minimum in (("max_resolution", 1), ("eval_split_every", 2)):
        if values[key] is not None:
            _positive_integer(values, key, minimum=minimum)

    arguments: list[str] = [
        "--total-train-iters", str(values["total_train_iters"]),
        "--export-every", str(values["export_every"]),
        "--eval-every", str(values["eval_every"]),
        "--refine-every", str(values["refine_every"]),
        "--max-splats", str(values["max_splats"]),
    ]
    if values["max_resolution"] is not None:
        arguments.extend(["--max-resolution", str(values["max_resolution"])])
    if values["eval_split_every"] is not None:
        arguments.extend(["--eval-split-every", str(values["eval_split_every"])])
    preset = "apple-silicon-safe" if values == _BRUSH_DEFAULTS else "custom"
    return ResolvedStageOptions("brush", 1, preset, values, tuple(arguments))


def _sammer_definition() -> dict[str, Any]:
    return {
        "version": 1,
        "title": "SAM3 Masking",
        "description": "Create transactional Brush and COLMAP masks from a subject prompt.",
        "preset": "standard-mask",
        "groups": [
            {"id": "subject", "title": "Subject", "description": "Describe the object to retain for reconstruction."},
            {"id": "mask", "title": "Mask quality", "description": "Confidence and instance selection."},
            {"id": "advanced", "title": "Advanced", "description": "Device, model, and output-replacement controls.", "advanced": True},
        ],
        "presets": [
            {"recommended": True, "id": "standard-mask", "title": "Standard mask", "description": "SAM3 defaults: highest-confidence instance at a 0.5 threshold.", "values": _SAMMER_DEFAULTS},
            {"id": "coverage-focus", "title": "Coverage focus", "description": "Lower threshold and light edge expansion when the standard mask misses valid subject pixels.", "values": {**_SAMMER_DEFAULTS, "score_threshold": 0.3, "dilation": 8, "feather": 12}},
        ],
        "warnings": [
            {"level": "warning", "message": "Replacing existing masks changes downstream reconstruction/training inputs. Inspect the prompt and preview the resolved command first."},
        ],
        "fields": [
            _field("prompt", "Subject prompt", "text", "", required=True, group="subject", help="Describe the object to preserve, for example ‘fountain’ or ‘statue’."),
            _field("score_threshold", "Confidence threshold", "decimal", 0.5, minimum=0, maximum=1, group="mask", help="Higher values reject uncertain detections."),
            _field("all_instances", "Merge all instances", "boolean", False, group="mask", help="Merge every detected instance instead of keeping only the highest-confidence result."),
            _field("dilation", "Mask dilation (pixels)", "integer", 0, minimum=0, group="mask", help="Expand mask edges before saving."),
            _field("feather", "Mask feather (pixels)", "integer", 0, minimum=0, group="mask", help="Blur mask edges before saving."),
            _field("device", "Compute device", "enum", "auto", choices=["auto", "mps", "cuda", "cpu"], group="advanced", advanced=True, help="Auto selects the supported local accelerator."),
            _field("checkpoint", "SAM3 checkpoint", "path", None, group="advanced", advanced=True, help="Optional local checkpoint or Hugging Face model identifier."),
            _field("force", "Replace existing masks", "boolean", False, group="advanced", advanced=True, help="Required when masks already exist; publication remains transactional."),
        ],
    }


def _resolve_sammer(options: Mapping[str, object]) -> ResolvedStageOptions:
    unexpected = sorted(set(options) - set(_SAMMER_DEFAULTS))
    if unexpected:
        raise StageOptionError(f"unsupported sammer options: {', '.join(unexpected)}")
    values: dict[str, Any] = dict(_SAMMER_DEFAULTS)
    values.update(options)
    if not isinstance(values["prompt"], str) or not values["prompt"].strip():
        raise StageOptionError("prompt must be a non-empty string")
    if isinstance(values["score_threshold"], bool) or not isinstance(values["score_threshold"], (int, float)) or not 0 <= values["score_threshold"] <= 1:
        raise StageOptionError("score_threshold must be between 0 and 1")
    for key in ("all_instances", "force"):
        if not isinstance(values[key], bool):
            raise StageOptionError(f"{key} must be a boolean")
    _enum(values, "device", ("auto", "mps", "cuda", "cpu"))
    for key in ("dilation", "feather"):
        _positive_integer(values, key, minimum=0)
    checkpoint = values["checkpoint"]
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint.strip()):
        raise StageOptionError("checkpoint must be a non-empty path or null")

    arguments: list[str] = ["--prompt", values["prompt"], "--score-threshold", str(values["score_threshold"]), "--device", values["device"], "--dilation", str(values["dilation"]), "--feather", str(values["feather"])]
    if values["all_instances"]:
        arguments.append("--all-instances")
    if checkpoint is not None:
        arguments.extend(["--checkpoint", checkpoint])
    if values["force"]:
        arguments.append("--force")
    standard = {**_SAMMER_DEFAULTS, "prompt": values["prompt"]}
    coverage = {**standard, "score_threshold": 0.3, "dilation": 8, "feather": 12}
    preset = "standard-mask" if values == standard else "coverage-focus" if values == coverage else "custom"
    return ResolvedStageOptions("sammer", 1, preset, values, tuple(arguments))


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
