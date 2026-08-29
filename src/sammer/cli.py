"""sammer — SAM3 masking with a beautiful Rich CLI."""

from __future__ import annotations

import argparse
import os
import sys
import time
import shutil
from pathlib import Path
from threading import Lock
from typing import Optional, TYPE_CHECKING

import numpy as np
from PIL import Image
from rich.panel import Panel
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from gausser.interactive import (
    BARE_COMMAND_EPILOG,
    InteractiveUsageError,
    prepare_stage_arguments,
)
from gausser.manifest import fingerprint_inputs
from gausser.publish import publish_stage
from gausser.scene import SceneError, ScenePaths
from gausser.scene_lock import SceneLockError, acquire_scene_lock
from gausser.ui import (
    NEON_CYAN,
    NEON_GREEN,
    NEON_MAGENTA,
    NEON_RED,
    NEON_YELLOW,
    console,
    make_summary_table,
    neon_bar,
    neon_spinner,
    stage_header,
    status_text as neon_status_text,
)

if TYPE_CHECKING:
    from .engine import ImageResult

# MPS fallback must be set before the lazy .engine import loads Torch.
# aten::_assert_async is not natively supported on MPS; without this env var
# SAM3 raises on every image on Apple Silicon.
_PYTORCH_MPS_FALLBACK = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
if _PYTORCH_MPS_FALLBACK is None or _PYTORCH_MPS_FALLBACK == "0":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# ── UI helpers ──────────────────────────────────────────────────────────────────


def _make_progress() -> Progress:
    return Progress(
        neon_spinner(style=NEON_MAGENTA),
        TextColumn("{task.description}", style=f"bold {NEON_MAGENTA}", markup=False),
        neon_bar(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[status]}", style=NEON_YELLOW, markup=False),
        TextColumn("{task.fields[hit_rate]}", style=NEON_CYAN, markup=False),
        TextColumn("{task.fields[max_conf]}", style=NEON_GREEN, markup=False),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def _header(
    image_path: Path,
    n_images: int,
    prompt: str,
    device: str,
    threshold: float,
    checkpoint: str | None,
) -> Panel:
    return stage_header(
        "SAMMER",
        "SAM3 masking signal",
        (
            ("Images", f"{image_path}  ({n_images} files)"),
            ("Target", f"“{prompt}”"),
            ("Device", device),
            ("Threshold", f"{threshold:.2f}"),
            ("Checkpoint", checkpoint or "default (facebook/sam3)"),
        ),
        accent=NEON_MAGENTA,
    )


def _print_summary(
    results: list[ImageResult],
    total: int,
    elapsed: float,
    failures_path: Path | None,
    brush_mask_dir: Path,
    colmap_mask_dir: Path,
) -> None:
    hits = sum(1 for r in results if r.success)
    misses = total - hits
    confidences = [
        r.confidence for r in results if r.success and r.confidence is not None
    ]

    console.print()
    table = make_summary_table("Masking Summary")
    table.add_row("Total images", str(total))
    table.add_row("Masked", Text(str(hits), style=NEON_GREEN))
    table.add_row("Missed", Text(str(misses), style=NEON_RED))
    hit_rate = hits / total * 100 if total > 0 else 0
    table.add_row("Hit rate", f"{hit_rate:.1f}%")
    if confidences:
        table.add_row("Mean confidence", f"{float(np.mean(confidences)):.3f}")
        table.add_row("Median confidence", f"{float(np.median(confidences)):.3f}")
        table.add_row("Min confidence", f"{float(min(confidences)):.3f}")
        table.add_row("Max confidence", f"{float(max(confidences)):.3f}")
    table.add_row("Elapsed", f"{elapsed:.1f}s")
    table.add_row("Avg per image", f"{elapsed / total:.2f}s" if total > 0 else "—")

    console.print(table)
    console.print(neon_status_text(f"Brush masks: {brush_mask_dir}/", "info"))
    console.print(neon_status_text(f"COLMAP masks: {colmap_mask_dir}/", "info"))

    if failures_path:
        console.print()
        console.print(
            neon_status_text(f"Failed files written to: {failures_path}", "warning")
        )
        console.print(
            neon_status_text(
                "Lower --score-threshold (for example, 0.3) if too many images miss",
                "info",
            )
        )


# ── Main ────────────────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="sammer",
        description="Mask images with SAM3 for 3D Gaussian Splatting pipelines.",
        epilog=BARE_COMMAND_EPILOG,
    )
    parser.add_argument(
        "--check-runtime",
        action="store_true",
        help="Validate the installed SAM3 Python API without loading model weights",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        required=False,
        help="Subject description for SAM3 (e.g. 'fountain', 'statue')",
    )
    parser.add_argument(
        "--path",
        metavar="DIR",
        default=None,
        help="Path to image folder (default: ./images/)",
    )
    parser.add_argument(
        "--scene",
        metavar="DIR",
        default=None,
        help="Scene root that owns managed artifacts (default: cwd)",
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        default=None,
        help="Deprecated alias for --scene",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="SAM3 confidence threshold 0-1 (default: 0.5)",
    )
    parser.add_argument(
        "--all-instances",
        action="store_true",
        help="Merge all detected instances (default: highest score only)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Computation device: auto, mps, cuda, cpu (default: auto)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Local path to sam3.pt or HuggingFace model ID",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run masking even if masks/ already exists",
    )
    parser.add_argument(
        "--dilation",
        type=int,
        default=0,
        help="Dilate mask by N pixels before saving",
    )
    parser.add_argument(
        "--feather",
        type=int,
        default=0,
        help="Feather mask edges by N pixels (Gaussian blur)",
    )

    try:
        prepared_argv = prepare_stage_arguments("sammer", argv, program="sammer")
    except InteractiveUsageError as exc:
        parser.error(str(exc))
    if prepared_argv is None:
        return
    args = parser.parse_args(prepared_argv)
    if args.check_runtime:
        try:
            from .engine import validate_sam3_runtime

            validate_sam3_runtime()
        except ImportError as exc:
            console.print(neon_status_text(f"SAM3 runtime unavailable: {exc}", "error"))
            raise SystemExit(1) from exc
        console.print(neon_status_text("SAM3 runtime import is ready", "success"))
        return
    if not args.prompt:
        parser.error("--prompt is required unless --check-runtime is used")

    # ── Resolve paths ──────────────────────────────────────────────────────────
    if args.scene and args.output:
        parser.error("--scene and --output cannot be used together")
    scene_paths = ScenePaths.from_root(args.scene or args.output or Path.cwd())
    if not scene_paths.root.is_dir():
        parser.error(f"scene directory does not exist: {scene_paths.root}")
    image_path = (
        Path(args.path).expanduser().resolve() if args.path else scene_paths.images
    )
    if image_path != scene_paths.images:
        parser.error("--path must resolve to the owning scene's images/ directory")

    try:
        with acquire_scene_lock(scene_paths.root):
            _run_masking(args, scene_paths, image_path)
    except SceneLockError as exc:
        parser.error(str(exc))


def _run_masking(
    args: argparse.Namespace, scene_paths: ScenePaths, image_path: Path
) -> None:
    try:
        from .engine import (
            ImageResult,
            Sam3Predictor,
            resolve_device,
            save_brush_mask,
            save_colmap_mask,
        )
    except ImportError as exc:
        console.print(
            neon_status_text(
                "Sammer runtime unavailable; install this project with pip install -e '.[sammer]'",
                "error",
            )
        )
        raise SystemExit(1) from exc

    brush_mask_dir = scene_paths.brush_masks
    colmap_mask_dir = scene_paths.colmap_masks

    # ── Validate ───────────────────────────────────────────────────────────────
    if not image_path.is_dir():
        console.print(
            neon_status_text(f"Image directory not found: {image_path}", "error")
        )
        console.print(
            "Create an [bold]images/[/bold] folder, or pass [bold]--path DIR[/bold]."
        )
        sys.exit(1)

    try:
        image_files = scene_paths.image_files()
    except SceneError as exc:
        console.print(neon_status_text(str(exc), "error"))
        sys.exit(1)
    n_images = len(image_files)
    if n_images == 0:
        console.print(neon_status_text(f"No images found in {image_path}", "error"))
        sys.exit(1)

    # Check for existing masks
    expected_brush = {f"{image.stem}.png" for image in image_files}
    expected_colmap = {f"{image.name}.png" for image in image_files}
    existing_brush = (
        {path.name for path in brush_mask_dir.glob("*.png")}
        if brush_mask_dir.is_dir()
        else set()
    )
    existing_colmap = (
        {path.name for path in colmap_mask_dir.glob("*.png")}
        if colmap_mask_dir.is_dir()
        else set()
    )
    if (existing_brush or existing_colmap) and not args.force:
        if existing_brush == expected_brush and existing_colmap == expected_colmap:
            console.print(
                neon_status_text(
                    f"Masks already exist ({len(existing_brush)} brush, {len(existing_colmap)} COLMAP)",
                    "warning",
                )
            )
            console.print("Pass [bold]--force[/bold] to re-run masking.")
            sys.exit(0)
        console.print(
            neon_status_text(
                "Existing masks are incomplete or do not match images/", "error"
            )
        )
        console.print("Pass [bold]--force[/bold] to replace them transactionally.")
        sys.exit(1)

    attempt_dir = scene_paths.staging_attempt("sammer", scene_paths.new_run_id())
    staged_brush_dir = attempt_dir / "masks"
    staged_colmap_dir = attempt_dir / "colmap-masks"

    # ── Device ─────────────────────────────────────────────────────────────────
    device = resolve_device(args.device)

    # ── UI setup ───────────────────────────────────────────────────────────────
    console.print(
        _header(
            image_path,
            n_images,
            args.prompt,
            device,
            args.score_threshold,
            args.checkpoint,
        )
    )
    console.print()

    console.print(f"[dim]Found {n_images} images, loading SAM3 model...[/dim]")

    # ── Load model ─────────────────────────────────────────────────────────────
    try:
        predictor = Sam3Predictor(device=args.device, checkpoint=args.checkpoint)
    except ImportError as exc:
        console.print(neon_status_text(str(exc), "error"))
        sys.exit(1)
    except Exception as exc:
        console.print(neon_status_text(f"Error loading SAM3 model: {exc}", "error"))
        sys.exit(1)
    console.print("[dim]Model loaded. Starting masking...[/dim]\n")

    # ── Masking loop ───────────────────────────────────────────────────────────
    progress = _make_progress()
    task_id = progress.add_task(
        "Masking", total=n_images, status="", hit_rate="", max_conf=""
    )

    overall_start = time.monotonic()
    results: list[ImageResult] = []
    hit_count = 0
    lock = Lock()

    with progress:
        for i, img_path in enumerate(image_files):
            seg_results: list = []
            error_msg: str | None = None
            try:
                image = Image.open(img_path).convert("RGB")
                seg_results = predictor.predict(
                    image,
                    prompt=args.prompt,
                    score_threshold=args.score_threshold,
                )
            except Exception as exc:
                error_msg = str(exc)

            if seg_results and not error_msg:
                merged = (
                    Sam3Predictor.merge_masks(seg_results)
                    if args.all_instances
                    else seg_results[0].mask.copy()
                )
                max_conf = max(r.score for r in seg_results)
                assert merged is not None

                save_brush_mask(
                    merged,
                    img_path,
                    staged_brush_dir,
                    dilation_px=args.dilation,
                    feather_px=args.feather,
                )
                save_colmap_mask(merged, img_path, staged_colmap_dir)

                with lock:
                    hit_count += 1
                    results.append(
                        ImageResult(
                            image_name=img_path.name, success=True, confidence=max_conf
                        )
                    )
                    status_text = f"✓ {img_path.name[:40]}"
                    conf_str = f"c={max_conf:.3f}"
            else:
                with lock:
                    results.append(
                        ImageResult(
                            image_name=img_path.name,
                            success=False,
                            confidence=None,
                            error=error_msg,
                        )
                    )
                    status_text = f"✗ {img_path.name[:40]}"
                    conf_str = ""

            with lock:
                hit_rate_str = f"{hit_count}/{i + 1}"
                progress.update(
                    task_id,
                    advance=1 if i > 0 else 0,
                    status=status_text,
                    hit_rate=hit_rate_str,
                    max_conf=conf_str,
                )
            progress.update(task_id, completed=i + 1)

    elapsed = time.monotonic() - overall_start

    # ── Write failures ─────────────────────────────────────────────────────────
    failures = [r for r in results if not r.success]
    failures_path: Path | None = None
    if failures:
        attempt_dir.mkdir(parents=True, exist_ok=True)
        failures_path = attempt_dir / "failures.txt"
        with open(failures_path, "w") as f:
            for r in failures:
                f.write(f"{r.image_name}")
                if r.error:
                    f.write(f"  # {r.error}")
                f.write("\n")

    if failures:
        console.print(
            neon_status_text(
                "Masks were not published because one or more images failed", "warning"
            )
        )
        console.print(
            neon_status_text(f"Staged attempt retained at: {attempt_dir}", "muted")
        )
        _print_summary(
            results,
            n_images,
            elapsed,
            failures_path,
            staged_brush_dir,
            staged_colmap_dir,
        )
        sys.exit(1)

    try:
        publish_stage(
            scene_paths,
            "sammer",
            generation="masks",
            replacements=(
                (staged_brush_dir, brush_mask_dir),
                (staged_colmap_dir, colmap_mask_dir),
            ),
            data={
                "inputs": fingerprint_inputs(image_files),
                "parameters": {
                    "prompt": args.prompt,
                    "score_threshold": args.score_threshold,
                    "all_instances": args.all_instances,
                    "dilation": args.dilation,
                    "feather": args.feather,
                    "device": device,
                    "checkpoint": args.checkpoint,
                },
                "outputs": {
                    "brush_masks": str(scene_paths.relative_path(brush_mask_dir)),
                    "colmap_masks": str(scene_paths.relative_path(colmap_mask_dir)),
                },
            },
        )
        shutil.rmtree(attempt_dir, ignore_errors=True)
    except Exception as exc:
        console.print(neon_status_text(f"Error publishing masks: {exc}", "error"))
        sys.exit(1)

    _print_summary(results, n_images, elapsed, None, brush_mask_dir, colmap_mask_dir)
    sys.exit(0)
