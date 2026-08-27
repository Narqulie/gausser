"""sammer — SAM3 masking with a beautiful Rich CLI."""

from __future__ import annotations

import argparse
import os
import sys
import time
import shutil
from pathlib import Path
from uuid import uuid4
from threading import Lock
from typing import Optional

# MPS fallback must be set before torch is imported (which happens via .engine)
# aten::_assert_async is not natively supported on MPS; without this env var
# SAM3 raises on every image on Apple Silicon.
_PYTORCH_MPS_FALLBACK = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
if _PYTORCH_MPS_FALLBACK is None or _PYTORCH_MPS_FALLBACK == "0":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
from PIL import Image
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from gausser.manifest import fingerprint_inputs, record_stage
from gausser.scene import SceneError, ScenePaths
from gausser.ui import console, list_images, make_summary_table

from .engine import (
    ImageResult,
    Sam3Predictor,
    resolve_device,
    validate_sam3_runtime,
    save_brush_mask,
    save_colmap_mask,
)


def _publish_masks(
    staged_brush: Path,
    staged_colmap: Path,
    brush_masks: Path,
    colmap_masks: Path,
) -> None:
    """Replace both mask sets with a completed staged attempt."""
    destinations = ((Path(staged_brush), Path(brush_masks)), (Path(staged_colmap), Path(colmap_masks)))
    backups: list[tuple[Path, Path]] = []
    published: list[tuple[Path, Path]] = []
    token = uuid4().hex
    try:
        for staged, destination in destinations:
            if not staged.is_dir():
                raise ValueError(f"staged mask directory does not exist: {staged}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                backup = destination.with_name(f".{destination.name}.backup-{token}")
                destination.replace(backup)
                backups.append((backup, destination))
            staged.replace(destination)
            published.append((staged, destination))
    except Exception:
        for staged, destination in reversed(published):
            if destination.exists():
                destination.replace(staged)
        for backup, destination in reversed(backups):
            if backup.exists():
                backup.replace(destination)
        raise
    else:
        for backup, _ in backups:
            shutil.rmtree(backup, ignore_errors=True)


# ── UI helpers ──────────────────────────────────────────────────────────────────


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        BarColumn(bar_width=36),
        MofNCompleteColumn(),
        TextColumn("[yellow]{task.fields[status]}"),
        TextColumn("[cyan]{task.fields[hit_rate]}"),
        TextColumn("[magenta]{task.fields[max_conf]}"),
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
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", no_wrap=True)
    grid.add_column()
    grid.add_row("Images", f"{image_path}  ({n_images} files)")
    grid.add_row("Prompt", f"'{prompt}'")
    grid.add_row("Device", device)
    grid.add_row("Threshold", f"{threshold:.2f}")
    grid.add_row("Checkpoint", checkpoint or "default (facebook/sam3)")
    return Panel(
        grid,
        title="[bold]SAMMER[/bold] · SAM3 Masking Pipeline",
        border_style="green",
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
    confidences = [r.confidence for r in results if r.success and r.confidence is not None]

    console.print()
    table = make_summary_table("Masking Summary")
    table.add_row("Total images", str(total))
    table.add_row("Masked", f"[green]{hits}[/green]")
    table.add_row("Missed", f"[red]{misses}[/red]")
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
    console.print(f"  Brush masks:  [bold]{brush_mask_dir}/[/bold]")
    console.print(f"  COLMAP masks: [bold]{colmap_mask_dir}/[/bold]")

    if failures_path:
        console.print(f"\n[yellow]Failed files written to:[/yellow] {failures_path}")
        console.print("[yellow]Tip:[/yellow] Lower --score-threshold (e.g. 0.3) if too many misses.")


# ── Main ────────────────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="sammer",
        description="Mask images with SAM3 for 3D Gaussian Splatting pipelines.",
    )
    parser.add_argument(
        "--check-runtime", action="store_true",
        help="Validate the installed SAM3 Python API without loading model weights",
    )
    parser.add_argument(
        "--prompt", "-p", required=False,
        help="Subject description for SAM3 (e.g. 'fountain', 'statue')",
    )
    parser.add_argument(
        "--path", metavar="DIR", default=None,
        help="Path to image folder (default: ./images/)",
    )
    parser.add_argument(
        "--scene", metavar="DIR", default=None,
        help="Scene root that owns managed artifacts (default: cwd)",
    )
    parser.add_argument(
        "--output", metavar="DIR", default=None,
        help="Deprecated alias for --scene",
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.5,
        help="SAM3 confidence threshold 0-1 (default: 0.5)",
    )
    parser.add_argument(
        "--all-instances", action="store_true",
        help="Merge all detected instances (default: highest score only)",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Computation device: auto, mps, cuda, cpu (default: auto)",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Local path to sam3.pt or HuggingFace model ID",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run masking even if masks/ already exists",
    )
    parser.add_argument(
        "--dilation", type=int, default=0,
        help="Dilate mask by N pixels before saving",
    )
    parser.add_argument(
        "--feather", type=int, default=0,
        help="Feather mask edges by N pixels (Gaussian blur)",
    )

    args = parser.parse_args(argv)
    if args.check_runtime:
        try:
            validate_sam3_runtime()
        except ImportError as exc:
            console.print(f"[bold red]SAM3 runtime unavailable:[/bold red] {exc}")
            raise SystemExit(1) from exc
        console.print("[green]SAM3 runtime import is ready.[/green]")
        return
    if not args.prompt:
        parser.error("--prompt is required unless --check-runtime is used")

    # ── Resolve paths ──────────────────────────────────────────────────────────
    if args.scene and args.output:
        parser.error("--scene and --output cannot be used together")
    scene_paths = ScenePaths.from_root(args.scene or args.output or Path.cwd())
    project_root = scene_paths.root
    image_path = Path(args.path).expanduser().resolve() if args.path else scene_paths.images
    brush_mask_dir = scene_paths.brush_masks
    colmap_mask_dir = scene_paths.colmap_masks

    # ── Validate ───────────────────────────────────────────────────────────────
    if not image_path.is_dir():
        console.print(f"[bold red]Error:[/bold red] Image directory not found: {image_path}")
        console.print("Create an [bold]images/[/bold] folder, or pass [bold]--path DIR[/bold].")
        sys.exit(1)

    try:
        image_files = scene_paths.image_files() if image_path == scene_paths.images else list_images(image_path)
    except SceneError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)
    n_images = len(image_files)
    if n_images == 0:
        console.print(f"[bold red]Error:[/bold red] No images found in {image_path}")
        sys.exit(1)

    # Check for existing masks
    expected_brush = {f"{image.stem}.png" for image in image_files}
    expected_colmap = {f"{image.name}.png" for image in image_files}
    existing_brush = {path.name for path in brush_mask_dir.glob("*.png")} if brush_mask_dir.is_dir() else set()
    existing_colmap = {path.name for path in colmap_mask_dir.glob("*.png")} if colmap_mask_dir.is_dir() else set()
    if (existing_brush or existing_colmap) and not args.force:
        if existing_brush == expected_brush and existing_colmap == expected_colmap:
            console.print(f"[yellow]Masks already exist[/yellow] ({len(existing_brush)} brush, {len(existing_colmap)} colmap)")
            console.print("Pass [bold]--force[/bold] to re-run masking.")
            sys.exit(0)
        console.print("[bold red]Error:[/bold red] Existing masks are incomplete or do not match images/.")
        console.print("Pass [bold]--force[/bold] to replace them transactionally.")
        sys.exit(1)

    attempt_dir = scene_paths.staging_attempt("sammer", scene_paths.new_run_id())
    staged_brush_dir = attempt_dir / "masks"
    staged_colmap_dir = attempt_dir / "colmap-masks"

    # ── Device ─────────────────────────────────────────────────────────────────
    device = resolve_device(args.device)

    # ── UI setup ───────────────────────────────────────────────────────────────
    console.print(_header(image_path, n_images, args.prompt, device, args.score_threshold, args.checkpoint))
    console.print()

    console.print(f"[dim]Found {n_images} images, loading SAM3 model...[/dim]")

    # ── Load model ─────────────────────────────────────────────────────────────
    try:
        predictor = Sam3Predictor(device=args.device, checkpoint=args.checkpoint)
    except ImportError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        sys.exit(1)
    except Exception as exc:
        console.print(f"[bold red]Error loading SAM3 model:[/bold red] {exc}")
        sys.exit(1)
    console.print("[dim]Model loaded. Starting masking...[/dim]\n")

    # ── Masking loop ───────────────────────────────────────────────────────────
    progress = _make_progress()
    task_id = progress.add_task("Masking", total=n_images, status="", hit_rate="", max_conf="")

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
                    image, prompt=args.prompt, score_threshold=args.score_threshold,
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

                save_brush_mask(merged, img_path, staged_brush_dir, dilation_px=args.dilation, feather_px=args.feather)
                save_colmap_mask(merged, img_path, staged_colmap_dir)

                with lock:
                    hit_count += 1
                    results.append(ImageResult(image_name=img_path.name, success=True, confidence=max_conf))
                    status_text = f"✓ {img_path.name[:40]}"
                    conf_str = f"c={max_conf:.3f}"
            else:
                with lock:
                    results.append(ImageResult(image_name=img_path.name, success=False, confidence=None, error=error_msg))
                    status_text = f"✗ {img_path.name[:40]}"
                    conf_str = ""

            with lock:
                hit_rate_str = f"{hit_count}/{i+1}"
                progress.update(task_id, advance=1 if i > 0 else 0, status=status_text, hit_rate=hit_rate_str, max_conf=conf_str)
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
        console.print("[yellow]Masks were not published because one or more images failed.[/yellow]")
        console.print(f"[dim]Staged attempt retained at: {attempt_dir}[/dim]")
        _print_summary(results, n_images, elapsed, failures_path, staged_brush_dir, staged_colmap_dir)
        sys.exit(1)

    try:
        _publish_masks(staged_brush_dir, staged_colmap_dir, brush_mask_dir, colmap_mask_dir)
        record_stage(
            scene_paths,
            "sammer",
            {
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
        console.print(f"[bold red]Error publishing masks:[/bold red] {exc}")
        sys.exit(1)

    _print_summary(results, n_images, elapsed, None, brush_mask_dir, colmap_mask_dir)
    sys.exit(0)
