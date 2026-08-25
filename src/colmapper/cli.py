"""colmapper — simple COLMAP SfM wrapper with a Rich CLI."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Optional

from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from gausser.colmap_model import read_model_stats
from gausser.scene import SceneError, ScenePaths
from gausser.ui import console, count_images, make_summary_table

from .pipeline import ColmapError, feature_extraction, feature_matching, sparse_reconstruction
from .runner import StepProgress

# ── Step definitions ───────────────────────────────────────────────────────────

STEPS = [
    ("Feature Extraction", "Detecting keypoints in every image"),
    ("Feature Matching", "Finding correspondences between image pairs"),
    ("Sparse Reconstruction", "Building 3-D point cloud from matched features"),
]


# ── Helpers ────────────────────────────────────────────────────────────────────


def resolve_gpu(*, no_gpu: bool) -> bool:
    """COLMAP's Homebrew build on macOS has no CUDA implementation."""
    return not no_gpu and sys.platform != "darwin"


def _prepare_artifacts(database_path: Path, output_path: Path, policy: str) -> None:
    existing = database_path.exists() or output_path.exists()
    if not existing or policy == "resume":
        return
    if policy == "replace":
        database_path.unlink(missing_ok=True)
        shutil.rmtree(output_path, ignore_errors=True)
        return
    raise ColmapError(
        "COLMAP artifacts already exist. Choose --existing resume to keep them "
        "or --existing replace to delete and rebuild them."
    )


def _read_model_stats(model_dir: Path) -> tuple[int, int]:
    """Read registered images and 3D points from a COLMAP sparse model dir.

    Returns (num_images, num_points), or (0, 0) if the model can't be read.
    """
    stats = read_model_stats(model_dir)
    return (stats.images, stats.points) if stats is not None else (0, 0)


def _pick_best_model(output_path: Path) -> Path | None:
    """If COLMAP produced multiple sub-models, pick the one with the most
    registered images and relocate it to <output_path>/0/ for compatibility."""
    models = sorted(output_path.glob("[0-9]*"))
    if not models:
        return None

    if len(models) == 1:
        return models[0]

    # Multiple sub-models — pick by largest registered image count
    best = max(models, key=lambda m: _read_model_stats(m)[0])
    n_reg = _read_model_stats(best)[0]
    console.print(
        f"[yellow]⚠  COLMAP produced {len(models)} sub-models; "
        f"using model with {n_reg} images.[/yellow]"
    )
    zero_path = output_path / "0"
    if best != zero_path:
        shutil.rmtree(zero_path, ignore_errors=True)
        best.rename(zero_path)
        return zero_path
    return best


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=36),
        MofNCompleteColumn(),
        TextColumn("[green]{task.fields[rate]}"),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def _header(project_root: Path, image_path: Path, n_images: int, matcher: str, masks: bool) -> Panel:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", no_wrap=True)
    grid.add_column()
    grid.add_row("Project root", str(project_root))
    grid.add_row("Images", f"{image_path}  ({n_images} files)")
    grid.add_row("Matcher", matcher)
    grid.add_row("Alpha masks", "[green]enabled[/green]" if masks else "[dim]none[/dim]")
    return Panel(grid, title="[bold]COLMAPPER[/bold] · Automatic SfM Pipeline", border_style="blue")


_TAIL_LINES = 30


def _run_step(
    progress: Progress, task_id: TaskID, fn, *args, **kwargs,
) -> list[str]:
    lock = Lock()
    last_update: dict = {"t": time.monotonic(), "n": 0}
    output_lines: list[str] = []
    fixed_total: int = progress.tasks[task_id].total or 1

    def on_progress(p: StepProgress) -> None:
        with lock:
            now = time.monotonic()
            elapsed = now - last_update["t"]
            delta = p.current - last_update["n"]
            if delta <= 0:
                return
            rate = f"{delta / elapsed:.1f}/s" if elapsed > 0.1 else ""
            last_update["t"] = now
            last_update["n"] = p.current
            scaled = min(int(p.current * fixed_total / p.total) if p.total else p.current, fixed_total)
            progress.update(task_id, completed=scaled, rate=rate)

    def on_line(line: str) -> None:
        output_lines.append(line)
        if len(output_lines) > _TAIL_LINES:
            output_lines.pop(0)

    fn(*args, on_progress=on_progress, on_line=on_line, **kwargs)
    progress.update(task_id, completed=fixed_total, rate="")
    return output_lines


# ── Main ───────────────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="colmapper",
        description="Run COLMAP automatic SfM reconstruction.",
    )
    parser.add_argument(
        "--scene", metavar="DIR", default=None,
        help="Scene root that owns the COLMAP database and sparse model (default: cwd)",
    )
    parser.add_argument(
        "--path", metavar="DIR", default=None,
        help="Path to the image folder (default: <scene>/images/)",
    )
    parser.add_argument(
        "--matcher", choices=["exhaustive", "sequential", "vocab_tree"], default="exhaustive",
        help="Feature matching strategy (default: exhaustive)",
    )
    parser.add_argument(
        "--no-gpu", action="store_true", help="Disable GPU acceleration",
    )
    parser.add_argument(
        "--existing", choices=["error", "resume", "replace"], default="error",
        help="Policy for existing database/sparse artifacts (default: error)",
    )
    parser.add_argument(
        "--colmap", metavar="BIN", default="colmap", help="Path to the colmap binary",
    )
    parser.add_argument(
        "--output", metavar="DIR", default=None, help="Output directory for sparse model",
    )
    parser.add_argument(
        "--mask-path", metavar="DIR", default=None,
        help="Path to COLMAP feature masks (auto-detected: <project>/colmap/masks/)",
    )
    parser.add_argument(
        "--vocab-tree", metavar="FILE", default=None,
        help="Vocabulary tree required by --matcher vocab_tree",
    )

    args = parser.parse_args(argv)

    # ── Resolve paths ──────────────────────────────────────────────────────────
    scene_paths = ScenePaths.from_root(args.scene or Path.cwd())
    project_root = scene_paths.root
    image_path = Path(args.path).expanduser().resolve() if args.path else scene_paths.images
    output_path = Path(args.output).resolve() if args.output else scene_paths.sparse
    database_path = scene_paths.database

    use_gpu = resolve_gpu(no_gpu=args.no_gpu)
    matcher = args.matcher
    colmap_bin = args.colmap
    vocab_tree_path = Path(args.vocab_tree).expanduser().resolve() if args.vocab_tree else None

    # Auto-detect colmap masks
    auto_mask_path = scene_paths.colmap_masks
    mask_path: Path | None = (
        Path(args.mask_path).resolve() if args.mask_path
        else auto_mask_path if auto_mask_path.is_dir()
        else None
    )
    has_masks = mask_path is not None and mask_path.is_dir()

    # ── Validate ───────────────────────────────────────────────────────────────
    if not shutil.which(colmap_bin):
        console.print(f"[bold red]Error:[/bold red] `{colmap_bin}` not found on PATH.")
        console.print("Install COLMAP: https://colmap.github.io/install.html")
        sys.exit(1)

    if not image_path.is_dir():
        console.print(f"[bold red]Error:[/bold red] Image directory not found: {image_path}")
        console.print("Create an [bold]images/[/bold] folder, or pass [bold]--path DIR[/bold].")
        sys.exit(1)

    try:
        n_images = len(scene_paths.image_files()) if image_path == scene_paths.images else count_images(image_path)
    except SceneError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)
    if n_images == 0:
        console.print(f"[bold red]Error:[/bold red] No images found in {image_path}")
        sys.exit(1)
    if matcher == "vocab_tree" and (vocab_tree_path is None or not vocab_tree_path.is_file()):
        console.print("[bold red]Error:[/bold red] --matcher vocab_tree requires --vocab-tree FILE.")
        sys.exit(1)

    try:
        _prepare_artifacts(database_path, output_path, args.existing)
    except ColmapError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    if not use_gpu and not args.no_gpu and sys.platform == "darwin":
        console.print("[dim]COLMAP GPU acceleration disabled: macOS builds do not provide CUDA.[/dim]")

    # ── Matcher suggestion ─────────────────────────────────────────────────────
    if matcher == "exhaustive" and n_images > 200:
        pairs = n_images * (n_images - 1) // 2
        console.print()
        console.print(Panel(
            f"[yellow]⚠  {n_images} images with [bold]exhaustive[/bold] matcher will match"
            f" [bold]{pairs:,}[/bold] image pairs.[/yellow]\n\n"
            f"  [bold]Suggestions:[/bold]\n"
            f"  [green]  --matcher sequential[/green]    if these are video frames (fastest)\n"
            f"  [green]  --matcher vocab_tree[/green]    for unordered photos at scale\n\n"
            f"  [dim]Pass --matcher to override, or wait — it will eventually finish.[/dim]",
            title="[bold yellow]Large dataset detected[/bold yellow]",
            border_style="yellow",
        ))
        console.print()

    # ── UI setup ───────────────────────────────────────────────────────────────
    console.print(_header(project_root, image_path, n_images, matcher, has_masks))
    console.print()

    step_results: list[tuple[str, float, str]] = []
    overall_start = time.monotonic()

    progress = _make_progress()

    step_configs = [
        {"name": STEPS[0][0], "description": STEPS[0][0], "total": n_images},
        {"name": STEPS[1][0], "description": STEPS[1][0],
         "total": n_images * (n_images - 1) // 2 if matcher == "exhaustive" else n_images},
        {"name": STEPS[2][0], "description": STEPS[2][0], "total": n_images},
    ]

    with progress:
        task_ids = [
            progress.add_task(cfg["description"], total=cfg["total"], rate="", visible=False)
            for cfg in step_configs
        ]

        for i, (cfg, task_id) in enumerate(zip(step_configs, task_ids)):
            progress.update(task_id, visible=True)
            t0 = time.monotonic()
            status = "✓"
            output_lines: list[str] = []
            try:
                if i == 0:
                    output_lines = _run_step(
                        progress, task_id, feature_extraction,
                        colmap_bin, database_path, image_path,
                        use_gpu=use_gpu, mask_path=mask_path, total_images=n_images,
                    )
                elif i == 1:
                    output_lines = _run_step(
                        progress, task_id, feature_matching,
                        colmap_bin, database_path, matcher=matcher, use_gpu=use_gpu, vocab_tree_path=vocab_tree_path,
                    )
                elif i == 2:
                    output_lines = _run_step(
                        progress, task_id, sparse_reconstruction,
                        colmap_bin, database_path, image_path, output_path, total_images=n_images,
                    )
            except ColmapError as exc:
                progress.stop()
                step_results.append((cfg["name"], time.monotonic() - t0, "✗"))
                console.print(f"\n[bold red]Pipeline failed:[/bold red] {exc}\n")
                if output_lines:
                    console.print("[bold yellow]COLMAP output:[/bold yellow]")
                    for line in output_lines:
                        console.print(f"  [dim]{line}[/dim]")
                    console.print()
                _print_summary(step_results, time.monotonic() - overall_start)
                sys.exit(1)

            elapsed = time.monotonic() - t0
            progress.update(task_id, description=f"[green]{cfg['name']}[/green]")
            step_results.append((cfg["name"], elapsed, status))

    # ── Select and validate a usable model before declaring success ─────────
    model_path = _pick_best_model(output_path)
    stats = read_model_stats(model_path) if model_path is not None else None
    if stats is None or stats.images == 0:
        step_results[-1] = (STEPS[2][0], step_results[-1][1], "✗")
        console.print("\n[bold red]COLMAP completed without a valid sparse model.[/bold red]")
        _print_summary(step_results, time.monotonic() - overall_start)
        sys.exit(1)
    _print_summary(step_results, time.monotonic() - overall_start, n_images, model_path)


def _print_summary(
    results: list[tuple[str, float, str]],
    total: float,
    n_images: int = 0,
    model_path: Path | None = None,
) -> None:
    console.print()
    table = make_summary_table("Pipeline Summary")
    table.add_column("Step", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Time", justify="right")

    for name, elapsed, status in results:
        style = "green" if status == "✓" else "red"
        table.add_row(name, f"[{style}]{status}[/{style}]", f"{elapsed:.1f}s")

    console.print(table)
    console.print(f"[dim]Total elapsed: {total:.1f}s[/dim]")

    if model_path and model_path.is_dir():
        n_imgs, n_pts = _read_model_stats(model_path)
        console.print()
        recon = Table.grid(padding=(0, 2))
        recon.add_column(style="bold cyan")
        recon.add_column()
        recon.add_row("Output", str(model_path))
        recon.add_row("Registered images", f"{n_imgs} / {n_images}")
        recon.add_row("3D points", f"{n_pts:,}")
        console.print(Panel(recon, title="[bold]Reconstruction[/bold]", border_style="green"))
