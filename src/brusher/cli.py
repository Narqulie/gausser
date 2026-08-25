"""brusher — beautiful CLI wrapper for Brush 3DGS training."""

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
)
from rich.table import Table

from gausser.manifest import fingerprint_inputs, record_stage
from gausser.scene import ScenePaths
from gausser.ui import console, IMAGE_EXTENSIONS, make_summary_table

from .runner import TrainProgress, run_brush


# ── Helpers ────────────────────────────────────────────────────────────────────


def _has_images(path: Path) -> bool:
    return any(f.suffix.lower() in IMAGE_EXTENSIONS for f in path.rglob("*") if f.is_file())


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold magenta]{task.description}"),
        BarColumn(bar_width=36),
        MofNCompleteColumn(),
        TextColumn("[yellow]{task.fields[rate]}"),
        TextColumn("[green]{task.fields[splats]}"),
        TextColumn("[cyan]{task.fields[psnr]}"),
        TimeElapsedColumn(),
        TextColumn("[dim]{task.fields[eta]}"),
        console=console,
        transient=False,
    )


def _header(scene_path: Path, total_iters: int, export_path: Path) -> Panel:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", no_wrap=True)
    grid.add_column()
    grid.add_row("Scene", str(scene_path))
    grid.add_row("Total steps", f"{total_iters:,}")
    grid.add_row("Export path", str(export_path))
    return Panel(
        grid, title="[bold]BRUSHER[/bold] · Brush 3DGS Training Pipeline", border_style="magenta",
    )


def _print_summary(elapsed: float, final: TrainProgress, export_path: Path, success: bool) -> None:
    console.print()
    table = make_summary_table("Training Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    status_str = "[green]✓ Done[/green]" if success else "[red]✗ Failed[/red]"
    table.add_row("Status", status_str)
    table.add_row("Total steps", f"{final.iter:,} / {final.total:,}")
    if final.splat_count:
        table.add_row("Final splat count", f"{final.splat_count:,}")
    if final.psnr is not None:
        table.add_row("Last PSNR", f"{final.psnr:.2f} dB")
    if final.ssim is not None:
        table.add_row("Last SSIM", f"{final.ssim:.4f}")
    table.add_row("Elapsed", f"{elapsed:.1f}s")

    console.print(table)
    if success:
        console.print(f"\n[bold green]Exports saved to:[/bold green] {export_path}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="brusher", description="Run Brush 3DGS training with a beautiful Rich UI.",
    )
    parser.add_argument("path", metavar="SCENE", nargs="?", default=None, help="Scene folder (default: cwd)")
    parser.add_argument("--total-train-iters", metavar="N", type=int, default=30000, help="Total training iterations")
    parser.add_argument("--export-every", metavar="N", type=int, default=5000, help="Export .ply every N steps")
    parser.add_argument("--export-path", metavar="DIR", default=None, help="Where to write .ply files")
    parser.add_argument("--eval-every", metavar="N", type=int, default=1000, help="Run eval every N steps")
    parser.add_argument("--refine-every", metavar="N", type=int, default=200, help="Densification frequency")
    parser.add_argument("--max-splats", metavar="N", type=int, default=None, help="Maximum number of Gaussians")
    parser.add_argument("--max-resolution", metavar="N", type=int, default=None, help="Max image resolution")
    parser.add_argument("--eval-split-every", metavar="N", type=int, default=None, help="Hold out every Nth image")
    parser.add_argument("--brush", metavar="BIN", default="brush", help="Path to brush binary")

    args = parser.parse_args(argv)

    if args.total_train_iters <= 0:
        parser.error("--total-train-iters must be positive")
    if not 0 < args.export_every < args.total_train_iters:
        parser.error("--export-every must be positive and less than --total-train-iters")
    if args.eval_every <= 0 or args.refine_every <= 0:
        parser.error("--eval-every and --refine-every must be positive")

    # ── Resolve paths ──────────────────────────────────────────────────────────
    scene_path = Path(args.path).expanduser().resolve() if args.path else Path.cwd()
    scene_paths = ScenePaths.from_root(scene_path)
    export_path = (
        Path(args.export_path).expanduser().resolve()
        if args.export_path
        else scene_paths.training_run("brush", scene_paths.new_run_id()) / "exports"
    )

    brush_bin = args.brush
    total_iters = args.total_train_iters

    # ── Validate ───────────────────────────────────────────────────────────────
    if not shutil.which(brush_bin):
        console.print(f"[bold red]Error:[/bold red] `{brush_bin}` not found on PATH.")
        console.print("Install Brush: https://github.com/ArthurBrussee/brush")
        sys.exit(1)

    if not scene_path.is_dir():
        console.print(f"[bold red]Error:[/bold red] Scene directory not found: {scene_path}")
        sys.exit(1)

    has_transforms = (scene_path / "transforms.json").exists()
    has_colmap = (scene_path / "sparse").is_dir() or (scene_path / "images").is_dir()
    if not has_transforms and not has_colmap and not _has_images(scene_path):
        console.print(f"[bold red]Error:[/bold red] No recognisable dataset found in {scene_path}")
        console.print("Expected: [bold]transforms.json[/bold] (NeRF-style) or [bold]images/[/bold] + [bold]sparse/[/bold] (COLMAP-style).")
        sys.exit(1)

    # ── Build brush command ────────────────────────────────────────────────────
    cmd = [
        brush_bin, str(scene_path),
        "--total-train-iters", str(total_iters),
        "--export-every", str(args.export_every),
        "--export-path", str(export_path),
        "--eval-every", str(args.eval_every),
        "--refine-every", str(args.refine_every),
    ]
    if args.max_splats is not None:
        cmd += ["--max-splats", str(args.max_splats)]
    if args.max_resolution is not None:
        cmd += ["--max-resolution", str(args.max_resolution)]
    if args.eval_split_every is not None:
        cmd += ["--eval-split-every", str(args.eval_split_every)]

    # ── UI setup ───────────────────────────────────────────────────────────────
    console.print(_header(scene_path, total_iters, export_path))
    console.print()

    progress = _make_progress()
    task_id = progress.add_task("Initializing", total=total_iters, splats="", psnr="", rate="", eta="")

    overall_start = time.monotonic()
    final_state = TrainProgress(total=total_iters)
    lock = Lock()
    _prev: dict = {"iter": 0, "t": None}

    def on_progress(p: TrainProgress) -> None:
        with lock:
            final_state.iter = p.iter
            final_state.total = p.total
            final_state.splat_count = p.splat_count
            if p.psnr is not None:
                final_state.psnr = p.psnr
            if p.ssim is not None:
                final_state.ssim = p.ssim
            final_state.phase = p.phase

            now = time.monotonic()
            rate_str = ""
            eta_str = "ETA -:--:--"
            if _prev["t"] is not None and p.iter > _prev["iter"]:
                delta_t = now - _prev["t"]
                delta_i = p.iter - _prev["iter"]
                if delta_t > 0.01:
                    its = delta_i / delta_t
                    rate_str = f"{its:.1f} it/s"
                    remaining = p.total - p.iter
                    if its > 0 and remaining > 0:
                        secs = int(remaining / its)
                        h, rem = divmod(secs, 3600)
                        m, s = divmod(rem, 60)
                        eta_str = f"ETA {h}:{m:02d}:{s:02d}"
                    elif remaining == 0:
                        eta_str = ""
            if p.iter > 0:
                _prev["iter"] = p.iter
                _prev["t"] = now

            splats_str = f"{p.splat_count:,} splats" if p.splat_count else ""
            psnr_str = f"PSNR {final_state.psnr:.2f}dB" if final_state.psnr is not None else ""

            desc = {"starting": "Initializing", "loading": "Loading dataset", "training": "Training", "done": "[green]Done[/green]"}.get(p.phase, "Training")

            progress.update(task_id, description=desc, completed=p.iter, rate=rate_str, eta=eta_str, splats=splats_str, psnr=psnr_str)

    with progress:
        progress.update(task_id, completed=0)
        result = run_brush(cmd, cwd=scene_path, on_progress=on_progress, total_iters=total_iters)
        if result.succeeded:
            progress.update(task_id, completed=total_iters)

    elapsed = time.monotonic() - overall_start
    success = result is not None and result.succeeded
    exports = sorted(export_path.rglob("*.ply")) if success and export_path.is_dir() else []
    if success and not exports:
        success = False
        console.print("\n[bold red]Brush completed without creating any .ply export.[/bold red]")

    if not success:
        rc = result.returncode if result is not None else 1
        console.print(f"\n[bold red]Brush exited with code {rc}.[/bold red]")
        if result and result.output_tail:
            console.print("[bold]Last Brush output:[/bold]")
            console.print("\n".join(result.output_tail[-20:]))

    if success:
        record_stage(
            scene_paths,
            "brush",
            {
                "inputs": fingerprint_inputs(scene_paths.image_files()),
                "parameters": {
                    "total_train_iters": total_iters,
                    "export_every": args.export_every,
                    "eval_every": args.eval_every,
                    "refine_every": args.refine_every,
                    "max_splats": args.max_splats,
                    "max_resolution": args.max_resolution,
                    "eval_split_every": args.eval_split_every,
                },
                "outputs": {
                    "run": str(scene_paths.relative_path(export_path.parent)),
                    "exports": [str(scene_paths.relative_path(path)) for path in exports],
                },
            },
        )
    _print_summary(elapsed, final_state, export_path, success)
    sys.exit(0 if success else 1)
