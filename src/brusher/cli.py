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
from rich.progress import MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.text import Text

from gausser.interactive import (
    BARE_COMMAND_EPILOG,
    InteractiveUsageError,
    prepare_stage_arguments,
)
from gausser.manifest import fingerprint_inputs
from gausser.process import resolve_command
from gausser.publish import publish_stage
from gausser.scene import SceneError, ScenePaths
from gausser.scene_lock import SceneLockError, acquire_scene_lock
from gausser.ui import (
    NEON_CYAN,
    NEON_GREEN,
    NEON_MAGENTA,
    NEON_YELLOW,
    console,
    make_summary_table,
    neon_bar,
    neon_spinner,
    stage_header,
    is_image_file,
    status_text as neon_status_text,
)

from .runner import TrainProgress, run_brush


# ── Helpers ────────────────────────────────────────────────────────────────────


def _has_images(path: Path) -> bool:
    return any(is_image_file(file) for file in path.rglob("*") if file.is_file())


def _make_progress() -> Progress:
    return Progress(
        neon_spinner(style=NEON_MAGENTA),
        TextColumn("{task.description}", style=f"bold {NEON_MAGENTA}", markup=False),
        neon_bar(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[rate]}", style=NEON_YELLOW, markup=False),
        TextColumn("{task.fields[splats]}", style=NEON_GREEN, markup=False),
        TextColumn("{task.fields[psnr]}", style=NEON_CYAN, markup=False),
        TimeElapsedColumn(),
        TextColumn("{task.fields[eta]}", style="dim", markup=False),
        console=console,
        transient=False,
    )


def _header(scene_path: Path, total_iters: int, export_path: Path) -> Panel:
    return stage_header(
        "BRUSHER",
        "Gaussian training signal",
        (
            ("Scene", str(scene_path)),
            ("Iterations", f"{total_iters:,}"),
            ("Export", str(export_path)),
        ),
        accent=NEON_MAGENTA,
    )


def _print_summary(
    elapsed: float, final: TrainProgress, export_path: Path, success: bool
) -> None:
    console.print()
    table = make_summary_table("Training Summary", accent=NEON_MAGENTA)

    table.add_row(
        "Status",
        neon_status_text("Done", "success")
        if success
        else neon_status_text("Failed", "error"),
    )
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
        console.print()
        console.print(neon_status_text(f"Exports saved to: {export_path}", "success"))


# ── Main ───────────────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="brusher",
        description="Run Brush 3DGS training for one Gausser scene.",
        epilog=BARE_COMMAND_EPILOG,
    )
    parser.add_argument(
        "path",
        metavar="SCENE",
        nargs="?",
        default=None,
        help="Scene folder (default: cwd)",
    )
    parser.add_argument("--total-train-iters", metavar="N", type=int, default=30000)
    parser.add_argument("--export-every", metavar="N", type=int, default=5000)
    parser.add_argument(
        "--export-path",
        metavar="DIR",
        default=None,
        help="Fresh scene-owned runs/brush/<id>/exports path",
    )
    parser.add_argument("--eval-every", metavar="N", type=int, default=1000)
    parser.add_argument("--refine-every", metavar="N", type=int, default=200)
    parser.add_argument("--max-splats", metavar="N", type=int, default=None)
    parser.add_argument("--max-resolution", metavar="N", type=int, default=None)
    parser.add_argument("--eval-split-every", metavar="N", type=int, default=None)
    parser.add_argument("--brush", metavar="BIN", default="brush")
    try:
        prepared_argv = prepare_stage_arguments("brush", argv, program="brusher")
    except InteractiveUsageError as exc:
        parser.error(str(exc))
    if prepared_argv is None:
        return
    args = parser.parse_args(prepared_argv)

    if args.total_train_iters <= 0:
        parser.error("--total-train-iters must be positive")
    if not 0 < args.export_every < args.total_train_iters:
        parser.error(
            "--export-every must be positive and less than --total-train-iters"
        )
    if args.eval_every <= 0 or args.refine_every <= 0:
        parser.error("--eval-every and --refine-every must be positive")

    scene_path = (
        Path(args.path).expanduser().resolve() if args.path else Path.cwd().resolve()
    )
    if not scene_path.is_dir():
        parser.error(f"scene directory does not exist: {scene_path}")
    paths = ScenePaths.from_root(scene_path)
    brush_bin = resolve_command(args.brush)
    if brush_bin is None:
        parser.error(f"{args.brush!r} is not on PATH; install Brush first")

    run_id = paths.new_run_id()
    destination_run = paths.training_run("brush", run_id)
    if args.export_path:
        requested_export = Path(args.export_path).expanduser().resolve()
        try:
            relative = paths.relative_path(requested_export)
        except SceneError as exc:
            parser.error(str(exc))
        if (
            len(relative.parts) != 4
            or relative.parts[:2] != ("runs", "brush")
            or relative.parts[3] != "exports"
        ):
            parser.error(
                "--export-path must be a fresh scene-owned runs/brush/<run-id>/exports directory"
            )
        run_id = relative.parts[2]
        destination_run = paths.training_run("brush", run_id)
    if destination_run.exists():
        parser.error(f"Brush run destination already exists: {destination_run}")

    try:
        with acquire_scene_lock(paths.root):
            _run_brusher(args, paths, brush_bin, run_id, destination_run)
    except SceneLockError as exc:
        parser.error(str(exc))


def _run_brusher(
    args: argparse.Namespace,
    paths: ScenePaths,
    brush_bin: str,
    run_id: str,
    destination_run: Path,
) -> None:
    scene_path = paths.root
    has_transforms = (scene_path / "transforms.json").exists()
    try:
        has_colmap = paths.sparse.is_dir() and bool(paths.image_files())
    except SceneError as exc:
        console.print(neon_status_text(str(exc), "error"))
        raise SystemExit(1) from exc
    if not has_transforms and not has_colmap and not _has_images(scene_path):
        console.print(
            neon_status_text(f"No recognisable dataset found in {scene_path}", "error")
        )
        console.print("Expected transforms.json or images/ + sparse/.")
        sys.exit(1)

    attempt = paths.staging_attempt("brush", run_id)
    staged_run = attempt / "run"
    staged_export = staged_run / "exports"
    total_iters = args.total_train_iters
    cmd = [
        brush_bin,
        str(scene_path),
        "--total-train-iters",
        str(total_iters),
        "--export-every",
        str(args.export_every),
        "--export-path",
        str(staged_export),
        "--eval-every",
        str(args.eval_every),
        "--refine-every",
        str(args.refine_every),
    ]
    if args.max_splats is not None:
        cmd += ["--max-splats", str(args.max_splats)]
    if args.max_resolution is not None:
        cmd += ["--max-resolution", str(args.max_resolution)]
    if args.eval_split_every is not None:
        cmd += ["--eval-split-every", str(args.eval_split_every)]

    final_export = destination_run / "exports"
    console.print(_header(scene_path, total_iters, final_export))
    console.print()
    progress = _make_progress()
    task_id = progress.add_task(
        "Initializing", total=total_iters, splats="", psnr="", rate="", eta=""
    )
    overall_start = time.monotonic()
    final_state = TrainProgress(total=total_iters)
    lock = Lock()
    previous: dict[str, float | int | None] = {"iter": 0, "time": None}

    def on_progress(progress_state: TrainProgress) -> None:
        with lock:
            final_state.iter = progress_state.iter
            final_state.total = progress_state.total
            final_state.splat_count = progress_state.splat_count
            if progress_state.psnr is not None:
                final_state.psnr = progress_state.psnr
            if progress_state.ssim is not None:
                final_state.ssim = progress_state.ssim
            final_state.phase = progress_state.phase

            now = time.monotonic()
            rate = ""
            eta = "ETA -:--:--"
            previous_time = previous["time"]
            previous_iter = int(previous["iter"] or 0)
            if isinstance(previous_time, float) and progress_state.iter > previous_iter:
                delta_time = now - previous_time
                delta_iterations = progress_state.iter - previous_iter
                if delta_time > 0.01:
                    iterations_per_second = delta_iterations / delta_time
                    rate = f"{iterations_per_second:.1f} it/s"
                    remaining = progress_state.total - progress_state.iter
                    if iterations_per_second > 0 and remaining > 0:
                        seconds = int(remaining / iterations_per_second)
                        hours, remainder = divmod(seconds, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        eta = f"ETA {hours}:{minutes:02d}:{seconds:02d}"
                    elif remaining == 0:
                        eta = ""
            if progress_state.iter > 0:
                previous["iter"] = progress_state.iter
                previous["time"] = now

            description = {
                "starting": "Initializing",
                "loading": "Loading dataset",
                "training": "Training",
                "done": "Done",
            }.get(progress_state.phase, "Training")
            progress.update(
                task_id,
                description=description,
                completed=progress_state.iter,
                rate=rate,
                eta=eta,
                splats=f"{progress_state.splat_count:,} splats"
                if progress_state.splat_count
                else "",
                psnr=f"PSNR {final_state.psnr:.2f}dB"
                if final_state.psnr is not None
                else "",
            )

    with progress:
        result = run_brush(
            cmd, cwd=scene_path, on_progress=on_progress, total_iters=total_iters
        )
        if result.succeeded:
            progress.update(task_id, completed=total_iters)

    elapsed = time.monotonic() - overall_start
    staged_exports = (
        sorted(staged_export.rglob("*.ply"))
        if result.succeeded and staged_export.is_dir()
        else []
    )
    success = result.succeeded and bool(staged_exports)
    if result.succeeded and not staged_exports:
        console.print()
        console.print(
            neon_status_text("Brush completed without creating a .ply export", "error")
        )
    if not success:
        console.print()
        console.print(
            neon_status_text(f"Brush exited with code {result.returncode}", "error")
        )
        if result.output_tail:
            console.print(neon_status_text("Last Brush output follows", "warning"))
            console.print(Text("\n".join(result.output_tail[-20:])))
        console.print(
            neon_status_text(f"Incomplete run retained at: {attempt}", "muted")
        )
        _print_summary(elapsed, final_state, final_export, False)
        sys.exit(1)

    try:
        image_files = paths.image_files() if paths.images.is_dir() else []
    except SceneError:
        image_files = []
    final_exports = [
        destination_run / export.relative_to(staged_run) for export in staged_exports
    ]
    try:
        publish_stage(
            paths,
            "brush",
            generation=None,
            replacements=((staged_run, destination_run),),
            data={
                "inputs": fingerprint_inputs(image_files),
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
                    "run": str(paths.relative_path(destination_run)),
                    "exports": [
                        str(paths.relative_path(export)) for export in final_exports
                    ],
                },
            },
        )
    except Exception as exc:
        console.print(
            neon_status_text(f"Brush result could not be published: {exc}", "error")
        )
        console.print(
            neon_status_text(f"Completed staged run retained at: {attempt}", "muted")
        )
        sys.exit(1)

    shutil.rmtree(attempt, ignore_errors=True)
    _print_summary(elapsed, final_state, final_export, True)
    sys.exit(0)
