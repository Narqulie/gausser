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
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from gausser.colmap_model import read_model_stats
from gausser.interactive import (
    BARE_COMMAND_EPILOG,
    InteractiveUsageError,
    prepare_stage_arguments,
)
from gausser.manifest import fingerprint_inputs
from gausser.process import resolve_command, run_command
from gausser.publish import publish_stage
from gausser.scene import SceneError, ScenePaths
from gausser.scene_lock import SceneLockError, acquire_scene_lock
from gausser.ui import (
    NEON_CYAN,
    NEON_GREEN,
    NEON_YELLOW,
    completion_panel,
    console,
    make_summary_table,
    neon_bar,
    neon_spinner,
    stage_header,
    status_text as neon_status_text,
)

from .pipeline import (
    ColmapError,
    feature_extraction,
    feature_matching,
    sparse_reconstruction,
)
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


def _stage_artifacts(
    paths: ScenePaths, attempt: Path, policy: str
) -> tuple[Path, Path]:
    """Prepare isolated database/model paths without mutating canonical state."""
    existing = paths.database.exists() or paths.sparse.exists()
    if existing and policy == "error":
        raise ColmapError(
            "COLMAP artifacts already exist. Choose --existing resume to copy and continue them "
            "or --existing replace to stage a clean rebuild."
        )

    database = attempt / "database.db"
    sparse = attempt / "sparse"
    attempt.mkdir(parents=True, exist_ok=False)
    if policy == "resume":
        if paths.database.is_file():
            shutil.copy2(paths.database, database)
        if paths.sparse.is_dir():
            shutil.copytree(paths.sparse, sparse)
    return database, sparse


def _colmap_version(colmap_bin: str, project_root: Path) -> str | None:
    result = run_command([colmap_bin, "-h"], cwd=project_root)
    for line in result.output_tail:
        if "COLMAP" in line:
            return line.strip()
    return None


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

    best = (
        models[0]
        if len(models) == 1
        else max(models, key=lambda m: _read_model_stats(m)[0])
    )
    if len(models) > 1:
        n_reg = _read_model_stats(best)[0]
        console.print(
            neon_status_text(
                f"COLMAP produced {len(models)} sub-models; using the model with {n_reg} images",
                "warning",
            )
        )
    zero_path = output_path / "0"
    if best != zero_path:
        shutil.rmtree(zero_path, ignore_errors=True)
        best.rename(zero_path)
        return zero_path
    return best


def _make_progress() -> Progress:
    return Progress(
        neon_spinner(style=NEON_CYAN),
        TextColumn("{task.description}", style=f"bold {NEON_CYAN}", markup=False),
        neon_bar(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[rate]}", style=NEON_GREEN, markup=False),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def _header(
    project_root: Path, image_path: Path, n_images: int, matcher: str, masks: bool
) -> Panel:
    return stage_header(
        "COLMAPPER",
        "Sparse reconstruction signal",
        (
            ("Scene", str(project_root)),
            ("Images", f"{image_path}  ({n_images} files)"),
            ("Matcher", matcher),
            ("Alpha masks", "enabled" if masks else "none"),
        ),
        accent=NEON_CYAN,
    )


_TAIL_LINES = 30


def _run_step(
    progress: Progress,
    task_id: TaskID,
    fn,
    *args,
    **kwargs,
) -> list[str]:
    lock = Lock()
    last_update: dict = {"t": time.monotonic(), "n": 0}
    output_lines: list[str] = []
    fixed_total = int(progress.tasks[task_id].total or 1)

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
            scaled = min(
                int(p.current * fixed_total / p.total) if p.total else p.current,
                fixed_total,
            )
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
        description="Run a transactional COLMAP SfM reconstruction for one Gausser scene.",
        epilog=BARE_COMMAND_EPILOG,
    )
    parser.add_argument(
        "--scene", metavar="DIR", default=None, help="Owning scene root (default: cwd)"
    )
    parser.add_argument(
        "--path",
        metavar="DIR",
        default=None,
        help="Image directory; must be <scene>/images",
    )
    parser.add_argument(
        "--matcher",
        choices=["exhaustive", "sequential", "vocab_tree"],
        default="exhaustive",
        help="Feature matching strategy (default: exhaustive)",
    )
    parser.add_argument(
        "--no-gpu", action="store_true", help="Disable GPU acceleration"
    )
    parser.add_argument(
        "--existing",
        choices=["error", "resume", "replace"],
        default="error",
        help="Policy for existing database/sparse artifacts (default: error)",
    )
    parser.add_argument(
        "--colmap", metavar="BIN", default="colmap", help="Path to the colmap binary"
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        default=None,
        help="Deprecated canonical sparse path override",
    )
    parser.add_argument(
        "--mask-path",
        metavar="DIR",
        default=None,
        help="COLMAP feature masks (default: <scene>/colmap/masks)",
    )
    parser.add_argument("--vocab-tree", metavar="FILE", default=None)
    try:
        prepared_argv = prepare_stage_arguments(
            "colmap", argv, program="colmapper"
        )
    except InteractiveUsageError as exc:
        parser.error(str(exc))
    if prepared_argv is None:
        return
    args = parser.parse_args(prepared_argv)

    inferred_root = (
        Path(args.path).expanduser().resolve().parent
        if args.path and not args.scene
        else Path.cwd()
    )
    paths = ScenePaths.from_root(args.scene or inferred_root)
    if not paths.root.is_dir():
        parser.error(f"scene directory does not exist: {paths.root}")
    image_path = Path(args.path).expanduser().resolve() if args.path else paths.images
    output_path = (
        Path(args.output).expanduser().resolve() if args.output else paths.sparse
    )
    if image_path != paths.images:
        parser.error("--path must resolve to the owning scene's images/ directory")
    if output_path != paths.sparse:
        parser.error("--output must resolve to the owning scene's sparse/ directory")

    colmap_bin = resolve_command(args.colmap)
    if colmap_bin is None:
        parser.error(f"{args.colmap!r} is not on PATH; install COLMAP first")

    vocab_tree_path = (
        Path(args.vocab_tree).expanduser().resolve() if args.vocab_tree else None
    )
    if args.matcher == "vocab_tree" and (
        vocab_tree_path is None or not vocab_tree_path.is_file()
    ):
        parser.error("--matcher vocab_tree requires --vocab-tree FILE")

    mask_path: Path | None = (
        Path(args.mask_path).expanduser().resolve()
        if args.mask_path
        else paths.colmap_masks
    )
    if mask_path is not None and not mask_path.is_dir():
        mask_path = None

    try:
        with acquire_scene_lock(paths.root):
            _run_colmapper(
                args, paths, image_path, mask_path, vocab_tree_path, colmap_bin
            )
    except SceneLockError as exc:
        parser.error(str(exc))


def _run_colmapper(
    args: argparse.Namespace,
    paths: ScenePaths,
    image_path: Path,
    mask_path: Path | None,
    vocab_tree_path: Path | None,
    colmap_bin: str,
) -> None:
    try:
        image_files = paths.image_files()
    except SceneError as exc:
        console.print(neon_status_text(str(exc), "error"))
        sys.exit(1)
    n_images = len(image_files)
    if n_images == 0:
        console.print(neon_status_text(f"No images found in {image_path}", "error"))
        sys.exit(1)

    attempt = paths.staging_attempt("colmap", paths.new_run_id())
    try:
        database_path, output_path = _stage_artifacts(paths, attempt, args.existing)
    except ColmapError as exc:
        console.print(neon_status_text(str(exc), "error"))
        sys.exit(1)

    use_gpu = resolve_gpu(no_gpu=args.no_gpu)
    matcher = args.matcher
    if not use_gpu and not args.no_gpu and sys.platform == "darwin":
        console.print(
            neon_status_text(
                "COLMAP GPU acceleration disabled: macOS builds do not provide CUDA",
                "muted",
            )
        )

    if matcher == "exhaustive" and n_images > 200:
        pairs = n_images * (n_images - 1) // 2
        warning = Text()
        warning.append(
            f"◇ {n_images} images with exhaustive matcher will match {pairs:,} image pairs.\n\n",
            style=NEON_YELLOW,
        )
        warning.append("  --matcher sequential", style=f"bold {NEON_GREEN}")
        warning.append("  for video frames\n")
        warning.append("  --matcher vocab_tree", style=f"bold {NEON_GREEN}")
        warning.append("  for unordered photos at scale")
        console.print(
            Panel(
                warning,
                title=Text("LARGE DATASET DETECTED", style=f"bold {NEON_YELLOW}"),
                border_style=NEON_YELLOW,
            )
        )

    console.print(
        _header(paths.root, image_path, n_images, matcher, mask_path is not None)
    )
    console.print()
    step_results: list[tuple[str, float, str]] = []
    overall_start = time.monotonic()
    progress = _make_progress()
    step_configs: list[tuple[str, int]] = [
        (STEPS[0][0], n_images),
        (
            STEPS[1][0],
            n_images * (n_images - 1) // 2 if matcher == "exhaustive" else n_images,
        ),
        (STEPS[2][0], n_images),
    ]

    with progress:
        task_ids = [
            progress.add_task(name, total=total, rate="", visible=False)
            for name, total in step_configs
        ]
        for index, ((step_name, _total), task_id) in enumerate(
            zip(step_configs, task_ids)
        ):
            progress.update(task_id, visible=True)
            started = time.monotonic()
            output_lines: list[str] = []
            try:
                if index == 0:
                    output_lines = _run_step(
                        progress,
                        task_id,
                        feature_extraction,
                        colmap_bin,
                        database_path,
                        image_path,
                        use_gpu=use_gpu,
                        mask_path=mask_path,
                        total_images=n_images,
                    )
                elif index == 1:
                    output_lines = _run_step(
                        progress,
                        task_id,
                        feature_matching,
                        colmap_bin,
                        database_path,
                        matcher=matcher,
                        use_gpu=use_gpu,
                        vocab_tree_path=vocab_tree_path,
                    )
                else:
                    output_lines = _run_step(
                        progress,
                        task_id,
                        sparse_reconstruction,
                        colmap_bin,
                        database_path,
                        image_path,
                        output_path,
                        total_images=n_images,
                    )
            except ColmapError as exc:
                progress.stop()
                step_results.append((step_name, time.monotonic() - started, "✗"))
                console.print()
                console.print(neon_status_text(f"Pipeline failed: {exc}", "error"))
                console.print()
                if output_lines:
                    console.print(neon_status_text("COLMAP output follows", "warning"))
                    for line in output_lines:
                        console.print(Text(f"  {line}"))
                console.print(
                    neon_status_text(f"Staged attempt retained at: {attempt}", "muted")
                )
                _print_summary(step_results, time.monotonic() - overall_start)
                sys.exit(1)
            elapsed = time.monotonic() - started
            progress.update(task_id, description=step_name)
            step_results.append((step_name, elapsed, "✓"))

    model_path = _pick_best_model(output_path)
    stats = read_model_stats(model_path) if model_path is not None else None
    if stats is None or stats.images == 0:
        step_results[-1] = (STEPS[2][0], step_results[-1][1], "✗")
        console.print()
        console.print(
            neon_status_text("COLMAP completed without a valid sparse model", "error")
        )
        console.print(
            neon_status_text(f"Staged attempt retained at: {attempt}", "muted")
        )
        _print_summary(step_results, time.monotonic() - overall_start)
        sys.exit(1)

    assert model_path is not None
    canonical_model = paths.sparse / model_path.name
    mask_files = (
        sorted(
            path
            for path in mask_path.rglob("*.png")
            if path.is_file() and not path.name.startswith("._")
        )
        if mask_path is not None
        else []
    )
    publish_stage(
        paths,
        "colmap",
        generation="reconstruction",
        replacements=((database_path, paths.database), (output_path, paths.sparse)),
        data={
            "inputs": {
                "images": fingerprint_inputs(image_files),
                "masks": fingerprint_inputs(mask_files),
            },
            "parameters": {
                "existing": args.existing,
                "gpu": use_gpu,
                "matcher": matcher,
                "vocab_tree": str(vocab_tree_path) if vocab_tree_path else None,
            },
            "tool": colmap_bin,
            "tool_version": _colmap_version(colmap_bin, paths.root),
            "model": {
                "format": stats.format,
                "registered_images": stats.images,
                "points": stats.points,
            },
            "outputs": {
                "database": str(paths.relative_path(paths.database)),
                "sparse_model": str(paths.relative_path(canonical_model)),
            },
        },
    )
    shutil.rmtree(attempt, ignore_errors=True)
    _print_summary(
        step_results, time.monotonic() - overall_start, n_images, canonical_model
    )


def _print_summary(
    results: list[tuple[str, float, str]],
    total: float,
    n_images: int = 0,
    model_path: Path | None = None,
) -> None:
    console.print()
    table = make_summary_table("Pipeline Summary")
    for name, elapsed, status in results:
        state = Text(status, style=NEON_GREEN if status == "✓" else "bold red")
        state.append(f"  {elapsed:.1f}s")
        table.add_row(name, state)

    console.print(table)
    console.print(neon_status_text(f"Total elapsed: {total:.1f}s", "muted"))

    if model_path and model_path.is_dir():
        n_imgs, n_pts = _read_model_stats(model_path)
        console.print()
        console.print(
            completion_panel(
                "Reconstruction online",
                (
                    ("Output", str(model_path)),
                    ("Registered images", f"{n_imgs} / {n_images}"),
                    ("3D points", f"{n_pts:,}"),
                ),
            )
        )
