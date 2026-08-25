"""Explicitly unmasked experimental Msplat trainer backend."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import Any

from gausser.colmap_model import read_model_stats
from gausser.manifest import fingerprint_inputs, record_stage
from gausser.scene import SceneError, ScenePaths
from gausser.ui import console


def _load_msplat() -> Any:
    try:
        import msplat
    except ImportError as exc:
        raise RuntimeError("Msplat is not installed. Install it with `pip install msplat`.") from exc
    return msplat


def _has_masks(paths: ScenePaths) -> bool:
    return paths.brush_masks.is_dir() or paths.colmap_masks.is_dir()


def _prepare_dataset(paths: ScenePaths, run_dir: Path, images: list[Path]) -> Path:
    """Materialize Msplat's co-located COLMAP model/image input convention."""
    dataset = run_dir / "dataset"
    dataset.mkdir(parents=True)
    model_dir = paths.sparse / "0"
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        shutil.copy2(model_dir / name, dataset / name)
    for image in images:
        destination = dataset / image.name
        try:
            destination.symlink_to(image)
        except OSError:
            shutil.copy2(image, destination)
    return dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="msplatter",
        description="Run experimental, explicitly unmasked Msplat training on a Gausser scene.",
    )
    parser.add_argument("scene", nargs="?", default=Path.cwd(), type=Path, help="Scene root (default: current directory)")
    parser.add_argument("--iterations", type=int, default=7000, help="Training iterations (default: 7000)")
    parser.add_argument("--num-downscales", type=int, default=0, help="Msplat downscale passes (default: 0)")
    parser.add_argument("--no-eval", action="store_true", help="Skip final Msplat evaluation")
    parser.add_argument(
        "--allow-unmasked",
        action="store_true",
        help="Required acknowledgement: Msplat does not consume Gausser's SAM/Brush masks.",
    )
    args = parser.parse_args(argv)
    if args.iterations <= 0 or args.num_downscales < 0:
        parser.error("--iterations must be positive and --num-downscales cannot be negative")

    paths = ScenePaths.from_root(args.scene)
    try:
        images = paths.image_files()
    except SceneError as exc:
        parser.error(str(exc))
    if not images:
        parser.error(f"No images found in {paths.images}")
    model = read_model_stats(paths.sparse / "0")
    if model is None:
        parser.error("Msplat requires a completed COLMAP sparse/0 model")
    if model.format != "binary":
        parser.error("Msplat currently requires COLMAP binary sparse/0 output; text models are unsupported upstream")
    if _has_masks(paths) and not args.allow_unmasked:
        parser.error(
            "This scene contains SAM masks, but Msplat has no mask-input support. "
            "Pass --allow-unmasked only to deliberately train without them."
        )
    if not args.allow_unmasked:
        parser.error("Msplat is experimental and unmasked; pass --allow-unmasked to acknowledge this")

    try:
        msplat = _load_msplat()
    except RuntimeError as exc:
        parser.error(str(exc))

    run_dir = paths.training_run("msplat", paths.new_run_id())
    exports = run_dir / "exports"
    export = exports / "scene.ply"
    checkpoint = run_dir / "checkpoint.msplat"
    run_dir.mkdir(parents=True, exist_ok=False)
    exports.mkdir()
    dataset_path = _prepare_dataset(paths, run_dir, images)

    try:
        dataset = msplat.load_dataset(str(dataset_path), eval_mode=not args.no_eval)
        config = msplat.TrainingConfig(iterations=args.iterations, num_downscales=args.num_downscales)
        trainer = msplat.GaussianTrainer(dataset, config)

        def progress(stats: Any) -> None:
            console.print(f"Msplat {stats.iteration}/{args.iterations}: {stats.splat_count:,} splats")

        trainer.train(progress, callback_every=max(1, min(100, args.iterations)))
        trainer.export_ply(str(export))
        trainer.save_checkpoint(str(checkpoint))
        if not export.is_file() or not checkpoint.is_file():
            raise RuntimeError("Msplat completed without publishing its expected export and checkpoint")
        metrics = {} if args.no_eval else dict(trainer.evaluate())
    except Exception as exc:
        console.print(f"[bold red]Msplat failed; incomplete run retained at {run_dir}: {exc}[/bold red]")
        sys.exit(1)

    record_stage(
        paths,
        "msplat",
        {
            "inputs": fingerprint_inputs(images),
            "parameters": {"iterations": args.iterations, "num_downscales": args.num_downscales, "masked": False},
            "outputs": {
                "run": str(paths.relative_path(run_dir)),
                "dataset": str(paths.relative_path(dataset_path)),
                "export": str(paths.relative_path(export)),
                "checkpoint": str(paths.relative_path(checkpoint)),
                "metrics": metrics,
            },
        },
    )
    console.print(f"[bold yellow]Experimental unmasked Msplat export:[/bold yellow] {export}")
