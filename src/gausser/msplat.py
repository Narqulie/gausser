"""Experimental Msplat trainer with checkpoint resume.

Msplat has no mask input. With masks, each image is composited onto a black
background and Msplat trains against that same black background, so masked-out
areas are learned as empty space.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
from pathlib import Path
from typing import Any

from .scene import GausserError, Scene

INSTALL_HINT = "pip install -e '.[msplat]' (Apple Silicon, macOS 14+)"
BACKGROUND = [0.0, 0.0, 0.0]


def _composite(image: Path, mask: Path, output: Path) -> None:
    from PIL import Image

    with Image.open(image) as source, Image.open(mask) as mask_file:
        rgb = source.convert("RGB")
        alpha = mask_file.convert("L")
    if alpha.size != rgb.size:
        alpha = alpha.resize(rgb.size)
    Image.composite(rgb, Image.new("RGB", rgb.size), alpha).save(output, quality=95)


def train(scene: Scene, *, iterations: int, downscale_factor: float, checkpoint_every: int, resume: str | None) -> Path:
    try:
        import msplat
    except ImportError as exc:
        raise GausserError(f"Msplat is not installed. Install: {INSTALL_HINT}") from exc

    if resume:
        run = scene.runs / "msplat" / resume
        if not (run / "settings.json").is_file():
            raise GausserError(f"No Msplat run {resume!r} in {run.parent}")
        settings: dict[str, Any] = json.loads((run / "settings.json").read_text())
        if (run / "exports" / "scene.ply").exists():
            raise GausserError(f"Msplat run {resume} is already finished")
    else:
        model = scene.require_model()
        images = scene.require_images()
        run = scene.new_run("msplat")
        dataset = run / "dataset"  # Msplat expects the model files beside the images.
        dataset.mkdir(parents=True)
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            shutil.copy2(model / name, dataset / name)
        masked = scene.masks_done(images)
        for index, image in enumerate(images, start=1):
            if masked:
                _composite(image, scene.masks / f"{image.stem}.png", dataset / image.name)
                print(f"\rMasking images for Msplat {index}/{len(images)}", end="", flush=True)
            else:
                (dataset / image.name).symlink_to(image)
        if masked:
            print()
        settings = {"iterations": iterations, "downscale_factor": downscale_factor, "masked": masked}
        (run / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    iterations = int(settings["iterations"])
    config = msplat.TrainingConfig(iterations=iterations, num_downscales=0, downscale_factor=settings["downscale_factor"])
    if settings.get("masked"):
        config.bg_color = BACKGROUND
    dataset_path = str(run / "dataset")
    trainer = msplat.GaussianTrainer(
        msplat.load_dataset(dataset_path, downscale_factor=settings["downscale_factor"], eval_mode=False),
        config,
    )
    checkpoint = run / "checkpoint.msplat"
    if checkpoint.exists():
        trainer.load_checkpoint(str(checkpoint))
        print(f"Resuming {run.name} at iteration {int(trainer.iteration):,}", flush=True)

    def save_checkpoint() -> None:
        temporary = checkpoint.with_name("checkpoint.tmp")
        trainer.save_checkpoint(str(temporary))
        os.replace(temporary, checkpoint)

    # Finish the current step on Ctrl+C, save, then stop: a native step must not be torn.
    stop: list[int] = []
    previous = signal.signal(signal.SIGINT, lambda signum, _frame: stop.append(signum))
    try:
        iteration = int(trainer.iteration)
        while iteration < iterations:
            stats = trainer.step()
            iteration = int(stats.iteration)
            if iteration % 500 == 0:
                print(f"{iteration:,}/{iterations:,}  {stats.splat_count:,} splats", flush=True)
            if stop or iteration % checkpoint_every == 0 or iteration == iterations:
                save_checkpoint()
            if stop:
                raise GausserError(f"Interrupted; resume with: gausser train --trainer msplat --resume {run.name}")
    finally:
        signal.signal(signal.SIGINT, previous)

    export = run / "exports" / "scene.ply"
    export.parent.mkdir(exist_ok=True)
    trainer.export_ply(str(export))
    return export
