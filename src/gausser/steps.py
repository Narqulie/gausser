"""The pipeline steps. Each one replaces its own output only after it succeeded."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from .scene import GausserError, Scene, is_image, registered_images, remove, replace, require_tool, run_tool

BRUSH_HINT = "build Brush (cargo build --release) and put `brush` on PATH"
COLMAP_HINT = "brew install colmap"
SHARP_FRAMES_HINT = "pip install sharp-frames && brew install ffmpeg"
SPLAT_TRANSFORM_HINT = "npm install -g @playcanvas/splat-transform"


def frames(scene: Scene, source: Path, extra: list[str]) -> int:
    """Select sharp frames from a video (or image folder) into images/."""
    tool = require_tool("sharp-frames", SHARP_FRAMES_HINT)
    source = source.expanduser().resolve()
    if not source.exists():
        raise GausserError(f"Source does not exist: {source}")
    staged = scene.staging("frames") / "images"
    run_tool([tool, str(source), str(staged), "--force-overwrite", *extra], cwd=scene.root)
    count = sum(1 for path in staged.iterdir() if is_image(path)) if staged.is_dir() else 0
    if count == 0:
        raise GausserError("Sharp Frames selected no images")
    replace(staged, scene.images)
    remove(scene.masks, scene.colmap_masks, scene.database, scene.sparse)  # built from the old images
    print(f"{count} frames in {scene.images}")
    return count


def mask(
    scene: Scene,
    prompt: str,
    *,
    threshold: float = 0.5,
    all_instances: bool = False,
    dilation: int = 0,
    feather: int = 0,
    device: str = "auto",
    checkpoint: str | None = None,
) -> None:
    """Mask the prompted subject with SAM3 into masks/ and colmap/masks/.

    An image without a hit gets an all-white mask (keep the whole image) so one
    miss never throws away the rest of the run.
    """
    images = scene.require_images()
    try:
        import numpy as np
        from PIL import Image

        from . import sam

        predictor = sam.Predictor(device=device, checkpoint=checkpoint)
    except ImportError as exc:
        raise GausserError(f"SAM3 is not available ({exc}). Install: pip install -e '.[sam]', then SAM3 itself (see README)") from exc

    staged = scene.staging("mask")
    brush_dir, colmap_dir = staged / "masks", staged / "colmap-masks"
    brush_dir.mkdir()
    colmap_dir.mkdir()
    misses: list[str] = []
    for index, path in enumerate(images, start=1):
        with Image.open(path) as image:
            result = predictor.predict(image, prompt, threshold, all_instances)
            if result is None:
                misses.append(path.name)
                subject = np.ones((image.height, image.width), dtype=bool)
                note = "no hit, keeping whole image"
            else:
                subject, score = result
                note = f"{score:.2f}"
        sam.save_masks(subject, path, brush_dir, colmap_dir, dilation, feather)
        print(f"[{index}/{len(images)}] {path.name}  {note}", flush=True)

    if len(misses) == len(images):
        raise GausserError(f"SAM3 found no {prompt!r} in any image; try another prompt or a lower --threshold")
    replace(brush_dir, scene.masks)
    replace(colmap_dir, scene.colmap_masks)
    remove(scene.database, scene.sparse)  # COLMAP features depend on the masks
    shutil.rmtree(staged, ignore_errors=True)
    print(f"Masked {len(images) - len(misses)}/{len(images)} images")
    if misses:
        print("No hit (unmasked): " + ", ".join(misses))


def solve(scene: Scene, matcher: str = "auto") -> None:
    """Run COLMAP structure-from-motion into database.db and sparse/0."""
    colmap = require_tool("colmap", COLMAP_HINT)
    images = scene.require_images()
    if matcher == "auto":
        # Exhaustive matching is quadratic; it takes hours on hundreds of video frames.
        matcher = "exhaustive" if len(images) <= 200 else "sequential"
    gpu = "0" if sys.platform == "darwin" else "1"  # Homebrew COLMAP on macOS has no CUDA.

    staged = scene.staging("solve")
    database, sparse = staged / "database.db", staged / "sparse"
    extract = [
        colmap, "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(scene.images),
        "--ImageReader.single_camera", "1",
        "--FeatureExtraction.use_gpu", gpu,
    ]
    if scene.colmap_masks.is_dir():
        extract += ["--ImageReader.mask_path", str(scene.colmap_masks)]
    run_tool(extract, cwd=staged)
    run_tool(
        [colmap, f"{matcher}_matcher", "--database_path", str(database), "--FeatureMatching.use_gpu", gpu],
        cwd=staged,
    )
    sparse.mkdir()
    run_tool(
        [colmap, "mapper", "--database_path", str(database), "--image_path", str(scene.images), "--output_path", str(sparse)],
        cwd=staged,
    )

    # COLMAP can split the scene into several models; keep the largest as sparse/0.
    models = [path for path in sparse.iterdir() if path.is_dir() and path.name.isdigit()]
    best = max(models, key=registered_images, default=None)
    if best is None or registered_images(best) == 0:
        raise GausserError("COLMAP did not produce a model; check the images overlap enough")
    final = staged / "final"
    final.mkdir()
    best.rename(final / "0")
    replace(database, scene.database)
    replace(final, scene.sparse)
    shutil.rmtree(staged, ignore_errors=True)
    print(f"Model: {registered_images(scene.sparse / '0')}/{len(images)} images registered ({matcher} matching)")


def train_brush(
    scene: Scene,
    *,
    iterations: int = 30000,
    export_every: int = 5000,
    max_splats: int = 1_500_000,
    max_resolution: int | None = None,
) -> Path:
    """Train with Brush into a new runs/brush/<time>/ folder."""
    brush = require_tool("brush", BRUSH_HINT)
    scene.require_model()
    staged = scene.staging("brush")
    command = [
        brush, str(scene.root),
        "--total-train-iters", str(iterations),
        "--export-every", str(export_every),
        "--export-path", str(staged / "exports"),
        "--eval-every", "1000",
        "--refine-every", "200",
        # Unbounded density stalls Metal on Apple Silicon; 1.5M is the tested ceiling.
        "--max-splats", str(max_splats),
    ]
    if max_resolution:
        command += ["--max-resolution", str(max_resolution)]
    run_tool(command, cwd=scene.root)
    if not list(staged.glob("exports/*.ply")):
        raise GausserError(f"Brush finished without a .ply export; output kept in {staged}")
    run = scene.new_run("brush")
    replace(staged, run)
    print(f"Exports in {run / 'exports'}")
    return run


def lod(
    scene: Scene,
    source: Path | None = None,
    *,
    chunk_count: int = 512,
    chunk_extent: float = 16.0,
    percentages: tuple[int, ...] = (100, 50, 25),
) -> Path:
    """Build a streamed PlayCanvas LOD set into runs/lod/<time>/."""
    tool = require_tool("splat-transform", SPLAT_TRANSFORM_HINT)
    if source is None:
        exports = scene.exports()
        if not exports:
            raise GausserError("No trained .ply export found; run `gausser train` first")
        source = exports[-1]
    source = source.expanduser().resolve()
    if not source.is_file():
        raise GausserError(f"LOD source does not exist: {source}")
    if percentages[0] != 100 or list(percentages) != sorted(set(percentages), reverse=True) or percentages[-1] <= 0:
        raise GausserError("LOD percentages must start at 100 and strictly decrease, e.g. 100,50,25")

    levels = scene.staging("lod-levels")
    staged = scene.staging("lod")
    level_files = [source]
    for level, percentage in enumerate(percentages[1:], start=1):
        level_file = levels / f"lod-{level}.ply"
        run_tool([tool, str(source), "--decimate", f"{percentage}%", str(level_file)], cwd=scene.root)
        level_files.append(level_file)
    bundle = [tool, "--lod-chunk-count", str(chunk_count), "--lod-chunk-extent", f"{chunk_extent:g}"]
    for level, level_file in enumerate(level_files):
        bundle += [str(level_file), "-l", str(level)]
    run_tool([*bundle, str(staged / "lod-meta.json")], cwd=scene.root)
    if not (staged / "lod-meta.json").is_file():
        raise GausserError("splat-transform did not write lod-meta.json")
    (staged / "source.txt").write_text(f"{source}\n")
    run = scene.new_run("lod")
    replace(staged, run)
    shutil.rmtree(levels, ignore_errors=True)
    print(f"LOD set in {run}")
    return run
