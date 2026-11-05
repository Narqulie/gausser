"""The pipeline steps. Each one replaces its own output only after it succeeded."""

from __future__ import annotations

from fnmatch import fnmatchcase
import json
import os
import shutil
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Any

from PIL import Image

from . import pano, qoocam
from .scene import GausserError, Scene, is_image, registered_images, remove, replace, require_tool, run_tool

BRUSH_HINT = "build Brush (cargo build --release) and put `brush` on PATH"
COLMAP_HINT = "brew install colmap"
SHARP_FRAMES_HINT = "pip install sharp-frames && brew install ffmpeg"
SPLAT_TRANSFORM_HINT = "npm install -g @playcanvas/splat-transform"


def frames(scene: Scene, source: Path, extra: list[str], ranges: list[tuple[float, float]] | None = None) -> int:
    """Select sharp frames from a video (or image folder) into images/; `ranges` keeps only those seconds.

    360° frames go to panos/ instead, and images/ gets their pinhole views (pano.py): 2:1
    equirectangular frames, or a QooCam 8K raw video's two fisheye images side by side, with
    the camera's calibration in panos/lenses.json and the lens-circle masks in masks-exclude/.
    """
    tool = require_tool("sharp-frames", SHARP_FRAMES_HINT)
    source = source.expanduser().resolve()
    if not source.exists():
        raise GausserError(f"Source does not exist: {source}")
    staging = scene.staging("frames")
    staged = staging / "images"
    rig = qoocam.calibration(source) if source.is_file() else None
    if rig or ranges:
        if not source.is_file():
            raise GausserError("--range needs a video file")
        source = _cut(source, staging / "videos", ranges or [], fisheyes=rig is not None)
    run_tool([tool, str(source), str(staged), "--force-overwrite", *extra], cwd=scene.root)
    count = sum(1 for path in staged.iterdir() if is_image(path)) if staged.is_dir() else 0
    if count == 0:
        raise GausserError("Sharp Frames selected no images")
    panos = sorted(path for path in staged.iterdir() if is_image(path))
    keep = staging / "keep"
    if rig or pano.is_pano(panos[0]):
        views = staging / "views"
        views.mkdir()
        print(f"{count} 360° frames: rendering {pano.VIEWS} views each", flush=True)
        size = pano.render(panos, views, rig)
        if rig:
            (staged / pano.LENSES).write_text(json.dumps(rig, indent=2))
            keep.mkdir()
            _write_keep_masks(views, keep, pano.keep_masks(rig, size) or [])
        replace(views, scene.images)
        replace(staged, scene.panos)
    else:
        replace(staged, scene.images)
        remove(scene.panos)
    remove(scene.masks, scene.masks_exclude, scene.colmap_masks, scene.database, scene.sparse)  # built from the old images
    if keep.is_dir():
        replace(keep, scene.masks_exclude)
    remove(staging)
    print(f"{count} frames in {scene.panos if scene.is_pano() else scene.images}")
    return count


def _cut(source: Path, out: Path, ranges: list[tuple[float, float]], fisheyes: bool) -> Path:
    """Re-encode `ranges` of `source` (all of it if none) as videos in `out`, the two lens streams side by side."""
    ffmpeg = require_tool("ffmpeg", SHARP_FRAMES_HINT)
    out.mkdir()
    if sys.platform == "darwin":
        encode = ["-c:v", "hevc_videotoolbox", "-q:v", "80", "-tag:v", "hvc1"]
    else:
        encode = ["-c:v", "libx264", "-crf", "16"]
    for index, span in enumerate(ranges or [None], start=1):
        command = [ffmpeg, "-hide_banner", "-y"]
        if span:
            command += ["-ss", f"{span[0]:g}", "-to", f"{span[1]:g}"]
        command += ["-i", str(source)]
        command += ["-filter_complex", "[0:0][0:1]hstack"] if fisheyes else ["-map", "0:v:0"]
        run_tool([*command, "-an", *encode, str(out / f"part{index}.mp4")], cwd=out)
    return out


def _write_keep_masks(views: Path, out: Path, keep: list[Any]) -> None:
    """Brush-layout masks out/<stem>.png of every view: white where the lens sees the scene."""
    import numpy as np

    images = [Image.fromarray(mask.astype(np.uint8) * 255, mode="L") for mask in keep]
    for view in views.iterdir():
        images[pano.view_index(view.name)].save(out / f"{view.stem}.png")


NO_MASKS = "no-masks"  # marker in sparse/: the model was solved without masks, so new masks keep it


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
    miss: str = "keep",
    exclude: bool = False,
) -> None:
    """Mask the prompted subject with SAM3 into masks/ and colmap/masks/.

    With `exclude`, write only masks-exclude/ instead (see `exclude_mask`); masks/, colmap/masks/
    and the COLMAP model stay as they are, and `miss` and `feather` do not apply.

    An image without a hit gets an all-white mask (`miss="keep"`: train on the whole image) so
    one miss never throws away the rest of the run. With `miss="drop"` its Brush mask is
    all black (ignored in training) and its COLMAP mask stays white (still registers).
    """
    images = scene.require_images()
    try:
        import numpy as np
        from PIL import Image

        from . import sam

        predictor = sam.Predictor(device=device, checkpoint=checkpoint)
    except ImportError as exc:
        raise GausserError(f"SAM3 is not available ({exc}). Install: pip install -e '.[sam]', then SAM3 itself (see README)") from exc
    except OSError as exc:  # missing checkpoint, or a gated Hugging Face download without login
        raise GausserError(
            f"Could not load the SAM3 model ({exc}). Put the checkpoint at ~/.cache/gausser/sam3.pt, "
            "pass --sam-checkpoint, or run `hf auth login`"
        ) from exc

    staged = scene.staging("mask")
    brush_dir, colmap_dir = staged / "masks", staged / "colmap-masks"
    brush_dir.mkdir()
    if exclude:
        _exclude(scene, images, predictor, prompt, threshold, dilation, brush_dir)
        replace(brush_dir, scene.masks_exclude)
        shutil.rmtree(staged, ignore_errors=True)
        if scene.is_pano() and not (scene.sparse / NO_MASKS).is_file():
            # The operator moves with a 360° camera; `solve` masks its features out.
            remove(scene.database, scene.sparse)
        return
    colmap_dir.mkdir()
    misses: list[str] = []
    for index, path in enumerate(images, start=1):
        with Image.open(path) as image:
            result = predictor.predict(image, prompt, threshold, all_instances)
            colmap_subject = None
            if result is None:
                misses.append(path.name)
                subject = np.ones((image.height, image.width), dtype=bool)
                note = "no hit, keeping whole image"
                if miss == "drop":
                    colmap_subject, subject = subject, ~subject
                    note = "no hit, dropped from training"
            else:
                subject, score = result
                note = f"{score:.2f}"
        sam.save_masks(subject, path, brush_dir, colmap_dir, dilation, feather, colmap_subject)
        print(f"[{index}/{len(images)}] {path.name}  {note}", flush=True)

    if len(misses) == len(images):
        raise GausserError(f"SAM3 found no {prompt!r} in any image; try another prompt or a lower --threshold")
    replace(brush_dir, scene.masks)
    replace(colmap_dir, scene.colmap_masks)
    if not (scene.sparse / NO_MASKS).is_file():  # a `solve --no-masks` model never read the masks
        remove(scene.database, scene.sparse)  # COLMAP features depend on the masks
    shutil.rmtree(staged, ignore_errors=True)
    print(f"Masked {len(images) - len(misses)}/{len(images)} images")
    if misses:
        outcome = "dropped from training, kept for COLMAP" if miss == "drop" else "unmasked"
        print(f"No hit ({outcome}): " + ", ".join(misses))


def exclude_mask(hit: Any, protect: Any, dilation: int) -> Any:
    """White (True) = train: everything except ``hit`` grown by ``dilation`` px, but never inside ``protect``."""
    import numpy as np
    from PIL import Image, ImageFilter

    grown = Image.fromarray(hit.astype(np.uint8) * 255, mode="L")
    if dilation > 0:
        grown = grown.filter(ImageFilter.MaxFilter(dilation * 2 + 1))
    return (np.asarray(grown) <= 127) | protect


def _exclude(
    scene: Scene, images: list[Path], predictor: Any, prompt: str, threshold: float, dilation: int, out: Path,
) -> None:
    """Write out/<stem>.png: black where every instance of ``prompt`` is, white elsewhere and inside masks/."""
    import numpy as np
    from PIL import Image

    hits, share = 0, 0.0
    with Image.open(images[0]) as first:
        keep = pano.keep_masks(pano.lenses(scene.panos), first.width) if scene.is_pano() else None
    for index, path in enumerate(images, start=1):
        with Image.open(path) as image:
            result = predictor.predict(image, prompt, threshold, True)
            hit = np.zeros((image.height, image.width), dtype=bool) if result is None else result[0]
            protect = np.zeros_like(hit)
            subject = scene.masks / f"{path.stem}.png"
            if subject.is_file():  # SAM3 may call a statue a person; the subject mask wins
                with Image.open(subject) as m:
                    protect = np.asarray(m.convert("L").resize(image.size)) > 127
        white = exclude_mask(hit, protect, dilation)
        if keep is not None:  # outside a fisheye's circle and around the grip
            white &= keep[pano.view_index(path.name)]
        Image.fromarray(white.astype(np.uint8) * 255, mode="L").save(out / f"{path.stem}.png")
        excluded = 1 - white.mean()
        hits += result is not None
        share += excluded
        note = "no hit" if result is None else f"{excluded:.1%} excluded"
        print(f"[{index}/{len(images)}] {path.name}  {note}", flush=True)
    print(f"Excluded {prompt!r} in {hits}/{len(images)} images, mean {share / len(images):.1%} of each image")


def solve(scene: Scene, matcher: str = "auto", no_masks: bool = False) -> None:
    """Run COLMAP structure-from-motion into database.db and sparse/0; `no_masks` ignores colmap/masks/.

    In a 360° scene the views of one panorama form a COLMAP rig with fixed, known rotations, and
    masks-exclude/ also masks features (the operator moves with the camera, so it would match).
    """
    colmap = require_tool("colmap", COLMAP_HINT)
    images = scene.require_images()
    if matcher == "auto":
        # Exhaustive matching is quadratic; it takes hours on hundreds of images.
        matcher = "exhaustive" if len(images) <= 200 else "sequential"
    gpu = "0" if sys.platform == "darwin" else "1"  # Homebrew COLMAP on macOS has no CUDA.

    staged = scene.staging("solve")
    database, sparse = staged / "database.db", staged / "sparse"
    # COLMAP would otherwise try every file in images/, including AppleDouble `._*` sidecars on exFAT.
    # The mapper reads only what is in the database, so it needs no list.
    image_list = staged / "images.txt"
    image_list.write_text("".join(f"{image.name}\n" for image in images))
    extract = [
        colmap, "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(scene.images),
        "--image_list_path", str(image_list),
        "--ImageReader.single_camera", "1",
        "--FeatureExtraction.use_gpu", gpu,
    ]
    rig = scene.is_pano()
    lenses = pano.lenses(scene.panos) if rig else None
    if rig:
        masks = staged / "masks"
        masks.mkdir()
        pano.feature_masks(images, masks, [] if no_masks else [scene.colmap_masks, scene.masks_exclude], lenses)
        with Image.open(images[0]) as first:
            params = pano.camera_params(first.width)
        extract += [
            "--ImageReader.mask_path", str(masks),
            "--ImageReader.camera_model", "SIMPLE_PINHOLE", "--ImageReader.camera_params", params,
        ]
    elif scene.colmap_masks.is_dir() and not no_masks:
        extract += ["--ImageReader.mask_path", str(scene.colmap_masks)]
    run_tool(extract, cwd=staged)
    match = [colmap, f"{matcher}_matcher", "--database_path", str(database), "--FeatureMatching.use_gpu", gpu]
    mapper = [colmap, "mapper", "--database_path", str(database), "--image_path", str(scene.images), "--output_path", str(sparse)]
    if rig:
        pano.split_cameras(database)
        pano.write_rig(staged / "rig.json", lenses)
        run_tool(
            [colmap, "rig_configurator", "--database_path", str(database), "--rig_config_path", str(staged / "rig.json")],
            cwd=staged,
        )
        # The views of one frame see disjoint directions (feature_masks); the rig is calibrated.
        match += ["--FeatureMatching.rig_verification", "1", "--FeatureMatching.skip_image_pairs_in_same_frame", "1"]
        mapper += [
            "--Mapper.ba_refine_sensor_from_rig", "0", "--Mapper.ba_refine_focal_length", "0",
            "--Mapper.ba_refine_principal_point", "0", "--Mapper.ba_refine_extra_params", "0",
        ]
    if matcher == "sequential":
        # Loop detection adds vocabulary-tree matches between non-neighbours, so a walk that
        # returns to its start, or unordered photos, still connect. COLMAP downloads the tree
        # once into ~/.cache/colmap.
        match += ["--SequentialMatching.loop_detection", "1"]
    run_tool(match, cwd=staged)
    sparse.mkdir()
    # Snapshots of the growing model, about 20 per solve, for `gausser monitor`.
    (staged / "snapshots").mkdir()
    mapper += [
        "--Mapper.snapshot_path", str(staged / "snapshots"),
        "--Mapper.snapshot_frames_freq", str(max(5, len(images) // (pano.VIEWS if rig else 1) // 20)),
    ]
    run_tool(mapper, cwd=staged)

    # COLMAP can split the scene into several models; keep the largest as sparse/0.
    models = [path for path in sparse.iterdir() if path.is_dir() and path.name.isdigit()]
    best = max(models, key=registered_images, default=None)
    if best is None or registered_images(best) == 0:
        raise GausserError("COLMAP did not produce a model; check the images overlap enough")
    final = staged / "final"
    final.mkdir()
    best.rename(final / "0")
    if no_masks:
        (final / NO_MASKS).touch()
    replace(database, scene.database)
    replace(final, scene.sparse)
    shutil.rmtree(staged, ignore_errors=True)
    registered = registered_images(scene.sparse / "0")
    print(f"Model: {registered}/{len(images)} images registered ({matcher} matching)")
    if registered * 2 < len(images):
        # Training on a fraction of the cameras wastes the run; stop `gausser run` here.
        raise GausserError(
            f"COLMAP placed only {registered} of {len(images)} images. Use more photos with more overlap "
            "(from a video: `gausser frames VIDEO -- --fps 20 --min-buffer 1`), "
            "or run `gausser train` to train on these anyway"
        )


def train_brush(
    scene: Scene,
    *,
    iterations: int = 30000,
    export_every: int = 5000,
    max_splats: int = 1_500_000,
    max_resolution: int | None = None,
    eval_split: int | None = None,
    extra: list[str] | None = None,
    no_masks: bool = False,
    exclude_masks: bool = False,
) -> Path:
    """Train with Brush into a new runs/brush/<time>/ folder; `extra` goes to Brush last.

    `no_masks` leaves masks/ out of Brush's folder, so it trains on whole images.
    `exclude_masks` gives Brush masks-exclude/ as its masks (a context run without transient objects).
    """
    brush = require_tool("brush", BRUSH_HINT)
    scene.require_model()
    if exclude_masks and no_masks:
        raise GausserError("--exclude-masks and --no-masks cannot be combined")
    if exclude_masks and not scene.masks_exclude.is_dir():
        raise GausserError("No masks-exclude/ folder; make it with `gausser mask PROMPT --exclude`")
    staged = scene.staging("brush")
    # Brush scans every file under the folder it is given and takes any cameras.bin,
    # image or "masks" folder it finds (Msplat datasets, failed attempts, colmap/masks).
    # Give it a folder that holds only the current inputs.
    dataset = staged / "dataset"
    dataset.mkdir()
    if not no_masks and not scene.masks.is_dir() and pano.lenses(scene.panos):
        exclude_masks = True  # fisheye views are black outside the lens circle; never train on that
    brush_masks = None if no_masks else scene.masks_exclude if exclude_masks else scene.masks
    for name, source in (("images", scene.images), ("sparse", scene.sparse), ("masks", brush_masks)):
        if source and source.is_dir():
            (dataset / name).symlink_to(source)
    command = [
        brush, str(dataset),
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
    if eval_split:
        # The renders of the held-out images land in exports/eval_<step>/images/; `gausser monitor` shows them.
        command += ["--eval-split-every", str(eval_split), "--eval-save-to-disk"]
    command += extra or []
    # Brush logs eval PSNR/SSIM, refine steps and timing only at info level; without a
    # terminal (unattended runs) nothing else is printed.
    run_tool(command, cwd=scene.root, env={"RUST_LOG": os.environ.get("RUST_LOG", "brush_cli=info")})
    if not list(staged.glob("exports/*.ply")):
        raise GausserError(f"Brush finished without a .ply export; output kept in {staged}")
    remove(dataset)
    # `gausser composite` finds the statue (masked) and context (unmasked, excluded) runs by this marker.
    marker = "excluded" if exclude_masks else "unmasked" if no_masks or not scene.masks.is_dir() else "masked"
    (staged / marker).touch()
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


def subset(scene: Scene, patterns: list[str], out: Path) -> Scene:
    """Make a new scene from the images of a solved scene whose file names match `patterns`.

    The COLMAP model keeps its cameras, 3D points and coordinates, only the other images are
    removed, so splats trained on different subsets share one coordinate frame. The source is only read.
    """
    colmap = require_tool("colmap", COLMAP_HINT)
    model = scene.require_model()
    images = scene.image_files()
    chosen = [image for image in images if any(fnmatchcase(image.name, pattern) for pattern in patterns)]
    if not chosen:
        raise GausserError(f"No image in {scene.images} matches {' '.join(patterns)}")
    out = out.expanduser().resolve()
    if out.joinpath("images").exists():
        raise GausserError(f"{out / 'images'} already exists; not overwriting")
    out.mkdir(parents=True, exist_ok=True)
    target = Scene(out)
    keep = {image.name for image in chosen}

    staged = target.staging("subset")
    (staged / "images").mkdir()
    for image in chosen:
        shutil.copy(image, staged / "images" / image.name)
    # Brush masks are <stem>.png, COLMAP masks <name><ext>.png; copy the ones that exist.
    for source, folder in ((scene.masks, "masks"), (scene.colmap_masks, "colmap-masks")):
        if source.is_dir():
            (staged / folder).mkdir()
            for image in chosen:
                mask = source / (f"{image.stem}.png" if source == scene.masks else f"{image.name}.png")
                if mask.is_file():
                    shutil.copy(mask, staged / folder / mask.name)
    # image_deleter takes the names to delete; names it does not know are skipped.
    names = staged / "delete.txt"
    names.write_text("".join(f"{image.name}\n" for image in images if image.name not in keep))
    (staged / "sparse" / "0").mkdir(parents=True)  # image_deleter needs the output folder to exist
    run_tool(
        [colmap, "image_deleter", "--input_path", str(model), "--output_path", str(staged / "sparse" / "0"),
         "--image_names_path", str(names),
         "--log_severity", "1"],  # else one INFO line per deleted image
        cwd=staged,
    )
    registered = registered_images(staged / "sparse" / "0")
    if registered == 0:
        raise GausserError("None of the matching images is registered in the COLMAP model")
    # The database only records which images the model was built from (`solve_done`); keep it consistent.
    if scene.database.is_file():
        shutil.copyfile(scene.database, staged / "database.db")
        _filter_database(staged / "database.db", keep)

    for folder, destination in (
        ("images", target.images), ("masks", target.masks), ("colmap-masks", target.colmap_masks),
        ("sparse", target.sparse), ("database.db", target.database),
    ):
        if (staged / folder).exists():
            replace(staged / folder, destination)
    shutil.rmtree(staged, ignore_errors=True)
    masks = sum(1 for _ in target.masks.glob("*.png")) if target.masks.is_dir() else 0
    print(f"Subset in {out}: {len(chosen)} images, {masks} masks, {registered} registered in the model")
    if not scene.database.is_file():
        print("No database.db in the source scene, so `gausser run` would solve the subset again; `gausser train` works")
    return target


def _filter_database(path: Path, keep: set[str]) -> None:
    """Drop every image not in `keep` (and its features, matches and rig frame) from a COLMAP database."""
    with closing(sqlite3.connect(path)) as database:
        database.execute("CREATE TEMP TABLE kept (id INTEGER PRIMARY KEY)")
        database.executemany("INSERT INTO kept SELECT image_id FROM images WHERE name = ?", [(name,) for name in keep])
        for table in ("keypoints", "descriptors", "images"):
            database.execute(f"DELETE FROM {table} WHERE image_id NOT IN (SELECT id FROM kept)")
        # pair_id = image_id1 * 2147483647 + image_id2
        for table in ("matches", "two_view_geometries"):
            database.execute(
                f"DELETE FROM {table} WHERE pair_id / 2147483647 NOT IN (SELECT id FROM kept) "
                "OR pair_id % 2147483647 NOT IN (SELECT id FROM kept)"
            )
        database.execute("DELETE FROM pose_priors WHERE corr_data_id NOT IN (SELECT id FROM kept)")
        database.execute("DELETE FROM frame_data WHERE data_id NOT IN (SELECT id FROM kept)")
        database.execute("DELETE FROM frames WHERE frame_id NOT IN (SELECT frame_id FROM frame_data)")
        database.commit()
        database.execute("VACUUM")
