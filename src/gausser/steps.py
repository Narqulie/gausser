"""The pipeline steps. Each one replaces its own output only after it succeeded."""

from __future__ import annotations

from fnmatch import fnmatchcase
import json
import math
import os
import shutil
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Any

from PIL import Image

from . import arkit, colmap_model, crop as cropping, geo, keyframes, pano, ply, qoocam
from .scene import GEO_FILE, GausserError, Scene, is_image, is_video, registered_images, remove, replace, require_tool, run_tool

DENSE_INIT_POINTS = 500_000  # Brush's starting splats from dense/; it grows to --max-splats from there
# `train --adaptive`: after --iterations, train this many more steps at a time while held-out PSNR gains this much.
EXTEND_STEPS = 5000
MIN_GAIN_DB = 0.1
ADAPTIVE_EVAL_SPLIT = 20  # every 20th image held out when --eval-split is not given


def frames(
    scene: Scene, source: Path, ranges: list[tuple[float, float]] | None = None, *, spacing: str = "medium",
    min_frames: int = keyframes.MIN_FRAMES, max_frames: int = keyframes.MAX_FRAMES, every: int | None = None,
    overlap: float | None = None,
) -> int:
    """Select frames from a video (or a folder of videos) into images/; `ranges` keeps only those seconds.

    keyframes.py picks sharp frames where the camera has moved: one pass analyses the video, one
    extracts the picked frames. A folder of photos is used whole. 360° frames go to panos/ instead,
    and images/ gets their pinhole views (pano.py): 2:1 equirectangular frames, or a QooCam 8K raw
    video's two fisheye images side by side, with the camera's calibration in panos/lenses.json and
    the lens-circle masks in masks-exclude/. A Stray Scanner LiDAR scan folder (arkit.py) gives frames
    from its rgb.mp4 plus arkit/ (ARKit poses and LiDAR points of those frames).

    `overlap` (0-1) is for mapping: a frame whenever the view has moved so far that the next one shares
    only that much of it, however many frames that gives. A DJI video's subtitle file (geo.py) gives each
    frame its GPS position in EXIF, as drone photos have.
    """
    source = source.expanduser().resolve()
    if not source.exists():
        raise GausserError(f"Source does not exist: {source}")
    staging = scene.staging("frames")
    staged = staging / "images"
    staged.mkdir()
    scan = arkit.is_scan(source)
    photos = sorted(path for path in source.iterdir() if is_image(path)) if source.is_dir() and not scan else []
    videos = [source / "rgb.mp4"] if scan else sorted(path for path in source.iterdir() if is_video(path)) if source.is_dir() else [source]
    if ranges and not source.is_file():
        raise GausserError("--range needs a video file")
    if photos and videos and source.is_dir():
        raise GausserError(f"{source} has both photos and videos; use a folder of one kind")
    rig = qoocam.calibration(photos[0] if photos else videos[0]) if (photos or videos) and not scan else None
    if photos:
        for photo in photos:
            if rig:
                qoocam.side_by_side(photo, staged / f"{photo.stem}.jpg")
            else:
                shutil.copy2(photo, staged / photo.name)
    elif videos:
        items = _select_frames(videos, ranges, staging, staged, rig, spacing, min_frames, max_frames, every, overlap)
        if scan:
            arkit.import_scan(source, staged, items, staging / "arkit")
    count = sum(1 for path in staged.iterdir() if is_image(path))
    if count == 0:
        raise GausserError(f"No photos or videos in {source}")
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
    # built from the old images
    remove(scene.masks, scene.masks_exclude, scene.colmap_masks, scene.database, scene.sparse, scene.dense, scene.arkit)
    if keep.is_dir():
        replace(keep, scene.masks_exclude)
    if scan:
        replace(staging / "arkit", scene.arkit)
    remove(staging)
    print(f"{count} frames in {scene.panos if scene.is_pano() else scene.images}")
    return count


def _select_frames(
    videos: list[Path], ranges: list[tuple[float, float]] | None, staging: Path, staged: Path, rig: dict[str, Any] | None,
    spacing: str, min_frames: int, max_frames: int, every: int | None, overlap: float | None,
) -> list[dict[str, Any]]:
    """Analyse every part (video or range), choose one level for all of them, extract into `staged` with frames.json.

    With `overlap`, each part has one level, the threshold for that overlap, and the frame count is not limited.
    """
    ffmpeg, ffprobe = require_tool("ffmpeg"), require_tool("ffprobe")
    spans = ranges or [None]  # type: ignore[list-item]
    parts = [(video, span) for video in videos for span in spans]
    infos, analysed = [], []
    for number, (video, span) in enumerate(parts, start=1):
        print(f"== {video.name}" + (f" {span[0]:g}-{span[1]:g} s" if span else "") + f" ({number}/{len(parts)})", flush=True)
        infos.append(keyframes.probe(ffprobe, video, span))
        levels = (keyframes.overlap_level(infos[-1], overlap),) if overlap else keyframes.LEVELS
        analysed.append(keyframes.analyse(ffmpeg, video, infos[-1], span, every, levels))
    counts = [sum(len(part[level]) for part in analysed) for level in range(len(analysed[0]))]
    fixed = every or overlap
    level = 0 if fixed else keyframes.choose(counts, keyframes.SPACINGS[spacing], min_frames, max_frames)
    note = (
        f"every {every} frames" if every else f"{overlap:.0%} overlap" if overlap
        else f"level {level} of {len(counts) - 1} (spacing {spacing}); frames per level {counts}"
    )
    print(f"Selecting {counts[level]} frames: {note}", flush=True)
    if counts[level] < min_frames and not fixed:
        print(f"warning: fewer than --min-frames {min_frames}; the video is short or hardly moves", flush=True)
    items = []
    tracks = {video: geo.track(video) for video in videos}
    for found in filter(None, tracks.values()):
        print(f"GPS from {found.source.name}: {len(found.times)} positions", flush=True)
    if overlap and not any(tracks.values()):
        print("warning: no GPS for these frames, so maps will not be georeferenced. "
              "Turn on Video Subtitles in DJI Fly; the drone then writes a .SRT next to each video", flush=True)
    for number, ((video, span), info, picked) in enumerate(zip(parts, infos, analysed, strict=True), start=1):
        picks = picked[level]
        if not picks:
            continue
        print(f"Extracting {len(picks)} frames from {video.name}", flush=True)
        prefix = f"p{number:02d}_" if len(parts) > 1 else ""
        paths = keyframes.extract(ffmpeg, video, info, span, [p.frame for p in picks], staging / f"extract{number}", rig is not None)
        for path, pick in zip(paths, picks, strict=True):
            name = f"{prefix}frame_{pick.frame:06d}.jpg"
            path.rename(staged / name)
            time = (span[0] if span else 0) + pick.frame / info.fps
            items.append({
                "name": name, "source": video.name, "frame": pick.frame, "time": round(time, 3),
                "motion": round(pick.motion, 4), "sharpness": round(pick.sharpness, 1),
            })
            if (track := tracks[video]) is not None:
                geo.add_gps(staged / name, *track.at(time))
    method = f"every {every}" if every else f"overlap {overlap:g}" if overlap else "motion"
    record = {"method": method, "level": level, "per_level": counts, "analysed": sum(i.frames for i in infos), "total_selected": len(items), "items": items}
    (staged / "frames.json").write_text(json.dumps(record, indent=1))
    return items


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
            remove(scene.database, scene.sparse, scene.dense)
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
        remove(scene.database, scene.sparse, scene.dense)  # COLMAP features depend on the masks
    shutil.rmtree(staged, ignore_errors=True)
    print(f"Masked {len(images) - len(misses)}/{len(images)} images")
    if misses:
        outcome = "dropped from training, kept for COLMAP" if miss == "drop" else "unmasked"
        print(f"No hit ({outcome}): " + ", ".join(misses))


def exclude_mask(hit: Any, protect: Any, dilation: int) -> Any:
    """White (True) = train: everything except ``hit`` grown by ``dilation`` px, but never inside ``protect``."""
    import numpy as np
    from PIL import ImageFilter

    grown = Image.fromarray(hit.astype(np.uint8) * 255, mode="L")
    if dilation > 0:
        grown = grown.filter(ImageFilter.MaxFilter(dilation * 2 + 1))
    return (np.asarray(grown) <= 127) | protect


def _exclude(
    scene: Scene, images: list[Path], predictor: Any, prompt: str, threshold: float, dilation: int, out: Path,
) -> None:
    """Write out/<stem>.png: black where every instance of ``prompt`` is, white elsewhere and inside masks/."""
    import numpy as np

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
    colmap = require_tool("colmap")
    images = scene.require_images()
    rig = scene.is_pano()
    positions = {} if rig else geo.image_positions(images)  # drone photos and DJI video frames carry GPS
    if matcher == "auto":
        # With GPS (drone photos and frames), COLMAP pairs each image with its neighbours on the ground,
        # which also links the strips of a mapping flight. Exhaustive matching is quadratic; it takes hours
        # on hundreds of images.
        matcher = "spatial" if len(positions) * 2 > len(images) else "exhaustive" if len(images) <= 200 else "sequential"
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
    poses = None if rig else scene.arkit_poses()
    if poses:  # ARKit's focal length is a better start than COLMAP's guess; bundle adjustment refines it
        extract += ["--ImageReader.camera_model", "SIMPLE_RADIAL", "--ImageReader.camera_params", arkit.camera_params(poses)]
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
    if positions:
        # A nadir mapping flight is close to a plane seen from one side; on one, the incremental mapper placed
        # 41 of 81 frames in 10 min and the global mapper (GLOMAP) all 81 in 1.5 min, 0.3 m RMS from their GPS.
        mapper[1] = "global_mapper"
    else:
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
    model = final / "0"

    def transform(text: str) -> None:
        """Move the model by a similarity, "scale qw qx qy qz tx ty tz" (x' = s·R·x + t)."""
        (staged / "transform.txt").write_text(text)
        moved = staged / "moved"
        moved.mkdir()
        run_tool(
            [colmap, "model_transformer", "--input_path", str(model), "--output_path", str(moved),
             "--transform_path", str(staged / "transform.txt")],
            cwd=staged,
        )
        replace(moved, model)

    if rig:  # COLMAP's frame is arbitrary (for fisheyes, a lens's); turn the model so gravity is +y
        transform(pano.level_transform(model / "images.bin", lenses))
    if poses:  # an iPhone LiDAR scan: move the model onto ARKit's cameras (metres, gravity +y, the LiDAR points' frame)
        text, note = arkit.snap_transform(model, scene.arkit, poses)
        transform(text)
        error = arkit.snap_error(model / "images.bin", poses)
        print(f"Snapped to ARKit: {note}; camera centres within {error * 100:.1f} cm RMS of ARKit's")
        if error > 0.1:
            print("warning: COLMAP and ARKit disagree by more than 10 cm; check the model before training", flush=True)
    elif len(positions) >= 3:  # drone GPS: move the model into a map projection (geo.py)
        text, record, note = geo.georeference(model / "images.bin", positions)
        transform(text)
        (final / GEO_FILE).write_text(json.dumps(record, indent=1))
        print(f"Georeferenced: {note}")
    if no_masks:
        (final / NO_MASKS).touch()
    replace(database, scene.database)
    replace(final, scene.sparse)
    remove(scene.dense)  # built from the old model
    shutil.rmtree(staged, ignore_errors=True)
    registered = registered_images(scene.sparse / "0")
    print(f"Model: {registered}/{len(images)} images registered ({matcher} matching)")
    if registered * 2 < len(images):
        # Training on a fraction of the cameras wastes the run; stop `gausser run` here.
        raise GausserError(
            f"COLMAP placed only {registered} of {len(images)} images. Use more photos with more overlap "
            "(from a video: `gausser frames VIDEO --spacing high`), "
            "or run `gausser train` to train on these anyway"
        )


def start_points(scene: Scene, cloud: Path | None, init: str, crop: cropping.Crop | None) -> tuple[Any, Any, str]:
    """Brush's starting points: the dense or LiDAR `cloud` (thinned to DENSE_INIT_POINTS) or COLMAP's, inside the crop."""
    import numpy as np

    if cloud:
        print(f"Reading {cloud.relative_to(scene.root)} ({ply.count(cloud):,} points)", flush=True)
        read = ply.read_dense_cloud if init == "dense" else ply.read_points
        xyz, rgb = read(cloud, DENSE_INIT_POINTS if crop is None else sys.maxsize)  # thinned after cropping
        name = str(cloud.relative_to(scene.root))
    else:
        points = colmap_model.read_points(scene.require_model())
        xyz, rgb, name = points.xyz, points.rgb, "the COLMAP model"
    if crop is None:
        return xyz, rgb, name
    inside = crop.inside(xyz)
    print(f"Crop keeps {int(inside.sum()):,} of {len(xyz):,} starting points")
    if not inside.any():
        raise GausserError("The crop box holds none of the starting points; check it in `gausser monitor`")
    xyz, rgb = xyz[inside], rgb[inside]
    if cloud and len(xyz) > DENSE_INIT_POINTS:
        pick = np.sort(np.random.default_rng(0).choice(len(xyz), DENSE_INIT_POINTS, replace=False))
        xyz, rgb = xyz[pick], rgb[pick]
    return xyz, rgb, name


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
    init: str = "auto",
    adaptive: bool = False,
    max_iterations: int = 60000,
) -> Path:
    """Train with Brush into a new runs/brush/<time>/ folder; `extra` goes to Brush last.

    `no_masks` leaves masks/ out of Brush's folder, so it trains on whole images.
    `exclude_masks` gives Brush masks-exclude/ as its masks (a context run without transient objects).
    `init` picks Brush's starting points: dense = thinned dense/ cloud, lidar = an iPhone scan's LiDAR points
    (arkit/), sparse = COLMAP's points; auto = dense, else lidar, when there is one and Brush gets no masks.
    With a crop (sparse/crop.json), Brush starts only from points in the box and the exports keep only
    splats in it. `adaptive` continues 5000 steps at a time (see `extend`) up to `max_iterations`.
    """
    brush = require_tool("brush")
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
    clouds = {"dense": scene.dense_cloud(), "lidar": scene.lidar_cloud()}
    if init == "dense" and clouds["dense"] is None:
        raise GausserError("No dense point cloud; run `gausser dense` first")
    if init == "lidar" and clouds["lidar"] is None:
        raise GausserError("No LiDAR points; they come from `gausser frames` on an iPhone LiDAR scan folder")
    # The clouds ignore masks: a masked run starting from one would keep background splats (see composite).
    masked = brush_masks is not None and brush_masks.is_dir()
    if init == "auto":
        init = next((kind for kind, cloud in clouds.items() if cloud and not masked), "sparse")
    crop = cropping.load(scene)
    if adaptive and not eval_split:
        eval_split = ADAPTIVE_EVAL_SPLIT
        print(f"--adaptive judges by held-out PSNR: holding out every {eval_split}th image")
    cloud = clouds.get(init)
    own_points = cloud is not None or crop is not None
    for name, source in (("images", scene.images), ("sparse", scene.sparse), ("masks", brush_masks)):
        if name == "sparse" and own_points:
            continue
        if source and source.is_dir():
            (dataset / name).symlink_to(source)
    if own_points:
        # Brush starts from a COLMAP model's points3D.bin, so give it the cameras with the chosen points.
        model = dataset / "sparse" / "0"
        model.mkdir(parents=True)
        for path in scene.require_model().iterdir():
            if path.name != "points3D.bin":
                (model / path.name).symlink_to(path)
        xyz, rgb, source_name = start_points(scene, cloud, init, crop)
        colmap_model.write_points3d(model / "points3D.bin", xyz, rgb)
        print(f"Starting from {len(xyz):,} points of {source_name}")
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
    env = {"RUST_LOG": os.environ.get("RUST_LOG", "brush_cli=info")}
    run_tool(command, cwd=scene.root, env=env)
    if not list(staged.glob("exports/*.ply")):
        raise GausserError(f"Brush finished without a .ply export; output kept in {staged}")
    if adaptive:
        extend(scene, command, staged / "exports", iterations, max_iterations, env)
    remove(dataset)
    if crop is not None:
        for export in sorted(staged.glob("exports/*.ply")):
            kept, total = cropping.crop_splats(export, crop)
            print(f"Cropped {export.name}: kept {kept:,} of {total:,} splats")
    # `gausser composite` finds the statue (masked) and context (unmasked, excluded) runs by this marker.
    marker = "excluded" if exclude_masks else "unmasked" if no_masks or not scene.masks.is_dir() else "masked"
    (staged / marker).touch()
    (staged / f"init-{init}").touch()
    run = scene.new_run("brush")
    replace(staged, run)
    print(f"Exports in {run / 'exports'}")
    return run


def psnr(render: Path, photo: Path) -> float:
    """PSNR of a render against its photo, scaled to the render's size, over the whole image."""
    import numpy as np

    with Image.open(render) as r:
        a = np.asarray(r.convert("RGB"), dtype=np.float32) / 255
    with Image.open(photo) as p:
        p.draft("RGB", (a.shape[1], a.shape[0]))
        b = np.asarray(p.convert("RGB").resize((a.shape[1], a.shape[0])), dtype=np.float32) / 255
    mse = float(((a - b) ** 2).mean())
    return round(10 * math.log10(1 / mse), 2) if mse > 0 else 99.0


def eval_psnr(scene: Scene, exports: Path, step: int) -> float:
    """Mean PSNR of Brush's renders of the held-out images at `step` (exports/eval_<step>/images/<stem>.png)."""
    photos = {image.stem: image for image in scene.image_files()}
    scores = [psnr(r, photos[r.stem]) for r in (exports / f"eval_{step}" / "images").glob("*.png") if r.stem in photos]
    if not scores:
        raise GausserError(f"Brush saved no held-out renders at step {step}")
    return sum(scores) / len(scores)


def extend(scene: Scene, command: list[str], exports: Path, step: int, max_iterations: int, env: dict[str, str]) -> None:
    """Continue a Brush run 5000 steps at a time while its held-out PSNR still gains 0.1 dB.

    Brush has no checkpoints: a continuation starts from the last export (an `init.ply` in its dataset
    folder overrides every other start), with `--start-iter` so growth stays off and exports and evals
    keep their step numbers. Its mean learning rate restarts where the first run's schedule ended.
    A continuation that made the held-out PSNR worse leaves no export.
    """
    dataset = Path(command[1])
    score = eval_psnr(scene, exports, step)
    while step + EXTEND_STEPS <= max_iterations:
        print(f"\nHeld-out PSNR at step {step}: {score:.2f} dB; training {EXTEND_STEPS} more steps", flush=True)
        init = dataset / "init.ply"
        init.symlink_to(exports / f"export_{step}.ply")
        more = [*command[:2], "--start-iter", str(step), *command[2:]]
        more[more.index("--total-train-iters") + 1] = str(step + EXTEND_STEPS)
        if "--lr-mean" not in more:
            more += ["--lr-mean", "2e-7", "--lr-mean-end", "1e-7"]  # Brush's default end of schedule, then half
        run_tool(more, cwd=scene.root, env=env)
        init.unlink()
        new = eval_psnr(scene, exports, step + EXTEND_STEPS)
        print(f"Held-out PSNR at step {step + EXTEND_STEPS}: {new:.2f} dB ({new - score:+.2f})", flush=True)
        if new < score:
            remove(exports / f"export_{step + EXTEND_STEPS}.ply")
            print(f"Worse: keeping export_{step}.ply as the final export")
            return
        step, gain, score = step + EXTEND_STEPS, new - score, new
        if gain < MIN_GAIN_DB:
            print(f"Gained less than {MIN_GAIN_DB} dB: stopping at step {step}")
            return
    print(f"Reached --max-iterations {max_iterations}")


def dense(scene: Scene, resolution_level: int = 1) -> None:
    """Make dense/: undistorted images and an OpenMVS dense point cloud of the COLMAP model.

    OpenMVS estimates a depth map per image (patch match, on Metal on a Mac) and fuses the ones that agree.
    `resolution_level` halves the images that many times first; 1 is about 4x faster than 0.
    """
    colmap = require_tool("colmap")
    interface = require_tool("InterfaceCOLMAP")
    densify = require_tool("DensifyPointCloud")
    model = scene.require_model()
    staged = scene.staging("dense")
    # OpenMVS needs ideal pinhole cameras; this writes images/ and sparse/ without lens distortion.
    run_tool(
        [colmap, "image_undistorter", "--image_path", str(scene.images), "--input_path", str(model),
         "--output_path", str(staged), "--output_type", "COLMAP"],
        cwd=staged,
    )
    run_tool([interface, "-i", ".", "-o", "scene.mvs", "--image-folder", "images/"], cwd=staged)
    run_tool(
        [densify, "scene.mvs", "--resolution-level", str(resolution_level), "--remove-dmaps", "1"],
        cwd=staged,
    )
    if not (staged / "scene_dense.ply").is_file():
        raise GausserError(f"DensifyPointCloud wrote no scene_dense.ply; output kept in {staged}")
    for path in [*staged.glob("*.log"), *staged.glob("*.sh"), staged / "stereo"]:
        remove(path)
    replace(staged, scene.dense)
    print(f"Dense point cloud: {ply.count(scene.dense / 'scene_dense.ply'):,} points in {scene.dense}")


def mesh(scene: Scene, *, faces: int | None = None, refine: bool = True, texture_size: int = 8192) -> Path:
    """Mesh and texture dense/ with OpenMVS into a new runs/mesh/<time>/mesh.glb.

    ReconstructMesh carves the surface out of the space the cameras see through, RefineMesh moves it
    until the photos agree, and TextureMesh picks a photo per triangle. `faces` decimates the mesh
    before texturing; `refine=False` skips the slowest stage (CPU only).
    """
    reconstruct = require_tool("ReconstructMesh")
    refiner = require_tool("RefineMesh")
    texture = require_tool("TextureMesh")
    if scene.dense_cloud() is None:
        raise GausserError("No dense point cloud; run `gausser dense` first")
    staged = scene.staging("mesh")
    project = str(scene.dense / "scene_dense.mvs")
    # OpenMVS finds the images relative to the project, so it runs in dense/ and writes into staging.
    run_tool([reconstruct, project, "-p", "scene_dense.ply", "-o", str(staged / "mesh.mvs")], cwd=scene.dense)
    surface = staged / "mesh.ply"
    crop = cropping.load(scene)
    if crop is not None:  # before refining: RefineMesh and TextureMesh then only work on the subject
        kept, total = cropping.crop_mesh(surface, crop)
        print(f"Crop keeps {kept:,} of {total:,} faces")
        if not kept:
            raise GausserError("The crop box holds none of the mesh; check it in `gausser monitor`")
    if refine:
        run_tool(
            [refiner, project, "-m", str(surface), "-o", str(staged / "refined.mvs"), "--resolution-level", "1"],
            cwd=scene.dense,
        )
        surface = staged / "refined.ply"
    command = [
        texture, project, "-m", str(surface), "-o", str(staged / "mesh-textured.mvs"),
        "--export-type", "glb", "--max-texture-size", str(texture_size),
    ]
    count = ply.count(surface, "face")
    if faces and faces < count:
        command += ["--decimate", f"{faces / count:.6f}"]
    run_tool(command, cwd=scene.dense)
    textured = staged / "mesh-textured.glb"
    if not textured.is_file():
        raise GausserError(f"TextureMesh wrote no mesh; output kept in {staged}")
    textured.rename(staged / "mesh.glb")  # its textures stay mesh-textured_diffuse*.png, as the .glb names them
    for path in [*staged.glob("*.mvs"), *staged.glob("*.log"), *scene.dense.glob("*.log")]:  # OpenMVS logs into its cwd
        remove(path)
    run = scene.new_run("mesh")
    replace(staged, run)
    print(f"Mesh in {run / 'mesh.glb'}")
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
    tool = require_tool("splat-transform")
    source = scene.splat(source)
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
    colmap = require_tool("colmap")
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
