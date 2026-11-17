"""`gausser`: photos or video in, Gaussian splats out."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import keyframes, mapping, pano, ply, steps
from .scene import TOOLS, GausserError, Scene, is_image, registered_images

STEPS = ("frames", "mask", "solve", "dense", "train", "lod", "mesh", "map")
COMMANDS = ("run", *STEPS, "crop", "subset", "composite", "align", "view", "orbit", "status", "monitor")
QUICK_START = """\
quick start:
  gausser clip.mp4              guided: makes clip/ next to the video and walks you through
  gausser ~/photos/statue       guided, on a folder of photos
  gausser                       guided, in the current scene folder

  cd statue                     a scene is a folder with photos in images/
  gausser run --prompt statue   unattended: mask → solve → train (skips what is up to date)
  gausser view                  open the result in SuperSplat

More: README.md in the gausser repository, or `gausser COMMAND --help`."""


def _add_mask_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("mask (SAM3)")
    group.add_argument("--threshold", type=float, default=0.5, help="minimum SAM3 score (default 0.5; try 0.3 for low hit rates)")
    group.add_argument("--all-instances", action="store_true", help="keep every matching instance, not only the best")
    group.add_argument("--dilation", type=int, default=0, help="grow Brush masks by N pixels; negative N shrinks them (edge pixels mixed with background)")
    group.add_argument("--feather", type=int, default=0, help="blur Brush mask edges by N pixels")
    group.add_argument("--device", default="auto", help="auto, mps, cuda or cpu")
    group.add_argument("--sam-checkpoint", help="local SAM3 checkpoint instead of Hugging Face")
    group.add_argument(
        "--miss", choices=("keep", "drop"), default="keep",
        help="when SAM3 finds nothing in an image: keep = train on the whole image (default); "
        "drop = Brush ignores the image, COLMAP still uses it",
    )


def _add_no_masks_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-masks", action="store_true",
        help="ignore existing masks: COLMAP sees whole images, Brush and Msplat train on whole images",
    )


def _add_solve_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("solve (COLMAP)")
    group.add_argument(
        "--matcher", choices=("auto", "exhaustive", "sequential", "spatial", "vocab_tree"), default="auto",
        help="auto = spatial (GPS neighbours) when most images have GPS, else exhaustive up to 200 images, "
        "else sequential with loop detection; "
        "vocab_tree suits large unordered photo sets",
    )


def _add_train_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("train")
    group.add_argument("--trainer", choices=("brush", "msplat"), default="brush")
    group.add_argument("--iterations", type=int, default=30000)
    group.add_argument("--export-every", type=int, default=5000, help="Brush: write a .ply every N steps")
    group.add_argument("--max-splats", type=int, default=1_500_000, help="Brush: splat ceiling (default 1.5M)")
    group.add_argument("--max-resolution", type=int, help="Brush: cap the longest image side")
    group.add_argument(
        "--eval-split", type=int, metavar="N",
        help="Brush: hold out every Nth image and report PSNR/SSIM on it every 1000 steps",
    )
    group.add_argument(
        "--adaptive", action="store_true",
        help=f"Brush: after --iterations, train {steps.EXTEND_STEPS} more steps at a time while held-out PSNR gains "
        f"{steps.MIN_GAIN_DB} dB (holds out every {steps.ADAPTIVE_EVAL_SPLIT}th image unless --eval-split is given)",
    )
    group.add_argument("--max-iterations", type=int, default=60000, help="Brush: --adaptive stops here (default 60000)")
    group.add_argument(
        "--init", choices=("auto", "dense", "lidar", "sparse"), default="auto",
        help="Brush: start from the dense cloud (dense/), an iPhone scan's LiDAR points (arkit/) or COLMAP's points; "
        "auto = dense, else lidar, when there is one and Brush gets no masks",
    )
    group.add_argument("--downscale", type=float, default=4.0, help="Msplat: load images at 1/N size (default 4)")
    group.add_argument("--checkpoint-every", type=int, default=1000, help="Msplat: checkpoint every N steps")


def _add_lod_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("lod (splat-transform)")
    group.add_argument("--chunk-count", type=int, default=512, help="splats per chunk, in thousands")
    group.add_argument("--chunk-extent", type=float, default=16.0, help="chunk size in metres")
    group.add_argument("--levels", default="100,50,25", help="percent of splats kept per level")


def _add_dense_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("dense (OpenMVS)")
    group.add_argument(
        "--resolution-level", type=int, default=1,
        help="halve the images N times before estimating depth (default 1; 0 = full size, about 4x slower)",
    )


def _add_mesh_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("mesh (OpenMVS)")
    group.add_argument("--faces", type=int, help="decimate to about N triangles before texturing (default: keep all)")
    group.add_argument("--no-refine", dest="refine", action="store_false", help="skip refining the surface against the photos (the slow, CPU-only stage)")
    group.add_argument("--texture-size", type=int, default=8192, help="largest texture image side (default 8192)")


def _add_map_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("map")
    group.add_argument("--resolution", type=float, default=0.03, metavar="M", help="orthophoto and DSM cell size in metres (default 0.03)")


def _add_frames_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("frames")
    group.add_argument(
        "--spacing", choices=list(keyframes.SPACINGS), default="medium",
        help="how far the camera moves between frames: low = fewer frames, high = more (default medium)",
    )
    group.add_argument("--min-frames", type=int, default=keyframes.MIN_FRAMES, help="use closer spacing if fewer frames result (default %(default)s)")
    group.add_argument("--max-frames", type=int, default=keyframes.MAX_FRAMES, help="use wider spacing if more frames result (default %(default)s)")
    group.add_argument("--every", type=int, metavar="N", help="instead of by motion: the sharpest of every N video frames")
    group.add_argument(
        "--overlap", type=int, metavar="PERCENT",
        help="mapping (straight-down drone video): a frame whenever the next one would share only PERCENT of the view, "
        "e.g. 80; no frame count limits. A DJI video's .SRT subtitle file gives the frames GPS",
    )


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-s", "--scene", type=Path, default=Path("."), help="scene folder (default: current folder)")
    # Commands that change the scene; they can open the web monitor.
    working = argparse.ArgumentParser(add_help=False, parents=[common])
    working.add_argument("--monitor", action="store_true", help="watch this command in the web monitor (`gausser monitor`)")

    parser = argparse.ArgumentParser(
        prog="gausser",
        description="Turn photos or video into Gaussian splats: frames → mask → solve → train → lod.",
        epilog=QUICK_START,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    run = commands.add_parser(
        "run", parents=[working], help="run every step that is missing or out of date",
        description="Run the pipeline unattended. Steps that are already up to date are skipped.",
    )
    run.add_argument("--video", type=Path, help="extract frames from this video first")
    run.add_argument(
        "--import", dest="import_photos", action="store_true",
        help="first move photos lying in the scene folder into images/",
    )
    run.add_argument("--prompt", help="SAM3 subject prompt; without it nothing is masked")
    run.add_argument("--dense", action="store_true", help="also build a dense point cloud before training; Brush then starts from it")
    run.add_argument("--lod", action="store_true", help="also build a streamed LOD set")
    run.add_argument("--mesh", action="store_true", help="also build a textured mesh (implies --dense)")
    run.add_argument("--map", action="store_true", help="also make an orthophoto and DSM (needs images with GPS)")
    run.add_argument("--from", dest="from_step", choices=STEPS, help="redo this step and every step after it")
    for add in (
        _add_frames_options, _add_mask_options, _add_solve_options, _add_dense_options, _add_train_options,
        _add_lod_options, _add_mesh_options, _add_map_options,
    ):
        add(run)
    _add_no_masks_option(run)
    run.set_defaults(ranges=None, exclude=False, exclude_masks=False, resume=None, extra=[])  # run_step reads them

    frames = commands.add_parser(
        "frames", parents=[working], help="select sharp video frames into images/",
        description="Select sharp frames into images/: one is picked wherever the camera has moved on, "
        "so a walk-around gives enough overlapping frames without tuning. A folder of photos is used whole.",
    )
    frames.add_argument("source", type=Path, help="video file, folder of videos or images, or an iPhone LiDAR scan folder (Stray Scanner app)")
    frames.add_argument(
        "--range", dest="ranges", action="append", type=_time_range, metavar="START-END",
        help="keep only this part of the video, in seconds or m:ss (repeatable), e.g. --range 1.5-32.5 --range 0:57-1:20",
    )
    _add_frames_options(frames)

    mask = commands.add_parser("mask", parents=[working], help="mask the subject with SAM3")
    mask.add_argument("prompt", help='what to keep, e.g. "statue"')
    _add_mask_options(mask)
    mask.add_argument(
        "--exclude", action="store_true",
        help="write masks-exclude/ instead: black = every instance of PROMPT grown by --dilation (try 8), white elsewhere "
        "and inside masks/; for `train --exclude-masks`. Leaves masks/ and COLMAP alone; --miss and --feather do not apply",
    )

    solve = commands.add_parser("solve", parents=[working], help="find camera poses with COLMAP")
    _add_solve_options(solve)
    _add_no_masks_option(solve)

    dense = commands.add_parser(
        "dense", parents=[working], help="build a dense point cloud with OpenMVS",
        description="Estimate a depth map per image with OpenMVS (on the GPU) and fuse them into dense/scene_dense.ply. "
        "`train` then starts Brush from it, and `mesh` builds a surface from it.",
    )
    _add_dense_options(dense)

    mesh = commands.add_parser(
        "mesh", parents=[working], help="build a textured mesh from the dense cloud with OpenMVS",
        description="Reconstruct, refine and texture a mesh from dense/ into runs/mesh/<time>/mesh.glb.",
    )
    _add_mesh_options(mesh)

    map_ = commands.add_parser(
        "map", parents=[working], help="orthophoto and DSM GeoTIFFs from the trained splat",
        description="Render the newest splat straight down into runs/map/<time>/: ortho.tif and dsm.tif in the scene's CRS "
        "and map.html, the orthophoto over a web map. Needs a model georeferenced from GPS (drone photos, or DJI video "
        "frames with a .SRT subtitle file).",
    )
    map_.add_argument("source", nargs="?", type=Path, help="splat .ply (default: the newest export)")
    _add_map_options(map_)

    train = commands.add_parser(
        "train", parents=[working], help="train splats with Brush or Msplat",
        description="Train splats. Extra arguments after -- go to Brush, e.g. -- --alpha-mode transparent.",
    )
    _add_train_options(train)
    _add_no_masks_option(train)
    train.add_argument(
        "--exclude-masks", action="store_true",
        help="Brush: train with masks-exclude/ as the masks (a context run without people; see `mask --exclude`)",
    )
    train.add_argument("--resume", metavar="RUN", help="Msplat: continue an interrupted run")
    train.add_argument("extra", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    lod = commands.add_parser("lod", parents=[working], help="build a streamed PlayCanvas LOD set")
    lod.add_argument("source", type=Path, nargs="?", help=".ply to convert (default: newest export)")
    _add_lod_options(lod)

    view = commands.add_parser("view", parents=[common], help="open a splat in a local SuperSplat editor")
    view.add_argument("source", type=Path, nargs="?", help=".ply to open (default: newest export)")

    orbit = commands.add_parser(
        "orbit", parents=[common], help="render a 360° orbit video of a splat",
        description="Render a 360° orbit video with SuperSplat's video renderer in a browser window. "
        "Needs the scene's COLMAP model (camera path) and a Brush export (vertical axis).",
    )
    orbit.add_argument("source", type=Path, nargs="?", help=".ply to render (default: newest export)")
    orbit.add_argument("--seconds", type=float, default=12.0, help="video length (default 12)")
    orbit.add_argument("--size", default="1920x1080", help="video size WIDTHxHEIGHT (default 1920x1080)")
    orbit.add_argument("--zoom", type=float, default=1.3, help="orbit radius times the median camera distance (default 1.3)")
    orbit.add_argument("--drop", type=float, default=0.35, help="aim this far below the centre, in splat units (default 0.35)")
    orbit.add_argument(
        "--background", metavar="BG",
        help="#rrggbb, `jutaka` (the public-art viewer's sky), or 'linear-gradient(to bottom, rgb(r,g,b) P%%, ...)' "
        "(default: SuperSplat's background); a gradient renders twice and blends with ffmpeg",
    )
    orbit.add_argument("--target", type=float, nargs=3, metavar=("X", "Y", "Z"), help="orbit this point (splat coordinates) instead of the cameras' centre")
    orbit.add_argument("--radius", type=float, help="orbit radius in splat units (overrides --zoom)")
    orbit.add_argument("--height", type=float, help="camera height above the target in splat units (default: the cameras' median)")
    orbit.add_argument("-o", "--output", type=Path, help="write the video here instead of runs/orbit/<time>/orbit.mp4")

    composite = commands.add_parser(
        "composite", parents=[common], help="put a masked statue splat into its unmasked context splat",
        description="Merge two splats trained on one COLMAP model: keep the statue splats that lie inside the masks, "
        "carve the statue's volume out of the context, append the statue. Needs sparse/0 and masks/. "
        "Defaults: the newest masked and the newest --no-masks Brush run.",
    )
    composite.add_argument("--statue", type=Path, help=".ply of the masked run (default: newest masked run)")
    composite.add_argument("--context", type=Path, help=".ply of the --no-masks run (default: newest unmasked run)")
    composite.add_argument("--radius", type=float, help="also drop context farther than N statue heights from the statue")
    composite.add_argument("--column", type=float, help="keep only a vertical column of radius N statue heights around the statue (e.g. 0.35 for statue + plinth)")
    composite.add_argument("--floor", type=float, default=0.3, help="with --column: reach N statue heights below the statue's feet (default 0.3)")
    composite.add_argument("--slack", type=int, default=2, help="grow the masks by N px (at 1/4 size) when filtering the statue (default 2)")
    composite.add_argument("--grow", type=int, default=2, help="carve N voxels beyond the statue's hull (default 2)")
    composite.add_argument("--cells", type=int, default=256, help="carve voxels along the statue's longest side (default 256)")
    composite.add_argument("-o", "--output", type=Path, help="write here instead of runs/composite/<time>/composite.ply")

    align = commands.add_parser(
        "align", help="move a splat into the coordinate frame of a published one",
        description="Estimate the similarity (scale, rotation, translation) that maps MOVING's opaque splat centres onto "
        "the --to splat(s), and write MOVING moved by it, so camera presets and geometry made for the published splat fit. "
        "REFERENCE is anything splat-transform reads (.ply, .sog, LOD child meta.json; several are concatenated) "
        "and may be a cropped or cleaned part of MOVING. Needs no scene.",
    )
    align.add_argument("moving", type=Path, metavar="MOVING", help=".ply to move (a Brush export)")
    align.add_argument("--to", dest="reference", type=Path, nargs="+", required=True, metavar="REFERENCE", help="published splat(s)")
    align.add_argument("--no-scale", dest="scale", action="store_false", help="estimate rotation and translation only")
    align.add_argument("-o", "--output", type=Path, required=True, help="aligned .ply to write")

    subset = commands.add_parser(
        "subset", parents=[common], help="make a new scene from the images of a solved scene that match name patterns",
        description="Copy the images (and their masks) whose file names match PATTERN into a new scene, with the "
        "COLMAP model filtered to them. Splats trained on different subsets share one coordinate frame, "
        "so `gausser composite` can merge them.",
    )
    subset.add_argument("patterns", nargs="+", metavar="PATTERN", help="glob on image file names, e.g. 'frame_*'")
    subset.add_argument("-o", "--output", type=Path, required=True, help="new scene folder (must not have images/)")

    crop = commands.add_parser(
        "crop", parents=[common], help="keep training and meshing to a box around the subject",
        description="Set the box `train` and `mesh` keep to (sparse/crop.json): Brush starts only from points in it, "
        "its exports keep only splats in it, and the mesh is cut to it before refining. The box is in the model's "
        "level frame (up = +y); draw it in `gausser monitor` (3D tab, crop) and paste the command it shows. "
        "Write negative numbers as --box=-1,0,-1,1,2,1. "
        "A new crop makes train and mesh out of date; a new solve drops it. Without options: show the crop.",
    )
    crop.add_argument("--box", type=_floats(6), metavar="x,y,z,X,Y,Z", help="minimum and maximum corner, level frame")
    crop.add_argument("--up", type=_floats(3), metavar="x,y,z", help="the model's up the box was drawn with (default: computed)")
    crop.add_argument("--clear", action="store_true", help="remove the crop")

    commands.add_parser("status", parents=[common], help="show scene progress and installed tools")

    monitor = commands.add_parser(
        "monitor", parents=[common], help="watch the scene in a web page: steps, logs, COLMAP cameras, splats",
        description="Open the scene in the read-only web monitor on 127.0.0.1, starting the monitor unless it runs. "
        "One monitor serves every scene gausser has worked on; it shows a running command live and works as a viewer "
        "afterwards. `--monitor` on run, frames, mask, solve, train and lod starts it in the background.",
    )
    monitor.add_argument("--all", action="store_true", help="open the list of all scenes")
    monitor.add_argument("--port", type=int, default=8737, help="port of a new monitor (default 8737)")
    monitor.add_argument("--no-open", action="store_true", help="do not open a browser")
    monitor.add_argument("--quit-when-idle", action="store_true", help=argparse.SUPPRESS)
    return parser


def _floats(count: int) -> Callable[[str], list[float]]:
    def parse(text: str) -> list[float]:
        values = [float(v) for v in text.split(",")]
        if len(values) != count:
            raise argparse.ArgumentTypeError(f"expected {count} comma-separated numbers")
        return values

    return parse


def _levels(raw: str) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in raw.split(","))
    except ValueError as exc:
        raise GausserError(f"--levels must be comma-separated integers, got {raw!r}") from exc


def _check(name: str, args: argparse.Namespace) -> None:
    """Refuse option combinations a step cannot honour, before it is recorded as started."""
    if name == "frames":
        if args.min_frames > args.max_frames or (args.every is not None and args.every < 1):
            raise GausserError("--min-frames must not exceed --max-frames, and --every must be at least 1")
        if args.overlap is not None and not 30 <= args.overlap <= 95:
            raise GausserError("--overlap is a percentage from 30 to 95")
        if args.overlap is not None and args.every is not None:
            raise GausserError("--overlap and --every cannot be combined")
    elif name == "train":
        if args.resume:
            args.trainer = "msplat"  # only Msplat runs can be resumed
        if args.extra[:1] == ["--"]:
            args.extra = args.extra[1:]
        if args.extra and args.trainer != "brush":
            raise GausserError("Arguments after -- go to Brush; Msplat takes none")
        if args.exclude_masks and (args.trainer != "brush" or args.no_masks):
            raise GausserError("--exclude-masks is for Brush and cannot be combined with --no-masks")
        if args.adaptive and args.trainer != "brush":
            raise GausserError("--adaptive is for Brush")


def run_step(scene: Scene, name: str, args: argparse.Namespace, source: Path | None = None) -> None:
    """Run one step inside `scene.step()` (history and log); a run folder it makes goes into the end record."""
    _check(name, args)
    info: dict[str, Any] = {}  # what the history records about the step
    if name == "frames":
        info = {"source": str(source)}
    elif name == "mask":
        info = {"prompt": args.prompt, "exclude": args.exclude}
    elif name == "solve":
        info = {"matcher": args.matcher}
    elif name == "train":
        info = {"trainer": args.trainer, "iterations": args.iterations}
    with scene.step(name, **info) as made:
        run = None
        if name == "frames":
            assert source is not None
            steps.frames(
                scene, source, args.ranges, spacing=args.spacing, min_frames=args.min_frames, max_frames=args.max_frames,
                every=args.every, overlap=args.overlap / 100 if args.overlap is not None else None,
            )
        elif name == "mask":
            steps.mask(
                scene, args.prompt, threshold=args.threshold, all_instances=args.all_instances, dilation=args.dilation,
                feather=args.feather, device=args.device, checkpoint=args.sam_checkpoint, miss=args.miss, exclude=args.exclude,
            )
        elif name == "solve":
            steps.solve(scene, args.matcher, args.no_masks)
        elif name == "dense":
            steps.dense(scene, args.resolution_level)
        elif name == "train":
            run = _train(scene, args)
        elif name == "lod":
            run = steps.lod(
                scene, source, chunk_count=args.chunk_count, chunk_extent=args.chunk_extent, percentages=_levels(args.levels),
            )
        elif name == "mesh":
            run = steps.mesh(scene, faces=args.faces, refine=args.refine, texture_size=args.texture_size)
        elif name == "map":
            run = mapping.make(scene, source, args.resolution)
        if run is not None:
            made["run"] = str(run.relative_to(scene.root))


def _train(scene: Scene, args: argparse.Namespace) -> Path:
    """Train and return the run folder."""
    if args.trainer == "msplat":
        from . import msplat

        export = msplat.train(
            scene, iterations=args.iterations, downscale_factor=args.downscale,
            checkpoint_every=args.checkpoint_every, resume=args.resume, no_masks=args.no_masks,
        )
        print(f"Export: {export}")
        return export.parent.parent
    return steps.train_brush(
        scene, iterations=args.iterations, export_every=args.export_every,
        max_splats=args.max_splats, max_resolution=args.max_resolution, eval_split=args.eval_split,
        extra=args.extra, no_masks=args.no_masks, exclude_masks=args.exclude_masks,
        init=args.init, adaptive=args.adaptive, max_iterations=args.max_iterations,
    )


def run_pipeline(scene: Scene, args: argparse.Namespace) -> None:
    """Run each step whose output is missing or older than its input; once one runs, all later ones do."""
    if args.from_step == "frames" and not args.video:
        raise GausserError("--from frames needs --video")
    if args.from_step == "mask" and not args.prompt:
        raise GausserError("--from mask needs --prompt")
    if args.from_step in ("dense", "lod", "mesh", "map"):
        setattr(args, args.from_step, True)
    if args.mesh:
        args.dense = True
    redo_from = STEPS.index(args.from_step) if args.from_step else len(STEPS)

    def due(step: str, done: bool) -> bool:
        if STEPS.index(step) >= redo_from or not done:
            print(f"\n== {step} ==", flush=True)
            return True
        print(f"== {step}: up to date, skipping (--from {step} redoes it)")
        return False

    def ran(step: str) -> None:
        nonlocal redo_from
        redo_from = min(redo_from, STEPS.index(step) + 1)

    if args.import_photos:
        scene.import_loose_photos()
    if args.video and due("frames", bool(scene.image_files())):
        run_step(scene, "frames", args, args.video)
        ran("frames")
    images = scene.require_images()
    if args.prompt and due("mask", scene.masks_done(images)):
        run_step(scene, "mask", args)
        ran("mask")
    if due("solve", scene.solve_done(images)):
        run_step(scene, "solve", args)
        ran("solve")
    for name, wanted, done in (
        ("dense", args.dense, scene.dense_done),
        ("train", True, scene.train_done),
        ("lod", args.lod, scene.lod_done),
        ("mesh", args.mesh, scene.mesh_done),
        ("map", args.map, scene.map_done),
    ):
        if wanted and due(name, done()):
            run_step(scene, name, args)
            ran(name)


def status(scene: Scene) -> None:
    images = scene.image_files()
    model = scene.model()
    masks = sum(1 for _ in scene.masks.glob("*.png")) if scene.masks.is_dir() else 0
    stale = "  (out of date)"
    print(f"Scene   {scene.root}")
    print(f"images  {len(images)}")
    if scene.is_pano():
        panos = sum(1 for path in scene.panos.iterdir() if is_image(path))
        rig = pano.lenses(scene.panos)
        kind = f"{rig['camera']} fisheyes" if rig else "equirectangular"
        print(f"        360°: {panos} frames in panos/ ({kind}), {pano.VIEWS} views each")
    if not images:
        loose = scene.loose_photos()
        if loose:
            print(f"        {len(loose)} photos are in the folder itself; `gausser` or `gausser run --import` moves them into images/")
        else:
            print("        put photos in images/, or run `gausser frames VIDEO` (or `gausser VIDEO` for the guided way)")
    print(f"masks   {masks or '-'}{stale if masks and not scene.masks_done(images) else ''}")
    if model:
        solved = f"{registered_images(model)}/{len(images)} images registered"
        print(f"model   {solved}{'' if scene.solve_done(images) else stale}")
        geo = scene.geo()
        if geo:
            print(f"        georeferenced: {geo['crs']}, cameras within {geo['rms_horizontal_m']} m of their GPS (RMS)")
    else:
        print("model   -")
    cloud = scene.dense_cloud()
    if cloud:
        print(f"dense   {ply.count(cloud):,} points{'' if scene.dense_done() else stale}")
    lidar = scene.lidar_cloud()
    if lidar:
        print(f"lidar   {ply.count(lidar):,} points, ARKit poses (iPhone scan)")
    exports = scene.exports()
    for export in exports[-5:]:
        print(f"export  {export.relative_to(scene.root)}")
    if exports and model and not scene.train_done():
        print(f"        newest export is older than the model or the crop{stale}")
    for run in scene.lod_runs()[-3:]:
        print(f"lod     {run.relative_to(scene.root)}")
    for run in scene.mesh_runs()[-3:]:
        print(f"mesh    {(run / 'mesh.glb').relative_to(scene.root)}")
    for run in scene.map_runs()[-3:]:
        print(f"map     {(run / 'map.html').relative_to(scene.root)}")
    if scene.tmp.is_dir():
        size = sum(path.stat().st_size for path in scene.tmp.rglob("*") if path.is_file() and not path.is_symlink())
        if size:
            print(f"tmp     {size / 1e9:.1f} GB of failed attempts in {scene.tmp} (delete when nothing is running)")
    print("\nTools")
    for name, hint in TOOLS.items():
        if name in ("InterfaceCOLMAP", "ReconstructMesh", "RefineMesh", "TextureMesh"):
            continue  # built and linked together with DensifyPointCloud
        if name == "DensifyPointCloud":
            hint = f"optional, for `dense` and `mesh`; {hint}"
        path = shutil.which(name)
        print(f"  {'✓' if path else '✗'} {name:18}{path or hint}")
    sam3 = importlib.util.find_spec("sam3")
    if sam3 is None or sam3.origin is None:
        print(f"  ✗ {'sam3':18}optional; run scripts/install-macos.sh")
    elif "_TRITON_AVAILABLE" not in (Path(sam3.origin).parent / "model" / "edt.py").read_text():
        print(f"  ✗ {'sam3':18}installed but missing patches/sam3-macos.patch; rerun scripts/install-macos.sh")
    else:
        print(f"  ✓ {'sam3':18}patched for macOS")
    from .orbit import has_playwright, node_dir

    found = has_playwright()
    print(f"  {'✓' if found else '✗'} {'playwright-core':18}{node_dir() if found else 'optional, for `gausser orbit`; run scripts/install-macos.sh'}")
    found = importlib.util.find_spec("msplat") is not None
    print(f"  {'✓' if found else '✗'} {'msplat':18}{'importable' if found else 'optional; pip install -e .[msplat]'}")
    from .view import supersplat_dist

    dist = supersplat_dist()
    ok = (dist / "index.html").is_file()
    print(f"  {'✓' if ok else '✗'} {'supersplat':18}{dist if ok else 'optional, for `gausser view`; run scripts/install-macos.sh'}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    if not argv or (len(argv) == 1 and _is_target(argv[0])):
        if not interactive():
            if argv:
                print("gausser: `gausser PATH` is the guided menu and needs a terminal; see `gausser --help`", file=sys.stderr)
            else:
                parser.print_help()
            return 2
        guided_argv = _guide(Path(argv[0]) if argv else Path("."))
        if not isinstance(guided_argv, list):
            return guided_argv
        argv = guided_argv
        print("→ gausser " + shlex.join(argv), flush=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "align":  # no scene: reads two splats and writes one file
            from .align import align

            align(args.moving, args.reference, args.output, with_scale=args.scale)
            return 0
        scene = Scene(args.scene)
        if args.command == "status":
            status(scene)
            return 0
        if args.command == "view":
            from .view import view

            view(scene, args.source)
            return 0
        if args.command == "orbit":  # like view: needs no lock; it only adds a new run folder
            from .orbit import orbit

            orbit(
                scene, args.source, seconds=args.seconds, size=args.size,
                zoom=args.zoom, drop=args.drop, out=args.output, background=args.background,
                target=args.target, radius=args.radius, height=args.height,
            )
            return 0
        if args.command == "composite":  # reads existing files and only adds a new output; needs no lock
            from .composite import composite

            composite(scene, args.statue, args.context, radius=args.radius, out=args.output, cells=args.cells, slack=args.slack, grow=args.grow, column=args.column, floor=args.floor)
            return 0
        if args.command == "subset":  # only reads the source scene; the new one is built in place
            steps.subset(scene, args.patterns, args.output)
            return 0
        if args.command == "monitor":
            from .monitor import monitor

            monitor(None if args.all else scene, port=args.port, open_browser=not args.no_open, quit_when_idle=args.quit_when_idle)
            return 0
        with scene.lock(argv):
            keep_awake()
            if getattr(args, "monitor", False):  # crop has no --monitor
                from .monitor import spawn

                spawn(scene)
            dispatch(scene, args)
    except GausserError as exc:
        print(f"gausser: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ngausser: interrupted", file=sys.stderr)
        return 130
    return 0


def keep_awake() -> None:
    """Keep macOS from sleeping while this command runs; a sleeping Mac pauses training.

    caffeinate -w exits by itself when this process ends. Closing the lid still
    sleeps a laptop without an external display.
    """
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _is_target(arg: str) -> bool:
    """`gausser PATH`: anything that is not a command or option and exists or looks like a path."""
    if arg in COMMANDS or arg.startswith("-"):
        return False
    return Path(arg).expanduser().exists() or "/" in arg or bool(Path(arg).suffix)


def _guide(target: Path) -> list[str] | int:
    """Run the guided menu; return the chosen command line, or an exit code."""
    from . import guide

    created: Path | None = None
    try:
        scene, video, new = guide.open_target(target)
        created = scene.root if new else None
        return guide.guided(scene, video)
    except GausserError as exc:
        print(f"gausser: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        if created and not any(created.iterdir()):
            created.rmdir()  # cancelled before anything was put in the new scene folder
        return 130


def _time_range(text: str) -> tuple[float, float]:
    def seconds(part: str) -> float:
        minutes, _, rest = part.rpartition(":")
        return float(minutes or 0) * 60 + float(rest)

    start, sep, end = text.partition("-")
    try:
        span = (seconds(start), seconds(end))
    except ValueError:
        span = (0.0, 0.0)
    if not sep or span[1] <= span[0]:
        raise argparse.ArgumentTypeError(f"{text!r} is not START-END, e.g. 1.5-32.5 or 0:57-1:20")
    return span


def _crop(scene: Scene, args: argparse.Namespace) -> None:
    from . import crop

    if args.clear:
        scene.crop.unlink(missing_ok=True)
        print("No crop")
    elif args.box:
        made = crop.save(scene, args.box, args.up)
        print(f"Crop set: {made.min} to {made.max} (level frame); train and mesh are now out of date")
    else:
        current = crop.load(scene)
        print(f"Crop: {current.min} to {current.max} (level frame, up {current.up})" if current else "No crop")


def dispatch(scene: Scene, args: argparse.Namespace) -> None:
    if args.command == "run":
        run_pipeline(scene, args)
    elif args.command == "crop":
        _crop(scene, args)
    else:
        run_step(scene, args.command, args, getattr(args, "source", None))


if __name__ == "__main__":
    sys.exit(main())
