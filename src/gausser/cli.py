"""`gausser`: photos or video in, Gaussian splats out."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from . import pano, steps
from .scene import GausserError, Scene, registered_images

STEPS = ("frames", "mask", "solve", "train", "lod")
COMMANDS = ("run", *STEPS, "subset", "composite", "align", "view", "orbit", "status")
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
        "--matcher", choices=("auto", "exhaustive", "sequential", "vocab_tree"), default="auto",
        help="auto = exhaustive up to 200 images, else sequential with loop detection; "
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
    group.add_argument("--downscale", type=float, default=4.0, help="Msplat: load images at 1/N size (default 4)")
    group.add_argument("--checkpoint-every", type=int, default=1000, help="Msplat: checkpoint every N steps")


def _add_lod_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("lod (splat-transform)")
    group.add_argument("--chunk-count", type=int, default=512, help="splats per chunk, in thousands")
    group.add_argument("--chunk-extent", type=float, default=16.0, help="chunk size in metres")
    group.add_argument("--levels", default="100,50,25", help="percent of splats kept per level")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-s", "--scene", type=Path, default=Path("."), help="scene folder (default: current folder)")

    parser = argparse.ArgumentParser(
        prog="gausser",
        description="Turn photos or video into Gaussian splats: frames → mask → solve → train → lod.",
        epilog=QUICK_START,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    run = commands.add_parser(
        "run", parents=[common], help="run every step that is missing or out of date",
        description="Run the pipeline unattended. Steps that are already up to date are skipped.",
    )
    run.add_argument("--video", type=Path, help="extract frames from this video first")
    run.add_argument(
        "--import", dest="import_photos", action="store_true",
        help="first move photos lying in the scene folder into images/",
    )
    run.add_argument("--prompt", help="SAM3 subject prompt; without it nothing is masked")
    run.add_argument("--lod", action="store_true", help="also build a streamed LOD set")
    run.add_argument("--from", dest="from_step", choices=STEPS, help="redo this step and every step after it")
    for add in (_add_mask_options, _add_solve_options, _add_train_options, _add_lod_options):
        add(run)
    _add_no_masks_option(run)

    frames = commands.add_parser(
        "frames", parents=[common], help="select sharp video frames into images/",
        description="Select sharp frames into images/. Extra arguments after -- go to sharp-frames.",
    )
    frames.add_argument("source", type=Path, help="video file, or folder of videos or images")
    frames.add_argument("extra", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    mask = commands.add_parser("mask", parents=[common], help="mask the subject with SAM3")
    mask.add_argument("prompt", help='what to keep, e.g. "statue"')
    _add_mask_options(mask)
    mask.add_argument(
        "--exclude", action="store_true",
        help="write masks-exclude/ instead: black = every instance of PROMPT grown by --dilation (try 8), white elsewhere "
        "and inside masks/; for `train --exclude-masks`. Leaves masks/ and COLMAP alone; --miss and --feather do not apply",
    )

    solve = commands.add_parser("solve", parents=[common], help="find camera poses with COLMAP")
    _add_solve_options(solve)
    _add_no_masks_option(solve)

    train = commands.add_parser(
        "train", parents=[common], help="train splats with Brush or Msplat",
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

    lod = commands.add_parser("lod", parents=[common], help="build a streamed PlayCanvas LOD set")
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

    commands.add_parser("status", parents=[common], help="show scene progress and installed tools")
    return parser


def _levels(raw: str) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in raw.split(","))
    except ValueError as exc:
        raise GausserError(f"--levels must be comma-separated integers, got {raw!r}") from exc


def _mask(scene: Scene, prompt: str, args: argparse.Namespace, exclude: bool = False) -> None:
    steps.mask(
        scene, prompt, threshold=args.threshold, all_instances=args.all_instances, dilation=args.dilation,
        feather=args.feather, device=args.device, checkpoint=args.sam_checkpoint, miss=args.miss, exclude=exclude,
    )


def _train(scene: Scene, args: argparse.Namespace, resume: str | None = None, extra: list[str] | None = None) -> None:
    if args.trainer == "msplat":
        from . import msplat

        export = msplat.train(
            scene, iterations=args.iterations, downscale_factor=args.downscale,
            checkpoint_every=args.checkpoint_every, resume=resume, no_masks=args.no_masks,
        )
        print(f"Export: {export}")
    else:
        steps.train_brush(
            scene, iterations=args.iterations, export_every=args.export_every,
            max_splats=args.max_splats, max_resolution=args.max_resolution, eval_split=args.eval_split,
            extra=extra or [], no_masks=args.no_masks, exclude_masks=getattr(args, "exclude_masks", False),
        )


def _lod(scene: Scene, source: Path | None, args: argparse.Namespace) -> None:
    steps.lod(scene, source, chunk_count=args.chunk_count, chunk_extent=args.chunk_extent, percentages=_levels(args.levels))


def run_pipeline(scene: Scene, args: argparse.Namespace) -> None:
    """Run each step whose output is missing or older than its input; once one runs, all later ones do."""
    if args.from_step == "frames" and not args.video:
        raise GausserError("--from frames needs --video")
    if args.from_step == "mask" and not args.prompt:
        raise GausserError("--from mask needs --prompt")
    if args.from_step == "lod":
        args.lod = True
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
        steps.frames(scene, args.video, [])
        ran("frames")
    images = scene.require_images()
    if args.prompt and due("mask", scene.masks_done(images)):
        _mask(scene, args.prompt, args)
        ran("mask")
    if due("solve", scene.solve_done(images)):
        steps.solve(scene, args.matcher, args.no_masks)
        ran("solve")
    if due("train", scene.train_done()):
        _train(scene, args)
        ran("train")
    if args.lod and due("lod", scene.lod_done()):
        _lod(scene, None, args)


def status(scene: Scene) -> None:
    images = scene.image_files()
    model = scene.model()
    masks = sum(1 for _ in scene.masks.glob("*.png")) if scene.masks.is_dir() else 0
    stale = "  (out of date)"
    print(f"Scene   {scene.root}")
    print(f"images  {len(images)}")
    if scene.is_pano():
        panos = sum(1 for _ in scene.panos.iterdir())
        print(f"        360°: {panos} frames in panos/, {pano.VIEWS} views each")
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
    else:
        print("model   -")
    exports = scene.exports()
    for export in exports[-5:]:
        print(f"export  {export.relative_to(scene.root)}")
    if exports and model and not scene.train_done():
        print(f"        newest export is older than the model{stale}")
    for run in scene.lod_runs()[-3:]:
        print(f"lod     {run.relative_to(scene.root)}")
    if scene.tmp.is_dir():
        size = sum(path.stat().st_size for path in scene.tmp.rglob("*") if path.is_file() and not path.is_symlink())
        if size:
            print(f"tmp     {size / 1e9:.1f} GB of failed attempts in {scene.tmp} (delete when nothing is running)")
    print("\nTools")
    tools = {
        "sharp-frames": steps.SHARP_FRAMES_HINT,
        "ffmpeg": "brew install ffmpeg",
        "colmap": steps.COLMAP_HINT,
        "brush": steps.BRUSH_HINT,
        "splat-transform": steps.SPLAT_TRANSFORM_HINT,
        "node": "brew install node",
    }
    for name, hint in tools.items():
        path = shutil.which(name)
        print(f"  {'✓' if path else '✗'} {name:16}{path or hint}")
    sam3 = importlib.util.find_spec("sam3")
    if sam3 is None or sam3.origin is None:
        print(f"  ✗ {'sam3':16}optional; run scripts/install-macos.sh")
    elif "_TRITON_AVAILABLE" not in (Path(sam3.origin).parent / "model" / "edt.py").read_text():
        print(f"  ✗ {'sam3':16}installed but missing patches/sam3-macos.patch; rerun scripts/install-macos.sh")
    else:
        print(f"  ✓ {'sam3':16}patched for macOS")
    from .orbit import has_playwright, node_dir

    found = has_playwright()
    print(f"  {'✓' if found else '✗'} {'playwright-core':16}{node_dir() if found else 'optional, for `gausser orbit`; run scripts/install-macos.sh'}")
    found = importlib.util.find_spec("msplat") is not None
    print(f"  {'✓' if found else '✗'} {'msplat':16}{'importable' if found else 'optional; pip install -e .[msplat]'}")
    from .view import supersplat_dist

    dist = supersplat_dist()
    ok = (dist / "index.html").is_file()
    print(f"  {'✓' if ok else '✗'} {'supersplat':16}{dist if ok else 'optional, for `gausser view`; run scripts/install-macos.sh'}")


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
        with scene.lock():
            keep_awake()
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


def dispatch(scene: Scene, args: argparse.Namespace) -> None:
    if args.command == "run":
        run_pipeline(scene, args)
    elif args.command == "frames":
        extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
        steps.frames(scene, args.source, extra)
    elif args.command == "mask":
        _mask(scene, args.prompt, args, args.exclude)
    elif args.command == "solve":
        steps.solve(scene, args.matcher, args.no_masks)
    elif args.command == "train":
        if args.resume:
            args.trainer = "msplat"  # only Msplat runs can be resumed
        extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
        if extra and args.trainer != "brush":
            raise GausserError("Arguments after -- go to Brush; Msplat takes none")
        if args.exclude_masks and (args.trainer != "brush" or args.no_masks):
            raise GausserError("--exclude-masks is for Brush and cannot be combined with --no-masks")
        _train(scene, args, args.resume, extra)
    elif args.command == "lod":
        _lod(scene, args.source, args)


if __name__ == "__main__":
    sys.exit(main())
