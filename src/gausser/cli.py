"""`gausser`: photos or video in, Gaussian splats out."""

from __future__ import annotations

import argparse
import importlib.util
import shlex
import shutil
import sys
from pathlib import Path

from . import steps
from .scene import GausserError, Scene, registered_images

STEPS = ("frames", "mask", "solve", "train", "lod")


def _add_mask_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("mask (SAM3)")
    group.add_argument("--threshold", type=float, default=0.5, help="minimum SAM3 score (default 0.5; try 0.3 for low hit rates)")
    group.add_argument("--all-instances", action="store_true", help="keep every matching instance, not only the best")
    group.add_argument("--dilation", type=int, default=0, help="grow Brush masks by N pixels")
    group.add_argument("--feather", type=int, default=0, help="blur Brush mask edges by N pixels")
    group.add_argument("--device", default="auto", help="auto, mps, cuda or cpu")
    group.add_argument("--sam-checkpoint", help="local SAM3 checkpoint instead of Hugging Face")


def _add_solve_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("solve (COLMAP)")
    group.add_argument(
        "--matcher", choices=("auto", "exhaustive", "sequential"), default="auto",
        help="auto = exhaustive up to 200 images, else sequential (video frames)",
    )


def _add_train_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("train")
    group.add_argument("--trainer", choices=("brush", "msplat"), default="brush")
    group.add_argument("--iterations", type=int, default=30000)
    group.add_argument("--export-every", type=int, default=5000, help="Brush: write a .ply every N steps")
    group.add_argument("--max-splats", type=int, default=1_500_000, help="Brush: splat ceiling (default 1.5M)")
    group.add_argument("--max-resolution", type=int, help="Brush: cap the longest image side")
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
        epilog="Run `gausser` with no arguments for a guided menu.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    run = commands.add_parser(
        "run", parents=[common], help="run every step that has not been done yet",
        description="Run the pipeline unattended. Steps whose output already exists are skipped.",
    )
    run.add_argument("--video", type=Path, help="extract frames from this video first")
    run.add_argument("--prompt", help="SAM3 subject prompt; without it nothing is masked")
    run.add_argument("--lod", action="store_true", help="also build a streamed LOD set")
    run.add_argument("--from", dest="from_step", choices=STEPS, help="redo this step and every step after it")
    for add in (_add_mask_options, _add_solve_options, _add_train_options, _add_lod_options):
        add(run)

    frames = commands.add_parser(
        "frames", parents=[common], help="select sharp video frames into images/",
        description="Select sharp frames into images/. Extra arguments after -- go to sharp-frames.",
    )
    frames.add_argument("source", type=Path, help="video file, or folder of videos or images")
    frames.add_argument("extra", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    mask = commands.add_parser("mask", parents=[common], help="mask the subject with SAM3")
    mask.add_argument("prompt", help='what to keep, e.g. "statue"')
    _add_mask_options(mask)

    solve = commands.add_parser("solve", parents=[common], help="find camera poses with COLMAP")
    _add_solve_options(solve)

    train = commands.add_parser("train", parents=[common], help="train splats with Brush or Msplat")
    _add_train_options(train)
    train.add_argument("--resume", metavar="RUN", help="Msplat: continue an interrupted run")

    lod = commands.add_parser("lod", parents=[common], help="build a streamed PlayCanvas LOD set")
    lod.add_argument("source", type=Path, nargs="?", help=".ply to convert (default: newest export)")
    _add_lod_options(lod)

    commands.add_parser("status", parents=[common], help="show scene progress and installed tools")
    return parser


def _levels(raw: str) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in raw.split(","))
    except ValueError as exc:
        raise GausserError(f"--levels must be comma-separated integers, got {raw!r}") from exc


def _mask(scene: Scene, prompt: str, args: argparse.Namespace) -> None:
    steps.mask(
        scene, prompt, threshold=args.threshold, all_instances=args.all_instances, dilation=args.dilation,
        feather=args.feather, device=args.device, checkpoint=args.sam_checkpoint,
    )


def _train(scene: Scene, args: argparse.Namespace, resume: str | None = None) -> None:
    if args.trainer == "msplat":
        from . import msplat

        export = msplat.train(
            scene, iterations=args.iterations, downscale_factor=args.downscale,
            checkpoint_every=args.checkpoint_every, resume=resume,
        )
        print(f"Export: {export}")
    else:
        steps.train_brush(
            scene, iterations=args.iterations, export_every=args.export_every,
            max_splats=args.max_splats, max_resolution=args.max_resolution,
        )


def _lod(scene: Scene, source: Path | None, args: argparse.Namespace) -> None:
    steps.lod(scene, source, chunk_count=args.chunk_count, chunk_extent=args.chunk_extent, percentages=_levels(args.levels))


def run_pipeline(scene: Scene, args: argparse.Namespace) -> None:
    """Run each step whose output is missing; after one step runs, every later step runs too."""
    redo_from = STEPS.index(args.from_step) if args.from_step else len(STEPS)

    def due(step: str, done: bool) -> bool:
        if STEPS.index(step) >= redo_from or not done:
            print(f"\n== {step} ==", flush=True)
            return True
        print(f"== {step}: done, skipping (--from {step} redoes it)")
        return False

    def ran(step: str) -> None:
        nonlocal redo_from
        redo_from = min(redo_from, STEPS.index(step) + 1)

    if args.video and due("frames", bool(scene.image_files())):
        steps.frames(scene, args.video, [])
        ran("frames")
    elif args.from_step == "frames":
        raise GausserError("--from frames needs --video")
    scene.require_images()
    if args.prompt and due("mask", scene.masks.is_dir()):
        _mask(scene, args.prompt, args)
        ran("mask")
    if due("solve", scene.model() is not None):
        steps.solve(scene, args.matcher)
        ran("solve")
    if due("train", bool(scene.exports())):
        _train(scene, args)
        ran("train")
    if args.lod and due("lod", bool(scene.lod_runs())):
        _lod(scene, None, args)


def status(scene: Scene) -> None:
    images = scene.image_files()
    model = scene.model()
    masks = sum(1 for _ in scene.masks.glob("*.png")) if scene.masks.is_dir() else 0
    print(f"Scene   {scene.root}")
    print(f"images  {len(images)}")
    print(f"masks   {masks or '-'}")
    print(f"model   {f'{registered_images(model)}/{len(images)} images registered' if model else '-'}")
    for export in scene.exports()[-5:]:
        print(f"export  {export.relative_to(scene.root)}")
    for run in scene.lod_runs()[-3:]:
        print(f"lod     {run.relative_to(scene.root)}")
    print("\nTools")
    tools = {
        "sharp-frames": steps.SHARP_FRAMES_HINT,
        "ffmpeg": "brew install ffmpeg",
        "colmap": steps.COLMAP_HINT,
        "brush": steps.BRUSH_HINT,
        "splat-transform": steps.SPLAT_TRANSFORM_HINT,
    }
    for name, hint in tools.items():
        path = shutil.which(name)
        print(f"  {'✓' if path else '✗'} {name:16}{path or hint}")
    for module, hint in (("sam3", "see README: SAM3"), ("msplat", "pip install -e '.[msplat]' (optional)")):
        found = importlib.util.find_spec(module) is not None
        print(f"  {'✓' if found else '✗'} {module:16}{'importable' if found else hint}")


def guided() -> list[str]:
    """Ask a few questions and return the equivalent command line."""
    import questionary

    scene = Scene(".")
    has_images = bool(scene.image_files())
    choice = questionary.select(
        "What do you want to do?",
        choices=[
            questionary.Choice("Run the whole pipeline", "run"),
            questionary.Choice("Extract frames from a video", "frames"),
            questionary.Choice("Mask the subject (SAM3)", "mask"),
            questionary.Choice("Solve cameras (COLMAP)", "solve"),
            questionary.Choice("Train splats", "train"),
            questionary.Choice("Build a streamed LOD set", "lod"),
            questionary.Choice("Show status", "status"),
        ],
    ).unsafe_ask()
    argv = [choice]
    if choice in ("run", "frames") and (choice == "frames" or not has_images):
        video = questionary.path("Video file:").unsafe_ask()
        argv += [video] if choice == "frames" else ["--video", video]
    if choice in ("run", "mask"):
        prompt = questionary.text("Subject to keep (blank = no masking):" if choice == "run" else "Subject to keep:").unsafe_ask()
        if prompt:
            argv += [prompt] if choice == "mask" else ["--prompt", prompt]
    if choice in ("run", "train"):
        trainer = questionary.select("Trainer:", choices=["brush", "msplat"]).unsafe_ask()
        iterations = questionary.text("Iterations:", default="30000").unsafe_ask()
        argv += ["--trainer", trainer, "--iterations", iterations]
    if choice == "run" and questionary.confirm("Also build a streamed LOD set?", default=False).unsafe_ask():
        argv.append("--lod")
    return argv


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    if not argv:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            parser.print_help()
            return 2
        try:
            argv = guided()
        except KeyboardInterrupt:
            return 130
        print("→ gausser " + shlex.join(argv))
    args = parser.parse_args(argv)
    try:
        scene = Scene(args.scene)
        if args.command == "run":
            run_pipeline(scene, args)
        elif args.command == "frames":
            extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
            steps.frames(scene, args.source, extra)
        elif args.command == "mask":
            _mask(scene, args.prompt, args)
        elif args.command == "solve":
            steps.solve(scene, args.matcher)
        elif args.command == "train":
            if args.resume:
                args.trainer = "msplat"  # only Msplat runs can be resumed
            _train(scene, args, args.resume)
        elif args.command == "lod":
            _lod(scene, args.source, args)
        else:
            status(scene)
    except GausserError as exc:
        print(f"gausser: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ngausser: interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
