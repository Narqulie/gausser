"""sharpframes — transactional Sharp Frames preprocessing for a Gausser scene."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from rich.text import Text

from gausser.interactive import (
    BARE_COMMAND_EPILOG,
    InteractiveUsageError,
    prepare_stage_arguments,
)
from gausser.manifest import fingerprint_file
from gausser.process import resolve_command, run_command
from gausser.publish import publish_stage
from gausser.scene import SceneError, ScenePaths
from gausser.scene_lock import SceneLockError, acquire_scene_lock
from gausser.ui import NEON_VIOLET, completion_panel, console, stage_header, status_text


def can_publish_into_scene(images: Path) -> bool:
    """Whether Sharp Frames may publish without replacing existing scene images."""
    return not images.exists() or not any(images.iterdir())


def sharp_frames_command(executable: str, source: Path, output: Path, extra: list[str]) -> list[str]:
    """Build the current upstream Sharp Frames invocation."""
    return [executable, str(source), str(output), "--force-overwrite", *extra]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="sharpframes",
        description="Select sharp frames into a Gausser scene's images/ directory.",
        epilog=BARE_COMMAND_EPILOG,
    )
    parser.add_argument("source", type=Path, help="Video file, video directory, or image directory")
    parser.add_argument("--scene", type=Path, default=Path.cwd(), help="Owning scene root (default: cwd)")
    parser.add_argument("--sharp-frames", default="sharp-frames", help="Sharp Frames executable")
    parser.add_argument("--force", action="store_true", help="Replace an existing images/ directory")
    parser.add_argument("--studio-settings", type=_parse_studio_settings, help=argparse.SUPPRESS)
    try:
        prepared_argv = prepare_stage_arguments(
            "sharpframes", argv, program="sharpframes"
        )
    except InteractiveUsageError as exc:
        parser.error(str(exc))
    if prepared_argv is None:
        return
    args, extra = parser.parse_known_args(prepared_argv)

    source = args.source.expanduser().resolve()
    scene = ScenePaths.from_root(args.scene)
    if not scene.root.is_dir():
        parser.error(f"scene directory does not exist: {scene.root}")
    if not source.exists():
        parser.error(f"source does not exist: {source}")
    if not resolve_command(args.sharp_frames):
        parser.error(f"{args.sharp_frames!r} is not on PATH; install with: pip install sharp-frames")

    try:
        with acquire_scene_lock(scene.root):
            _run(args, extra, parser, source, scene)
    except SceneLockError as exc:
        parser.error(str(exc))


def _run(
    args: argparse.Namespace,
    extra: list[str],
    parser: argparse.ArgumentParser,
    source: Path,
    scene: ScenePaths,
) -> None:
    if not can_publish_into_scene(scene.images) and not args.force:
        parser.error(f"{scene.images} already contains images; pass --force to replace them")

    attempt = scene.staging_attempt("sharpframes", scene.new_run_id())
    staged_images = attempt / "images"
    cmd = sharp_frames_command(args.sharp_frames, source, staged_images, extra)
    console.print(stage_header(
        "SHARP FRAMES",
        "Frame selection signal",
        (
            ("Source", str(source)),
            ("Scene", str(scene.root)),
            ("Publish", str(scene.images)),
            ("Mode", "replace" if args.force else "new generation"),
        ),
        accent=NEON_VIOLET,
    ))
    console.print()
    result = run_command(cmd, cwd=scene.root, on_line=lambda line: console.print(Text(line)))
    if not result.succeeded:
        if result.output_tail:
            console.print(status_text("Sharp Frames failed", "error"))
            console.print(Text("\n".join(result.output_tail[-20:])))
        sys.exit(1)

    try:
        staged_scene = ScenePaths.from_root(attempt)
        images = staged_scene.image_files()
    except SceneError as exc:
        console.print(status_text(f"Sharp Frames produced an invalid image set: {exc}", "error"))
        sys.exit(1)
    if not images:
        console.print(status_text(
            "Sharp Frames selected no images; existing images/ was left unchanged", "error"
        ))
        sys.exit(1)

    publish_stage(
        scene,
        "sharpframes",
        generation="images",
        replacements=((staged_images, scene.images),),
        data={
            "source": str(source),
            "source_fingerprint": fingerprint_file(source) if source.is_file() else None,
            "parameters": {"command": cmd[3:]},
            "studio_settings": args.studio_settings,
            "outputs": {"images": str(scene.relative_path(scene.images)), "count": len(images)},
        },
    )
    shutil.rmtree(attempt, ignore_errors=True)
    console.print(completion_panel(
        "Frame selection complete",
        (
            ("Published", f"{len(images)} images"),
            ("Destination", str(scene.images)),
            ("Elapsed", f"{result.duration_seconds:.1f}s"),
        ),
    ))


def _parse_studio_settings(raw: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("studio settings must be a JSON object") from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("studio settings must be a JSON object")
    return value
