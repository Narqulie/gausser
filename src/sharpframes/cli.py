"""sharpframes — transactional Sharp Frames preprocessing for a Gausser scene."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from gausser.manifest import fingerprint_file, record_stage
from gausser.process import run_command
from gausser.publish import replace_directory
from gausser.scene import SceneError, ScenePaths
from gausser.ui import console


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="sharpframes",
        description="Select sharp frames into a Gausser scene's images/ directory.",
    )
    parser.add_argument("source", type=Path, help="Video file, video directory, or image directory")
    parser.add_argument("--scene", type=Path, default=Path.cwd(), help="Owning scene root (default: cwd)")
    parser.add_argument("--sharp-frames", default="sharp-frames", help="Sharp Frames executable")
    parser.add_argument("--force", action="store_true", help="Replace an existing images/ directory")
    args, extra = parser.parse_known_args(argv)

    source = args.source.expanduser().resolve()
    scene = ScenePaths.from_root(args.scene)
    if not source.exists():
        parser.error(f"source does not exist: {source}")
    if scene.images.exists() and not args.force:
        parser.error(f"{scene.images} already exists; pass --force to replace it")
    if not shutil.which(args.sharp_frames):
        parser.error(f"{args.sharp_frames!r} is not on PATH; install with: pip install sharp-frames")

    attempt = scene.staging_attempt("sharpframes", scene.new_run_id())
    staged_images = attempt / "images"
    cmd = [args.sharp_frames, str(source), str(staged_images), "--overwrite", *extra]
    console.print(f"[bold]Sharp Frames[/bold] → {scene.images}")
    result = run_command(cmd, cwd=scene.root, on_line=console.print)
    if not result.succeeded:
        if result.output_tail:
            console.print("[bold red]Sharp Frames failed:[/bold red]\n" + "\n".join(result.output_tail[-20:]))
        sys.exit(1)

    try:
        staged_scene = ScenePaths.from_root(attempt)
        images = staged_scene.image_files()
    except SceneError as exc:
        console.print(f"[bold red]Sharp Frames produced an invalid image set:[/bold red] {exc}")
        sys.exit(1)
    if not images:
        console.print("[bold red]Sharp Frames selected no images; existing images/ was left unchanged.[/bold red]")
        sys.exit(1)

    replace_directory(staged_images, scene.images)
    record_stage(
        scene,
        "sharpframes",
        {
            "source": str(source),
            "source_fingerprint": fingerprint_file(source) if source.is_file() else None,
            "parameters": {"command": cmd[3:]},
            "outputs": {"images": str(scene.relative_path(scene.images)), "count": len(images)},
        },
    )
    shutil.rmtree(attempt, ignore_errors=True)
    console.print(f"[green]Published {len(images)} selected images to {scene.images}[/green]")
