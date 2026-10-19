"""The guided entry: `gausser`, `gausser FOLDER` or `gausser VIDEO` in a terminal.

It reads the scene, recommends the next step, asks only what that step needs and
returns the equivalent command line, which the CLI prints and then runs.
"""

from __future__ import annotations

from pathlib import Path

from .scene import GausserError, Scene, registered_images

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"})


def open_target(target: Path) -> tuple[Scene, Path | None, bool]:
    """Scene, optional video and whether the scene folder was just created."""
    target = target.expanduser().resolve()
    if target.is_dir():
        return Scene(target), None, False
    if target.is_file() and target.suffix.lower() in VIDEO_EXTENSIONS:
        folder = target.parent / target.stem  # clip.mp4 → clip/ next to it
        created = not folder.exists()
        folder.mkdir(exist_ok=True)
        return Scene(folder), target, created
    if target.exists():
        raise GausserError(f"{target.name} is not a folder or a video ({', '.join(sorted(VIDEO_EXTENSIONS))})")
    raise GausserError(f"No such file or folder: {target}")


def summary(scene: Scene) -> str:
    images = scene.image_files()
    if not images:
        loose = len(scene.loose_photos())
        return f"no images yet ({loose} photos in the folder itself)" if loose else "no images yet"
    parts = [f"{len(images)} images"]
    if scene.masks_done(images):
        parts.append("masked ✓")
    elif scene.masks.is_dir():
        parts.append("masks out of date")
    model = scene.model()
    if model is None:
        parts.append("cameras —")
    else:
        parts.append(f"cameras {registered_images(model)}/{len(images)}{' ✓' if scene.solve_done(images) else ' out of date'}")
    if scene.exports():
        parts.append("trained ✓" if scene.train_done() else "training out of date")
    else:
        parts.append("not trained")
    if scene.lod_runs():
        parts.append("LOD ✓" if scene.lod_done() else "LOD out of date")
    return " · ".join(parts)


def menu(scene: Scene, video: Path | None) -> list[tuple[str, str]]:
    """(label, command) pairs: the recommended step first, impossible steps left out."""
    images = scene.image_files()
    exports = scene.exports()
    available = {
        "run": "Run the pipeline (skips what is up to date)" if images or video else None,
        "frames": f"Extract frames from {video.name}" if video else "Extract frames from a video",
        "mask": "Mask the subject (SAM3)" if images else None,
        "solve": "Solve cameras (COLMAP)" if images else None,
        "train": "Train splats" if scene.model() else None,
        "lod": "Build a streamed LOD set" if exports else None,
        "view": "View the newest splat (SuperSplat)" if exports else None,
        "status": "Show full status and installed tools",
    }
    if not images:
        recommended = "run" if video else "frames"
    elif scene.solve_done(images) and scene.train_done():
        recommended = "view"
    else:
        recommended = "run"
    choices = [(f"{available[recommended]}  ← next", recommended)]
    choices += [(label, command) for command, label in available.items() if label and command != recommended]
    return choices


def _whole_number(text: str) -> bool | str:
    return (text.isdigit() and int(text) > 0) or "Enter a whole number above 0"


def _video_file(text: str) -> bool | str:
    path = Path(text.strip()).expanduser()
    return (path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS) or "Not a video file"


def guided(scene: Scene, video: Path | None) -> list[str]:
    """Ask a few questions and return the equivalent command line."""
    import questionary

    loose = scene.loose_photos()
    new = not scene.images.is_dir() and not loose
    print(f"Scene  {scene.root}{'  (new: images/ will be created here)' if new else ''}")
    if loose and not scene.image_files():
        if questionary.confirm(
            f"Move the {len(loose)} photos in this folder into images/, where gausser reads them?", default=True
        ).unsafe_ask():
            with scene.lock():
                scene.import_loose_photos()
    print(f"       {summary(scene)}\n")

    choice = questionary.select(
        "What next?", choices=[questionary.Choice(label, command) for label, command in menu(scene, video)]
    ).unsafe_ask()
    argv = [choice]
    if scene.root != Path.cwd().resolve():
        argv += ["-s", str(scene.root)]
    images = scene.image_files()

    if choice == "frames" and images:
        warning = f"This replaces the {len(images)} images and deletes their masks and cameras (runs/ is kept). Continue?"
        if not questionary.confirm(warning, default=False).unsafe_ask():
            raise KeyboardInterrupt
    if choice == "frames" or (choice == "run" and not images):
        source = str(video) if video else questionary.path("Video file:", validate=_video_file).unsafe_ask().strip()
        argv += [source] if choice == "frames" else ["--video", source]
    if choice == "mask":
        argv.append(questionary.text('Subject to keep, e.g. "statue":', validate=lambda t: bool(t.strip()) or "Required").unsafe_ask())
    if choice == "run":
        keep = "keep current masks" if images and scene.masks_done(images) else "no masking"
        subject = questionary.text(f'Subject to keep, e.g. "statue" (blank = {keep}):').unsafe_ask().strip()
        if subject:
            argv += ["--prompt", subject]
    if choice in ("run", "train"):
        trainer = questionary.select(
            "Trainer:",
            choices=[
                questionary.Choice("Brush (default)", "brush"),
                questionary.Choice("Msplat (experimental, resumable)", "msplat"),
            ],
        ).unsafe_ask()
        iterations = questionary.text("Iterations:", default="30000", validate=_whole_number).unsafe_ask()
        argv += ["--trainer", trainer, "--iterations", iterations]
    if choice == "run" and questionary.confirm("Also build a streamed LOD set at the end?", default=False).unsafe_ask():
        argv.append("--lod")
    return argv
