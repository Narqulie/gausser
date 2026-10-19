"""Scene folder layout, tool execution and atomic replacement of step outputs.

The files in the scene folder are the only state. A step writes into a
temporary folder under ``.gausser/tmp`` and renames the result into place only
after it succeeded, so a failed or interrupted step never leaves partial output.
"""

from __future__ import annotations

from contextlib import closing
from datetime import datetime
import shutil
import sqlite3
import struct
import subprocess
from pathlib import Path

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"})


class GausserError(Exception):
    """A user-facing failure; the CLI prints the message and exits 1."""


def is_image(path: Path) -> bool:
    # "._name" files are macOS AppleDouble sidecars, not images.
    return path.is_file() and not path.name.startswith("._") and path.suffix.lower() in IMAGE_EXTENSIONS


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def registered_images(model: Path) -> int:
    """Registered image count from a binary COLMAP model's images.bin header."""
    try:
        with (model / "images.bin").open("rb") as stream:
            return int(struct.unpack("<Q", stream.read(8))[0])
    except (OSError, struct.error):
        return 0


class Scene:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise GausserError(f"Scene folder does not exist: {self.root}")
        self.images = self.root / "images"
        self.masks = self.root / "masks"  # Brush: <stem>.png
        self.colmap_masks = self.root / "colmap" / "masks"  # COLMAP: <name><ext>.png
        self.database = self.root / "database.db"
        self.sparse = self.root / "sparse"
        self.runs = self.root / "runs"
        self.tmp = self.root / ".gausser" / "tmp"

    def image_files(self) -> list[Path]:
        if not self.images.is_dir():
            return []
        files = sorted(path for path in self.images.iterdir() if is_image(path))
        stems = [path.stem for path in files]
        if len(set(stems)) != len(stems):
            raise GausserError("Two images share a file stem (e.g. a.jpg and a.png); Brush masks would collide")
        return files

    def require_images(self) -> list[Path]:
        files = self.image_files()
        if not files:
            raise GausserError(f"No images in {self.images}; add photos there or run `gausser frames VIDEO`")
        return files

    def model(self) -> Path | None:
        """The COLMAP model at sparse/0, if complete."""
        model = self.sparse / "0"
        if all((model / name).is_file() for name in ("cameras.bin", "images.bin", "points3D.bin")):
            return model
        return None

    def require_model(self) -> Path:
        model = self.model()
        if model is None:
            raise GausserError("No COLMAP model at sparse/0; run `gausser solve` first")
        return model

    def masks_done(self, images: list[Path]) -> bool:
        return all((self.masks / f"{image.stem}.png").is_file() for image in images)

    def solve_done(self, images: list[Path]) -> bool:
        """A model exists and was built from exactly the current images."""
        if self.model() is None or not self.database.is_file():
            return False
        with closing(sqlite3.connect(self.database)) as database:
            solved = {name for (name,) in database.execute("SELECT name FROM images")}
        return solved == {image.name for image in images}

    def train_done(self) -> bool:
        """The newest export was trained after the current model was built."""
        model = self.model()
        exports = self.exports()
        return model is not None and bool(exports) and mtime(exports[-1]) > mtime(model / "images.bin")

    def lod_done(self) -> bool:
        exports, lods = self.exports(), self.lod_runs()
        return bool(exports and lods) and max(mtime(run / "lod-meta.json") for run in lods) > mtime(exports[-1])

    def exports(self) -> list[Path]:
        """Trained PLY exports, oldest first."""
        found = [*self.runs.glob("brush/*/exports/*.ply"), *self.runs.glob("msplat/*/exports/*.ply")]
        return sorted(found, key=lambda path: path.stat().st_mtime)

    def lod_runs(self) -> list[Path]:
        return sorted(path.parent for path in self.runs.glob("lod/*/lod-meta.json"))

    def staging(self, name: str) -> Path:
        """A fresh, empty temporary folder on the scene's filesystem."""
        path = self.tmp / f"{name}-{timestamp()}"
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True)
        return path

    def new_run(self, kind: str) -> Path:
        path = self.runs / kind / timestamp()
        suffix = 1
        while path.exists():
            suffix += 1
            path = self.runs / kind / f"{timestamp()}-{suffix}"
        return path


def remove(*paths: Path) -> None:
    for path in paths:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def replace(staged: Path, destination: Path) -> None:
    """Move a finished step output into place, removing what it replaces."""
    remove(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged.replace(destination)


def require_tool(name: str, hint: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise GausserError(f"`{name}` is not on PATH. Install: {hint}")
    return path


def run_tool(command: list[str], *, cwd: Path) -> None:
    """Run an external tool with live output; raise if it fails."""
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        raise GausserError(f"`{Path(command[0]).name}` failed with exit code {result.returncode}")
