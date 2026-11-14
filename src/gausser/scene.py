"""Scene folder layout, tool execution and atomic replacement of step outputs.

The files in the scene folder are the only state. A step writes into a
temporary folder under ``.gausser/tmp`` and renames the result into place only
after it succeeded, so a failed or interrupted step never leaves partial output.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager, suppress
from datetime import datetime
import fcntl
import json
import os
import pty
import shutil
import signal
import sqlite3
import struct
import subprocess
import sys
import termios
import time
from pathlib import Path
from typing import Any, BinaryIO, TextIO, cast

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"})


class GausserError(Exception):
    """A user-facing failure; the CLI prints the message and exits 1."""


def is_image(path: Path) -> bool:
    # "._name" files are macOS AppleDouble sidecars, not images.
    return path.is_file() and not path.name.startswith("._") and path.suffix.lower() in IMAGE_EXTENSIONS


def is_video(path: Path) -> bool:
    return path.is_file() and not path.name.startswith("._") and path.suffix.lower() in VIDEO_EXTENSIONS


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


class _Tee:
    """sys.stdout or sys.stderr during a step: text goes to the terminal and to the step's log."""

    def __init__(self, stream: TextIO, log: BinaryIO) -> None:
        self.stream, self.log = stream, log

    def write(self, text: str) -> int:
        self.log.write(text.encode(errors="replace"))
        self.log.flush()
        return self.stream.write(text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)


_log: BinaryIO | None = None  # the running step's log; run_tool copies tool output into it


GEO_FILE = "geo.json"  # in sparse/: the CRS and origin of a georeferenced model (geo.py)


def read_json(path: Path) -> dict[str, Any] | None:
    """A JSON object, or None if the file does not exist."""
    if not path.is_file():
        return None
    try:
        record: dict[str, Any] = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise GausserError(f"Cannot read {path}: {exc}") from exc
    return record


class Scene:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise GausserError(f"Scene folder does not exist: {self.root}")
        self.images = self.root / "images"
        self.panos = self.root / "panos"  # 360° source frames; images/ then holds their views (pano.py)
        self.masks = self.root / "masks"  # Brush: <stem>.png
        self.masks_exclude = self.root / "masks-exclude"  # Brush: <stem>.png, black = transient objects
        self.colmap_masks = self.root / "colmap" / "masks"  # COLMAP: <name><ext>.png
        self.database = self.root / "database.db"
        self.sparse = self.root / "sparse"
        self.crop = self.sparse / "crop.json"  # `gausser crop`: the box train and mesh keep to (crop.py)
        self.dense = self.root / "dense"  # OpenMVS: undistorted images/, scene_dense.mvs and .ply
        self.arkit = self.root / "arkit"  # iPhone LiDAR scans: poses.json and points.ply (arkit.py)
        self.runs = self.root / "runs"
        self.tmp = self.root / ".gausser" / "tmp"
        # Read only by `gausser monitor`; no step depends on them.
        self.logs = self.root / ".gausser" / "logs"
        self.history = self.root / ".gausser" / "history.jsonl"

    def image_files(self) -> list[Path]:
        if not self.images.is_dir():
            return []
        files = sorted(path for path in self.images.iterdir() if is_image(path))
        stems = [path.stem for path in files]
        if len(set(stems)) != len(stems):
            raise GausserError("Two images share a file stem (e.g. a.jpg and a.png); Brush masks would collide")
        return files

    def is_pano(self) -> bool:
        return self.panos.is_dir()

    def require_images(self) -> list[Path]:
        files = self.image_files()
        if files:
            return files
        loose = self.loose_photos()
        if loose:
            raise GausserError(
                f"{len(loose)} photos are in {self.root} itself, but gausser reads {self.images}. "
                "Add --import to `gausser run` to move them there."
            )
        raise GausserError(f"No images in {self.images}; add photos there or run `gausser frames VIDEO`")

    def loose_photos(self) -> list[Path]:
        """Photos lying in the scene folder itself instead of images/."""
        return sorted(path for path in self.root.iterdir() if is_image(path))

    def import_loose_photos(self) -> int:
        """Move loose photos into images/; never overwrites."""
        loose = self.loose_photos()
        taken = [path.name for path in loose if (self.images / path.name).exists()]
        if taken:
            raise GausserError(f"images/ already has {', '.join(taken[:3])}{' …' if len(taken) > 3 else ''}; not moving")
        self.images.mkdir(exist_ok=True)
        for path in loose:
            path.rename(self.images / path.name)
        if loose:
            print(f"Moved {len(loose)} photos into {self.images}")
        return len(loose)

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

    def dense_cloud(self) -> Path | None:
        """The dense point cloud from `gausser dense`, if there is one."""
        cloud = self.dense / "scene_dense.ply"
        return cloud if cloud.is_file() and (self.dense / "scene_dense.mvs").is_file() else None

    def arkit_poses(self) -> dict[str, Any] | None:
        return read_json(self.arkit / "poses.json")

    def geo(self) -> dict[str, Any] | None:
        """The CRS and origin of a georeferenced model (sparse/geo.json, written by `solve` from GPS)."""
        return read_json(self.sparse / GEO_FILE)

    def lidar_cloud(self) -> Path | None:
        cloud = self.arkit / "points.ply"
        return cloud if cloud.is_file() else None

    @contextmanager
    def lock(self, argv: list[str] | None = None) -> Iterator[None]:
        """Hold the scene for one command; a second command fails instead of clobbering it.

        The lock file names the holder (pid, command line) for `gausser monitor`.
        """
        path = self.root / ".gausser" / "lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+") as handle:  # not "w": a refused second command must not erase the holder
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise GausserError(f"Another gausser command is already working on {self.root}") from None
            handle.truncate(0)
            handle.write(json.dumps({"pid": os.getpid(), "argv": argv or [], "start": time.time()}))
            handle.flush()
            yield  # the OS releases the lock when the file closes or the process dies

    @contextmanager
    def step(self, name: str, **info: Any) -> Iterator[dict[str, Any]]:
        """Record a step in .gausser/history.jsonl and copy everything it prints to .gausser/logs/.

        The caller may add what the step made (e.g. ``run``) to the yielded dict; it goes into the end record.
        Every record also goes into the index of all scenes (`index_path()`), which `gausser monitor` lists.
        """
        global _log
        self.logs.mkdir(parents=True, exist_ok=True)
        log_path = self.logs / f"{timestamp()}-{name}.log"
        record = {"step": name, "start": time.time(), "log": log_path.name}
        outputs: dict[str, Any] = {}
        self._journal({**record, **info})
        stdout, stderr = sys.stdout, sys.stderr
        with log_path.open("ab") as log:
            sys.stdout, sys.stderr = cast(TextIO, _Tee(stdout, log)), cast(TextIO, _Tee(stderr, log))
            _log = log
            try:
                yield outputs
            except BaseException as exc:
                error = "interrupted" if isinstance(exc, KeyboardInterrupt) else str(exc) or type(exc).__name__
                self._journal({**record, **outputs, "end": time.time(), "ok": False, "error": error})
                raise
            finally:
                sys.stdout, sys.stderr, _log = stdout, stderr, None
        self._journal({**record, **outputs, "end": time.time(), "ok": True})

    def _journal(self, record: dict[str, Any]) -> None:
        with self.history.open("a") as history:
            history.write(json.dumps(record) + "\n")
        index(self, record)

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
        """The newest export was trained after the current model was built and the crop was set."""
        model = self.model()
        exports = self.exports()
        return model is not None and bool(exports) and mtime(exports[-1]) > max(mtime(model / "images.bin"), self._crop_time())

    def _crop_time(self) -> float:
        return mtime(self.crop) if self.crop.is_file() else 0.0

    def dense_done(self) -> bool:
        """The dense cloud was built after the current model."""
        model, cloud = self.model(), self.dense_cloud()
        return model is not None and cloud is not None and mtime(cloud) > mtime(model / "images.bin")

    def mesh_done(self) -> bool:
        cloud, meshes = self.dense_cloud(), self.mesh_runs()
        return cloud is not None and bool(meshes) and mtime(meshes[-1] / "mesh.glb") > max(mtime(cloud), self._crop_time())

    def mesh_runs(self) -> list[Path]:
        return sorted((path.parent for path in self.runs.glob("mesh/*/mesh.glb")), key=lambda run: mtime(run / "mesh.glb"))

    def map_runs(self) -> list[Path]:
        return sorted(path.parent for path in self.runs.glob("map/*/map.html"))

    def map_done(self) -> bool:
        exports, maps = self.exports(), self.map_runs()
        return bool(exports and maps) and mtime(maps[-1] / "ortho.tif") > mtime(exports[-1])

    def lod_done(self) -> bool:
        exports, lods = self.exports(), self.lod_runs()
        return bool(exports and lods) and max(mtime(run / "lod-meta.json") for run in lods) > mtime(exports[-1])

    def exports(self) -> list[Path]:
        """Trained PLY exports, oldest first."""
        found = [*self.runs.glob("brush/*/exports/*.ply"), *self.runs.glob("msplat/*/exports/*.ply")]
        return sorted((p for p in found if not p.name.startswith("._")), key=lambda path: path.stat().st_mtime)

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


def index_path() -> Path:
    """The index of every scene gausser has worked on or shown: one JSON record per line, with its scene folder."""
    return Path(os.environ.get("GAUSSER_HOME", "~/.local/share/gausser")).expanduser() / "history.jsonl"


def index(scene: Scene, record: dict[str, Any]) -> None:
    """Add a record to the index of all scenes; the index is only a list, so failing to write it is not an error."""
    path = index_path()
    with suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps({"scene": str(scene.root), **record}) + "\n")


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


def run_tool(command: list[str], *, cwd: Path, env: Mapping[str, str] | None = None) -> None:
    """Run an external tool with live output; raise if it fails. ``env`` adds variables."""
    print("$ " + " ".join(command), flush=True)
    full_env = {**os.environ, **env} if env else None
    if _log is None:
        code = subprocess.run(command, cwd=cwd, env=full_env, check=False).returncode
    else:
        code = _run_logged(command, cwd, full_env, _log)
    if code != 0:
        raise GausserError(f"`{Path(command[0]).name}` failed with exit code {code}")


def _run_logged(command: list[str], cwd: Path, env: Mapping[str, str] | None, log: BinaryIO) -> int:
    """Run a tool with its output going to stdout and to the step's log.

    On a terminal the tool gets a pseudo-terminal of the same size, so it still draws its progress bars.
    """
    out = sys.__stdout__.buffer if sys.__stdout__ else None
    resized = None
    if os.isatty(1):
        source, slave = pty.openpty()

        def resize(*_: Any) -> None:  # the tool redraws its progress bars for the new width
            with suppress(OSError):
                fcntl.ioctl(source, termios.TIOCSWINSZ, fcntl.ioctl(1, termios.TIOCGWINSZ, bytes(8)))

        resize()
        resized = signal.signal(signal.SIGWINCH, resize)
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=slave, stderr=slave)
        os.close(slave)
    else:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        source = cast(BinaryIO, process.stdout).fileno()
    try:
        while True:
            try:
                chunk = os.read(source, 65536)
            except OSError:  # EIO: the tool closed its pseudo-terminal
                chunk = b""
            if not chunk:
                break
            log.write(chunk)
            log.flush()
            if out:
                out.write(chunk)
                out.flush()
        return process.wait()
    finally:
        if resized is not None:
            signal.signal(signal.SIGWINCH, resized)
        if process.stdout:
            process.stdout.close()
        else:
            os.close(source)
        if process.poll() is None:  # interrupted
            process.kill()
            process.wait()
