"""`gausser orbit`: render a 360° orbit video of a splat with SuperSplat's own video renderer.

The camera path comes from the COLMAP model (cameras look at the subject), the vertical
axis from the Brush export. A headed Chromium driven by ``orbit.mjs`` (Playwright) does the rendering.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
from pathlib import Path
from typing import Any

from .scene import GausserError, Scene, replace, require_tool, run_tool
from .view import resolve_splat, serve

NODE_HINT = "brew install node"
DRIVER = Path(__file__).with_name("orbit.mjs")


def node_dir() -> Path:
    """Where `npm install playwright-core` put its node_modules (scripts/install-macos.sh)."""
    return Path(os.environ.get("GAUSSER_NODE_DIR", "~/.local/share/gausser/node")).expanduser()


def has_playwright() -> bool:
    return (node_dir() / "node_modules" / "playwright-core").is_dir()


def read_cameras(images_bin: Path) -> tuple[Any, Any]:
    """Centres and viewing directions of all registered cameras in a binary COLMAP images.bin."""
    import numpy as np

    centres, directions = [], []
    with images_bin.open("rb") as f:
        (count,) = struct.unpack("<Q", f.read(8))
        for _ in range(count):
            _id, qw, qx, qy, qz, tx, ty, tz, _camera = struct.unpack("<idddddddi", f.read(64))
            while f.read(1) not in (b"\0", b""):  # image name
                pass
            (points,) = struct.unpack("<Q", f.read(8))
            f.seek(points * 24, 1)
            r = np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ])
            centres.append(-r.T @ np.array([tx, ty, tz]))
            directions.append(r.T @ np.array([0, 0, 1.0]))
    return np.array(centres), np.array(directions)


def vertical_axis(ply: Path) -> Any:
    """Unit up vector from Brush's `comment Vertical axis: x y z` PLY header line."""
    import numpy as np

    with ply.open("rb") as f:
        for line in iter(f.readline, b""):
            if line.startswith(b"comment Vertical axis:"):
                up = np.array([float(x) for x in line.split(b":")[1].split()])
                return up / np.linalg.norm(up)
            if line.startswith(b"end_header"):
                break
    raise GausserError(f"{ply.name} has no `comment Vertical axis` header; only Brush exports have one")


def orbit_params(images_bin: Path, up: Any, zoom: float = 1.3, drop: float = 0.35) -> dict[str, Any]:
    """Orbit centre, radius and height in the PLY's own coordinates.

    The centre is the point nearest all side-looking camera axes. ``zoom`` widens the
    radius and ``drop`` lowers the aim point (the camera keeps its height) for framing.
    """
    import numpy as np

    centres, directions = read_cameras(images_bin)
    side = np.abs(directions @ up) < 0.7  # top-down frames do not aim at the subject's centre
    if side.sum() < 3:
        raise GausserError(f"Only {int(side.sum())} registered cameras look sideways; cannot find an orbit centre")
    a = np.zeros((3, 3))
    b = np.zeros(3)
    for centre, direction in zip(centres[side], directions[side]):
        p = np.eye(3) - np.outer(direction, direction)
        a += p
        b += p @ centre
    target = np.linalg.solve(a, b)
    offset = centres[side] - target
    height = float(np.median(offset @ up))
    radius = float(np.median(np.linalg.norm(offset - np.outer(offset @ up, up), axis=1)))
    return {
        "target": (target - drop * up).tolist(), "up": up.tolist(),
        "radius": radius * zoom, "height": height + drop,
    }


def orbit(
    scene: Scene, source: Path | None, *, seconds: float, size: str, zoom: float, drop: float, out: Path | None,
) -> Path:
    """Render runs/orbit/<time>/orbit.mp4 (or ``out``); needs sparse/0 and a Brush export."""
    match = re.fullmatch(r"(\d+)x(\d+)", size)
    if not match or seconds <= 0 or zoom <= 0:
        raise GausserError("--size must look like 1920x1080; --seconds and --zoom must be positive")
    node = require_tool("node", NODE_HINT)
    if not has_playwright():
        raise GausserError(f"playwright-core not found in {node_dir()}; run scripts/install-macos.sh")
    model = scene.require_model()
    splat = resolve_splat(scene, source)
    params = orbit_params(model / "images.bin", vertical_axis(splat), zoom, drop)

    staged = scene.staging("orbit")
    (staged / "orbit.json").write_text(json.dumps(params))
    video = staged / "orbit.mp4"
    with serve(splat) as url:
        run_tool(
            [node, str(DRIVER), url, str(staged / "orbit.json"), str(video), str(seconds), *match.groups()],
            cwd=scene.root, env={"GAUSSER_NODE_DIR": str(node_dir())},
        )
    if not video.is_file():
        raise GausserError("The browser did not save a video")
    if out:
        out = out.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(video, out)  # not replace(): -o may be on another filesystem than the scene
        shutil.rmtree(staged)
    else:
        out = scene.new_run("orbit")
        replace(staged, out)
        out /= "orbit.mp4"
    print(f"Orbit video: {out}")
    return out
