"""`gausser orbit`: render a 360° orbit video of a splat with SuperSplat's own video renderer.

The camera path and the vertical axis come from the COLMAP model (cameras look at the
subject; their image rows stay level). A headed Chromium driven by ``orbit.mjs`` (Playwright) does the rendering.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from .colmap_model import read_images, read_points
from .scene import GausserError, Scene, replace, require_tool, run_tool
from .view import serve

# The JUTAKA public-art viewer's default sky (PlayCanvasSplatCanvas.tsx SKY_GRADIENT).
JUTAKA_GRADIENT = "linear-gradient(to bottom, rgb(38,102,204) 0%, rgb(173,217,255) 45%, rgb(102,140,191) 55%, rgb(5,8,20) 100%)"
Rgb = tuple[int, int, int]
Stops = list[tuple[float, Rgb]]
_RGB_STOP = r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s+(\d+(?:\.\d+)?)%"
DRIVER = Path(__file__).with_name("orbit.mjs")


def node_dir() -> Path:
    """Where `npm install playwright-core` put its node_modules (scripts/install-macos.sh)."""
    return Path(os.environ.get("GAUSSER_NODE_DIR", "~/.local/share/gausser/node")).expanduser()


def has_playwright() -> bool:
    return (node_dir() / "node_modules" / "playwright-core").is_dir()


def read_cameras(images_bin: Path) -> tuple[Any, Any, Any, list[str]]:
    """Centres, viewing directions, right/down axes and names of the registered cameras in a binary images.bin."""
    import numpy as np

    images = read_images(images_bin.parent)
    if not images:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 2, 3)), []
    return (
        np.array([image.centre for image in images]),
        np.array([image.rotation[2] for image in images]),  # the camera's z (forward) axis in world coordinates
        np.array([image.rotation[:2] for image in images]),  # camera right and image-down axes in world coordinates
        [image.name for image in images],
    )


# Upright camera right/down axes from the stored photo's, per EXIF orientation: a portrait shot's stored
# x axis points up or down.
_UPRIGHT = {3: [[-1, 0], [0, -1]], 6: [[0, -1], [1, 0]], 8: [[0, 1], [-1, 0]]}


def scene_up(scene: Scene, model: Path | None = None) -> Any:
    """Unit up vector of the scene's model: -y when `solve` levelled it (360° rig, ARKit, GPS), else `camera_up`."""
    import numpy as np

    if scene.is_pano() or scene.arkit_poses() or scene.geo():
        return np.array([0.0, -1.0, 0.0])
    return camera_up((model or scene.require_model()) / "images.bin", scene.images)


def camera_up(images_bin: Path, images: Path | None = None) -> Any:
    """Unit up vector of a COLMAP model from its cameras, turned upright by each photo's EXIF orientation
    (in `images`): `level_up`; for handheld sets, whose right axes are not level enough, the ground
    (`ground_up`) or else the photos' mean up.
    """
    import numpy as np
    from PIL import Image

    centres, _, axes, names = read_cameras(images_bin)
    if images is not None:
        for index, name in enumerate(names):
            if (images / name).is_file():
                with Image.open(images / name) as opened:
                    turn = _UPRIGHT.get(opened.getexif().get(0x0112, 1))  # Orientation
                if turn:
                    axes[index] = np.einsum("ij,jk->ik", np.array(turn, dtype=float), axes[index])
    up = level_up(axes[:, 0], axes[:, 1])
    if up is not None:
        return up
    mean_up = -axes[:, 1].mean(0)
    mean_up /= np.linalg.norm(mean_up)
    ground = ground_up(read_points(images_bin.parent).xyz, centres, mean_up)
    return mean_up if ground is None else ground


def level_up(rights: Any, downs: Any) -> Any:
    """Normal to the cameras' (level) right axes, on the side image-up points to; None when they are not level.

    Brush's `Vertical axis` guess can be far off when many frames look straight down (72° on a
    drone set with top-down passes); a stabilised gimbal keeps every camera's right axis level.
    Handheld photos of one side of a subject, and the tilted views of a fisheye rig, are not level enough.
    """
    import numpy as np

    _, spread, vt = np.linalg.svd(np.asarray(rights), full_matrices=False)
    if len(rights) < 3 or spread[2] > 0.3 * spread[1]:
        return None
    up = vt[-1]
    return up if up @ -np.asarray(downs).mean(0) > 0 else -up


def ground_up(xyz: Any, centres: Any, near: Any, max_angle: float = 15.0, share: float = 0.3) -> Any:
    """Normal (on the side of `near`) of the dominant plane of the points within `max_angle`° of `near`:
    cobblestones, a lawn, a floor. None when under `share` of the points lie on one plane.

    Handheld photos' mean up carries the phones' average pitch; the ground does not (4.3° apart on a
    set of a market hut, where 73% of the points were cobblestones).
    """
    import numpy as np

    xyz = np.asarray(xyz, dtype=float)
    if len(xyz) < 3:
        return None
    tolerance = 0.005 * np.percentile(np.linalg.norm(xyz - np.mean(centres, 0), axis=1), 90)  # 4 cm at 8 m
    rng = np.random.default_rng(0)
    best, count = None, 0
    for _ in range(500):
        a, b, c = xyz[rng.choice(len(xyz), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length == 0 or abs(float(np.dot(normal, near))) < np.cos(np.radians(max_angle)) * length:
            continue
        inliers = np.abs(np.einsum("nj,j->n", xyz - a, normal / length)) < tolerance
        if inliers.sum() > count:
            best, count = inliers, int(inliers.sum())
    if best is None or count < share * len(xyz):
        return None
    plane = xyz[best]
    normal = np.linalg.svd(plane - plane.mean(0), full_matrices=False)[2][-1]
    return normal if np.dot(normal, near) > 0 else -normal


def orbit_params(images_bin: Path, up: Any, zoom: float = 1.3, drop: float = 0.35) -> dict[str, Any]:
    """Orbit centre, radius and height in the PLY's own coordinates.

    The centre is the point nearest all side-looking camera axes. ``zoom`` widens the
    radius and ``drop`` lowers the aim point (the camera keeps its height) for framing.
    """
    import numpy as np

    centres, directions, _, _ = read_cameras(images_bin)
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


def parse_background(text: str) -> Rgb | Stops:
    """`#rrggbb` → (r, g, b); `jutaka` or `linear-gradient(to bottom, rgb(r,g,b) P%, ...)` → [(P, (r, g, b)), ...]."""
    text = text.strip()
    if text.lower() == "jutaka":
        text = JUTAKA_GRADIENT
    if re.fullmatch(r"#[0-9a-fA-F]{6}", text):
        return int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16)
    match = re.fullmatch(rf"linear-gradient\(\s*to bottom\s*,((?:\s*{_RGB_STOP}\s*,?)+)\)", text)
    if match:
        stops = [(float(p), (int(r), int(g), int(b))) for r, g, b, p in re.findall(_RGB_STOP, match[1])]
        if len(stops) >= 2 and all(max(c) <= 255 for _, c in stops) and all(a[0] <= b[0] <= 100 for a, b in zip(stops, stops[1:])):
            return stops
    raise GausserError(
        f"--background {text!r}: use #rrggbb, jutaka, or linear-gradient(to bottom, rgb(r,g,b) P%, ...) "
        "with at least two rgb() stops in rising order, P from 0 to 100"
    )


def write_gradient(path: Path, stops: Stops, width: int, height: int) -> None:
    """A vertical CSS-style gradient (stops interpolated in sRGB, clamped outside the end stops) as a binary PPM."""
    import numpy as np

    where = (np.arange(height) + 0.5) / height * 100
    column = np.stack([np.interp(where, [p for p, _ in stops], [c[i] for _, c in stops]) for i in range(3)], axis=1)
    rows = np.rint(column).astype(np.uint8)
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode() + np.repeat(rows[:, None, :], width, axis=1).tobytes())


def composite_gradient(ffmpeg: str, black: Path, white: Path, gradient: Path, out: Path, cwd: Path) -> None:
    """Alpha-blend the splat over ``gradient`` as a browser would, from its renders over black and over white.

    Over black the frame is B = premultiplied colour, over white W = B + (1 - alpha) * 255, so
    out = B + (1 - alpha) * G = B + (W - B) * G / 255.
    """
    graph = (
        "[0:v]format=gbrp,split[b][b2];[1:v]format=gbrp[w];[2:v]format=gbrp[g];"  # split: a reused label corrupts the sum
        "[w][b]blend=all_mode=subtract:shortest=1[d];"
        "[d][g]blend=all_mode=multiply:shortest=1[dg];"
        "[b2][dg]blend=all_mode=addition:shortest=1,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"
    )
    run_tool(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(black), "-i", str(white), "-loop", "1", "-i", str(gradient),
         "-filter_complex", graph, "-c:v", "libx264", "-crf", "16", "-preset", "slow",
         "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv", str(out)],
        cwd=cwd,
    )


def orbit(
    scene: Scene, source: Path | None, *, seconds: float, size: str, zoom: float, drop: float, out: Path | None,
    background: str | None = None, target: list[float] | None = None, radius: float | None = None,
    height: float | None = None,
) -> Path:
    """Render runs/orbit/<time>/orbit.mp4 (or ``out``); needs sparse/0 and a Brush export.

    ``target`` (x y z in the splat's coordinates) and ``radius`` override the camera-derived
    orbit centre and radius (``height``: camera height above the target), e.g. to circle one of
    several statues in a scene.
    """
    match = re.fullmatch(r"(\d+)x(\d+)", size)
    if not match or seconds <= 0 or zoom <= 0:
        raise GausserError("--size must look like 1920x1080; --seconds and --zoom must be positive")
    backdrop = parse_background(background) if background else None
    node = require_tool("node")
    ffmpeg = require_tool("ffmpeg") if isinstance(backdrop, list) else ""
    if not has_playwright():
        raise GausserError(f"playwright-core not found in {node_dir()}; run scripts/install-macos.sh")
    model = scene.require_model()
    splat = scene.splat(source)
    up = scene_up(scene, model)
    params = orbit_params(model / "images.bin", up, zoom, drop)
    if target is not None:
        params["target"] = [t - drop * u for t, u in zip(target, up.tolist())]
    if radius is not None:
        params["radius"] = radius
    if height is not None:
        params["height"] = height

    staged = scene.staging("orbit")
    (staged / "orbit.json").write_text(json.dumps(params))
    video = staged / "orbit.mp4"
    black, white = staged / "over-black.mp4", staged / "over-white.mp4"
    if isinstance(backdrop, tuple):
        backgrounds = [f"#{backdrop[0]:02x}{backdrop[1]:02x}{backdrop[2]:02x}={video}"]
    elif isinstance(backdrop, list):  # a gradient needs the splat's alpha: render over black and over white, then blend
        backgrounds = [f"#000000={black}", f"#ffffff={white}"]
    else:
        backgrounds = []
    with serve(splat) as url:
        run_tool(
            [node, str(DRIVER), url, str(staged / "orbit.json"), str(video), str(seconds), *match.groups(), *backgrounds],
            cwd=scene.root, env={"GAUSSER_NODE_DIR": str(node_dir())},
        )
    if isinstance(backdrop, list):
        gradient = staged / "gradient.ppm"
        write_gradient(gradient, backdrop, *(int(n) for n in match.groups()))
        composite_gradient(ffmpeg, black, white, gradient, video, scene.root)
        for name in (black, white, gradient):
            name.unlink()
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
