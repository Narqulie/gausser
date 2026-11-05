"""360° panoramas: render each equirectangular frame into pinhole views that COLMAP solves as a rig.

This follows COLMAP's own panorama workflow (``pycolmap.panorama``, ``PERSPECTIVE_OVERLAPPING``):
4 yaws at pitches -35°, 0° and 35°, square 90° views, plus one view straight up and one straight down
(pycolmap has none; they hold the ceiling and floor, and everything when the camera was held lens-up). View i of panorama P is ``images/cII_P.jpg``, so
image stems stay unique and every later step (masks, Brush, composite, orbit) sees ordinary photos.
COLMAP groups the views of one panorama into a rig frame by stripping the ``cII_`` prefix.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import Any

from .scene import GausserError

YAW_STEPS = 4
PITCHES = (-35.0, 0.0, 35.0)
POLES = (90.0, -90.0)
VIEWS = YAW_STEPS * len(PITCHES) + len(POLES)
_VIEW = re.compile(r"c(\d\d)_")


def is_pano(path: Path) -> bool:
    from PIL import Image

    with Image.open(path) as image:
        return image.width == 2 * image.height


def view_name(view: int, pano: Path) -> str:
    return f"c{view:02d}_{pano.stem}.jpg"


def view_index(name: str) -> int:
    match = _VIEW.match(name)
    if match is None or int(match[1]) >= VIEWS:
        raise ValueError(f"{name} is not a panorama view (cNN_ prefix)")
    return int(match[1])


def rotations() -> list[Any]:
    """cam_from_pano rotation of each view (same order and convention as pycolmap.panorama, then the poles)."""
    import numpy as np

    views = [
        (pitch, yaw)
        for pitch in PITCHES
        for yaw in np.linspace(0, 360, YAW_STEPS, endpoint=False) + (360 / YAW_STEPS / 2 if pitch > 0 else 0)
    ]
    result = []
    for pitch_deg, yaw_deg in [*views, *((pole, 0.0) for pole in POLES)]:
        pitch, yaw = np.deg2rad([-pitch_deg, -yaw_deg])
        rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        result.append(rx @ ry)
    return result


def view_size(pano_width: int) -> int:
    return pano_width // 4  # 90° of the 360° width; square, since 90° of 180° height is the same


def camera_params(size: int) -> str:
    """SIMPLE_PINHOLE f,cx,cy of a 90° view."""
    return f"{size / 2:g},{size / 2:g},{size / 2:g}"


def _rays(size: int) -> Any:
    """Unit rays through the pixel centres of a 90° view, shape (size, size, 3), rows = y."""
    import numpy as np

    t = (np.arange(size, dtype=np.float32) + 0.5 - size / 2) / (size / 2)
    x, y = np.meshgrid(t, t)
    rays = np.stack([x, y, np.ones_like(x)], -1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def _to_pano(rays: Any, cam_from_pano: Any) -> Any:
    """Rotate camera rays into the panorama frame. einsum, because numpy 1.26's Accelerate matmul
    raises spurious divide-by-zero warnings on macOS."""
    import numpy as np

    return np.einsum("...i,ij->...j", rays, cam_from_pano).astype(np.float32)


def _sample(pano: Any, rays_in_pano: Any) -> Any:
    """Bilinear lookup of equirectangular ``pano`` along rays (yaw wraps, pitch clamps)."""
    import numpy as np

    height, width = pano.shape[:2]
    yaw = np.arctan2(rays_in_pano[..., 0], rays_in_pano[..., 2])
    pitch = -np.arctan2(rays_in_pano[..., 1], np.hypot(rays_in_pano[..., 0], rays_in_pano[..., 2]))
    x = (1 + yaw / np.pi) / 2 * width - 0.5
    y = np.clip((1 - pitch * 2 / np.pi) / 2 * height - 0.5, 0, height - 1)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    x0 %= width
    x1, y1 = (x0 + 1) % width, np.minimum(y0 + 1, height - 1)
    top = pano[y0, x0] * (1 - fx) + pano[y0, x1] * fx
    bottom = pano[y1, x0] * (1 - fx) + pano[y1, x1] * fx
    return np.rint(top * (1 - fy) + bottom * fy).astype(np.uint8)


def render(panos: list[Path], out: Path) -> int:
    """Write the views of every panorama into ``out``; return the view side in pixels."""
    import numpy as np
    from PIL import Image

    with Image.open(panos[0]) as first:
        width, height = first.size
    size = view_size(width)
    rays = _rays(size)
    views = [_to_pano(rays, rotation) for rotation in rotations()]

    def one(path: Path) -> None:
        with Image.open(path) as image:
            if image.size != (width, height):
                raise GausserError(f"{path.name} is {image.width}x{image.height}, the first panorama {width}x{height}")
            pano = np.asarray(image.convert("RGB"), dtype=np.float32)
        for index, rays_in_pano in enumerate(views):
            Image.fromarray(_sample(pano, rays_in_pano)).save(out / view_name(index, path), quality=95)

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        for done, _ in enumerate(pool.map(one, panos), start=1):
            print(f"[{done}/{len(panos)}] panorama → {VIEWS} views", flush=True)
    return size


def partition(size: int) -> list[Any]:
    """Per view, the pixels whose panorama direction is closest to this view's axis.

    Each panorama pixel then gives COLMAP features in one view only, as in pycolmap.panorama.
    """
    import numpy as np

    rays, rots = _rays(size), rotations()
    axes = np.stack([rotation[2] for rotation in rots])  # view axis in the pano frame
    return [
        np.argmax(np.einsum("...i,ji->...j", _to_pano(rays, rotation), axes), -1) == index
        for index, rotation in enumerate(rots)
    ]


def feature_masks(images: list[Path], out: Path, masks: list[Path]) -> None:
    """COLMAP masks ``out/<name>.png``: the view's `partition` and, where they exist, the masks
    ``<folder>/<name>.png`` (COLMAP layout) or ``<folder>/<stem>.png`` (Brush layout) in ``masks``."""
    import numpy as np
    from PIL import Image

    with Image.open(images[0]) as first:
        parts = partition(first.width)
    for image in images:
        keep = parts[view_index(image.name)].copy()
        for folder in masks:
            for path in (folder / f"{image.name}.png", folder / f"{image.stem}.png"):
                if path.is_file():
                    with Image.open(path) as mask:
                        keep &= np.asarray(mask.convert("L")) > 127
                    break
        Image.fromarray(keep.astype(np.uint8) * 255, mode="L").save(out / f"{image.name}.png")


def split_cameras(database: Path) -> None:
    """Give each view index its own copy of the one extracted camera; a rig sensor is a camera."""
    with closing(sqlite3.connect(database)) as db:
        ((camera,),) = db.execute("SELECT camera_id FROM cameras").fetchall()
        for index in range(VIEWS):
            if index:
                db.execute(
                    "INSERT INTO cameras (camera_id, model, width, height, params, prior_focal_length) "
                    "SELECT ?, model, width, height, params, prior_focal_length FROM cameras WHERE camera_id = ?",
                    (camera + index, camera),
                )
            db.execute("UPDATE images SET camera_id = ? WHERE substr(name, 1, 4) = ?", (camera + index, f"c{index:02d}_"))
        db.commit()


def write_rig(path: Path) -> None:
    """COLMAP rig config: one rig, sensor i = prefix cII_, all views share the panorama's centre."""
    from scipy.spatial.transform import Rotation

    rots = rotations()
    cameras: list[dict[str, Any]] = []
    for index, rotation in enumerate(rots):
        camera: dict[str, Any] = {"image_prefix": f"c{index:02d}_"}
        if index == 0:
            camera["ref_sensor"] = True
        else:
            x, y, z, w = Rotation.from_matrix(rotation @ rots[0].T).as_quat()
            camera["cam_from_rig_rotation"] = [w, x, y, z]
            camera["cam_from_rig_translation"] = [0, 0, 0]
        cameras.append(camera)
    path.write_text(json.dumps([{"cameras": cameras}], indent=2))
