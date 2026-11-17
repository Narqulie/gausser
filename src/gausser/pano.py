"""360° frames: render each frame in panos/ into 14 pinhole views that COLMAP solves as a rig.

Two kinds of frame:

* Equirectangular (2:1, already stitched). COLMAP's own panorama layout (``pycolmap.panorama``,
  ``PERSPECTIVE_OVERLAPPING``): 4 yaws at pitches -35°, 0° and 35°, plus straight up and straight
  down. All views share the panorama's centre.
* Dual fisheye straight from the camera (``panos/lenses.json``, written by `frames` from the
  QooCam's own calibration, see qoocam.py): the two lens images side by side. Each lens gives 7
  views (its axis and a ring of 6 at 55°), so every view is a real pinhole camera at its lens's
  centre: no stitch seam, and the 3 cm between the lenses is part of the rig. Pixels outside the
  lens circle and in a cone around the grip (`GRIP_DEG`, the operator's hand or a stick) are
  black in ``masks-exclude/``, so neither COLMAP nor Brush learns from them.

View i of frame P is ``images/cII_P.jpg``, square and 90° wide, so image stems stay unique and every
later step (masks, Brush, orbit) sees ordinary photos. COLMAP groups the views of one
frame into a rig frame by stripping the ``cII_`` prefix.
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

from .scene import GausserError, read_json

YAW_STEPS = 4
PITCHES = (-35.0, 0.0, 35.0)
POLES = (90.0, -90.0)
RING, RING_DEG = 6, 55.0  # per fisheye lens: its axis plus a ring of 6 views tilted 55° (covers 98°)
VIEWS = YAW_STEPS * len(PITCHES) + len(POLES)  # 14, also 2 lenses × (1 + RING)
GRIP_DEG = 35.0  # half-angle of the cone around the camera's grip that no view trains on
LENSES = "lenses.json"  # in panos/: the frames are side-by-side fisheye images of this rig
_VIEW = re.compile(r"c(\d\d)_")


def is_pano(path: Path) -> bool:
    from PIL import Image

    with Image.open(path) as image:
        return image.width == 2 * image.height


def lenses(panos: Path) -> dict[str, Any] | None:
    return read_json(panos / LENSES)


def view_name(view: int, pano: Path) -> str:
    return f"c{view:02d}_{pano.stem}.jpg"


def view_index(name: str) -> int:
    match = _VIEW.match(name)
    if match is None or int(match[1]) >= VIEWS:
        raise ValueError(f"{name} is not a panorama view (cNN_ prefix)")
    return int(match[1])


def _rotation(pitch_deg: float, yaw_deg: float) -> Any:
    """cam_from_frame of a view at this pitch and yaw (pycolmap.panorama's convention)."""
    import numpy as np

    pitch, yaw = np.deg2rad([-pitch_deg, -yaw_deg])
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    return rx @ ry


def rotations() -> list[Any]:
    """cam_from_pano rotation of each equirect view (pycolmap.panorama's order, then the poles)."""
    import numpy as np

    views = [
        (pitch, yaw)
        for pitch in PITCHES
        for yaw in np.linspace(0, 360, YAW_STEPS, endpoint=False) + (360 / YAW_STEPS / 2 if pitch > 0 else 0)
    ]
    return [_rotation(pitch, yaw) for pitch, yaw in [*views, *((pole, 0.0) for pole in POLES)]]


def _lens_rotations() -> list[Any]:
    """cam_from_lens of a fisheye's views: its axis, then the ring tilted `RING_DEG` around it."""
    import numpy as np

    views = [np.eye(3)]
    tilt = np.radians(RING_DEG)
    for azimuth in np.radians(np.arange(0, 360, 360 / RING)):
        z = np.array([np.sin(tilt) * np.cos(azimuth), np.sin(tilt) * np.sin(azimuth), np.cos(tilt)])
        x = np.cross([0.0, 0.0, 1.0], z)
        x /= np.linalg.norm(x)
        views.append(np.stack([x, np.cross(z, x), z]))  # rows: the view's axes in the lens frame
    return views


def layout(rig: dict[str, Any] | None) -> list[tuple[Any, Any, int | None]]:
    """(cam_from_rig rotation, translation in m, lens index or None) of each view."""
    import numpy as np

    if rig is None:
        return [(rotation, np.zeros(3), None) for rotation in rotations()]
    views: list[tuple[Any, Any, int | None]] = []
    for index, lens in enumerate(rig["lenses"]):
        rotation, translation = np.array(lens["rotation"]), np.array(lens["translation"])
        views += [(view @ rotation, view @ translation, index) for view in _lens_rotations()]
    if len(views) != VIEWS:
        raise GausserError(f"{LENSES} describes {len(rig['lenses'])} lenses; gausser handles 2")
    return views


def view_size(pano_width: int) -> int:
    return pano_width // 4  # equirect: 90° of the 360° width; side-by-side fisheyes: half a lens image


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


def _to_frame(rays: Any, cam_from_frame: Any) -> Any:
    """Rotate camera rays into the frame (pano or lens). einsum, because numpy 1.26's Accelerate
    matmul raises spurious divide-by-zero warnings on macOS."""
    import numpy as np

    return np.einsum("...i,ij->...j", rays, cam_from_frame).astype(np.float32)


def _equirect_pixels(rays: Any, width: int, height: int) -> tuple[Any, Any, Any]:
    import numpy as np

    yaw = np.arctan2(rays[..., 0], rays[..., 2])
    pitch = -np.arctan2(rays[..., 1], np.hypot(rays[..., 0], rays[..., 2]))
    x = (1 + yaw / np.pi) / 2 * width - 0.5
    y = (1 - pitch * 2 / np.pi) / 2 * height - 0.5
    return x, y, np.ones(x.shape, dtype=bool)


def _fisheye_pixels(rays: Any, lens: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Equidistant fisheye (OpenCV model) projection into the side-by-side frame; valid inside the circle."""
    import numpy as np

    theta = np.arccos(np.clip(rays[..., 2], -1, 1))
    rho = np.maximum(np.hypot(rays[..., 0], rays[..., 1]), 1e-9)
    k1, k2, k3, k4 = lens["k"]
    t2 = theta * theta
    r = lens["f"] * theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    x = lens["c"][0] + r * rays[..., 0] / rho
    y = lens["c"][1] + r * rays[..., 1] / rho
    size = lens["size"]
    valid = (theta <= np.radians(lens["max_deg"])) & (x >= 0) & (x <= size - 1) & (y >= 0) & (y <= size - 1)
    return x + lens["x0"], y, valid


def _maps(rig: dict[str, Any] | None, width: int, height: int) -> list[tuple[Any, Any, Any]]:
    """Per view: source x, y in the frame and the valid pixels."""
    rays = _rays(view_size(width))
    maps = []
    for rotation, _, index in layout(rig):
        if rig is None or index is None:
            maps.append(_equirect_pixels(_to_frame(rays, rotation), width, height))
            continue
        lens = rig["lenses"][index]
        lens_from_rig = _matrix(lens["rotation"])
        maps.append(_fisheye_pixels(_to_frame(rays, rotation @ lens_from_rig.T), lens))  # cam_from_lens
    return maps


def _matrix(values: Any) -> Any:
    import numpy as np

    return np.array(values, dtype=float)


def _sample(image: Any, x: Any, y: Any, wrap: bool) -> Any:
    """Bilinear lookup; x wraps (equirect) or clamps."""
    import numpy as np

    height, width = image.shape[:2]
    x = x if wrap else np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    x1 = (x0 + 1) % width if wrap else np.minimum(x0 + 1, width - 1)
    x0 %= width
    y1 = np.minimum(y0 + 1, height - 1)
    top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
    return np.rint(top * (1 - fy) + bottom * fy).astype(np.uint8)


def render(panos: list[Path], out: Path, rig: dict[str, Any] | None = None) -> int:
    """Write the views of every frame into ``out``; return the view side in pixels."""
    import numpy as np
    from PIL import Image

    with Image.open(panos[0]) as first:
        width, height = first.size
    maps = _maps(rig, width, height)

    def one(path: Path) -> None:
        with Image.open(path) as image:
            if image.size != (width, height):
                raise GausserError(f"{path.name} is {image.width}x{image.height}, the first frame {width}x{height}")
            frame = np.asarray(image.convert("RGB"), dtype=np.float32)
        for index, (x, y, valid) in enumerate(maps):
            view = _sample(frame, x, y, wrap=rig is None)
            view[~valid] = 0
            Image.fromarray(view).save(out / view_name(index, path), quality=95)

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        for done, _ in enumerate(pool.map(one, panos), start=1):
            print(f"[{done}/{len(panos)}] 360° frame → {VIEWS} views", flush=True)
    return view_size(width)


def keep_masks(rig: dict[str, Any] | None, size: int) -> list[Any] | None:
    """Per view of a fisheye rig: True where the lens sees the scene, outside the grip cone; None for equirect."""
    import numpy as np

    if rig is None:
        return None
    rays = _rays(size)
    grip = _matrix(rig["grip"]) / np.linalg.norm(rig["grip"])
    masks = []
    for (rotation, _, _), (_, _, valid) in zip(layout(rig), _maps(rig, size * 4, size * 2), strict=True):
        in_rig = _to_frame(rays, rotation)
        masks.append(valid & (np.einsum("...i,i->...", in_rig, grip) < np.cos(np.radians(GRIP_DEG))))
    return masks


def partition(size: int, rig: dict[str, Any] | None = None) -> list[Any]:
    """Per view, the pixels whose direction is closest to this view's axis.

    Each scene direction then gives COLMAP features in one view only, as in pycolmap.panorama.
    """
    import numpy as np

    rays, views = _rays(size), layout(rig)
    axes = np.stack([rotation[2] for rotation, _, _ in views])  # view axis in the rig frame
    return [
        np.argmax(np.einsum("...i,ji->...j", _to_frame(rays, rotation), axes), -1) == index
        for index, (rotation, _, _) in enumerate(views)
    ]


def feature_masks(images: list[Path], out: Path, masks: list[Path], rig: dict[str, Any] | None = None) -> None:
    """COLMAP masks ``out/<name>.png``: the view's `partition` and, where they exist, the masks
    ``<folder>/<name>.png`` (COLMAP layout) or ``<folder>/<stem>.png`` (Brush layout) in ``masks``."""
    import numpy as np
    from PIL import Image

    with Image.open(images[0]) as first:
        parts = partition(first.width, rig)
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


def level_transform(images_bin: Path, rig: dict[str, Any] | None = None) -> str:
    """`colmap model_transformer` similarity ("scale qw qx qy qz tx ty tz") that makes gravity +y.

    COLMAP leaves the model in an arbitrary frame (for a fisheye rig, a lens's, so the room lies
    on its side). Down is the normal of the plane the camera centres lie in (the camera was carried
    at about one height), on the side the frames' own down points to: the panorama's down, or for
    a fisheye rig the grip. Without such a plane, the frames' mean down.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    from .orbit import read_cameras

    centres, _, axes, names = read_cameras(images_bin)
    views = layout(rig)
    down_in_rig = np.array([0.0, 1.0, 0.0]) if rig is None else _matrix(rig["grip"])
    world_from_cam = np.stack([np.stack([a[0], a[1], np.cross(a[0], a[1])], 1) for a in axes])
    in_cam = np.stack([views[view_index(name)][0] @ down_in_rig for name in names])  # cam_from_rig · down
    vote = np.einsum("nij,nj->ni", world_from_cam, in_cam).mean(0)
    _, spread, vt = np.linalg.svd(centres - centres.mean(0), full_matrices=False)
    down = vt[2] * np.sign(vt[2] @ vote) if spread[2] < 0.5 * spread[1] else vote
    rotation = Rotation.align_vectors([[0.0, 1.0, 0.0]], [down / np.linalg.norm(down)])[0]
    x, y, z, w = rotation.as_quat()
    return f"1 {w:.12g} {x:.12g} {y:.12g} {z:.12g} 0 0 0\n"


def write_rig(path: Path, rig: dict[str, Any] | None = None) -> None:
    """COLMAP rig config: one rig, sensor i = prefix cII_, poses relative to view 0."""
    from scipy.spatial.transform import Rotation

    views = layout(rig)
    r0, t0, _ = views[0]
    cameras: list[dict[str, Any]] = []
    for index, (rotation, translation, _) in enumerate(views):
        camera: dict[str, Any] = {"image_prefix": f"c{index:02d}_"}
        if index == 0:
            camera["ref_sensor"] = True
        else:
            relative = rotation @ r0.T
            x, y, z, w = Rotation.from_matrix(relative).as_quat()
            camera["cam_from_rig_rotation"] = [w, x, y, z]
            camera["cam_from_rig_translation"] = (translation - relative @ t0).tolist()
        cameras.append(camera)
    path.write_text(json.dumps([{"cameras": cameras}], indent=2))
