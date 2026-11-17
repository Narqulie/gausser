"""Read binary COLMAP models (cameras.bin, images.bin, points3D.bin); split database.db's camera."""

from __future__ import annotations

import sqlite3
import struct
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Parameters per camera model id; the models listed in SINGLE_FOCAL have one focal length (f, cx, cy, …).
PARAM_COUNTS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12, 11: 16}
SINGLE_FOCAL = {0, 2, 3, 8, 9}


@dataclass
class Camera:
    model: int
    width: int
    height: int
    params: tuple[float, ...]

    @property
    def intrinsics(self) -> tuple[float, float, float, float]:
        """fx, fy, cx, cy."""
        p = self.params
        return (p[0], p[0], p[1], p[2]) if self.model in SINGLE_FOCAL else (p[0], p[1], p[2], p[3])


@dataclass
class Image:
    name: str
    camera_id: int
    rotation: Any  # cam_from_world R (3×3)
    translation: Any  # cam_from_world t
    observations: Any  # (N, 3): x, y, point id (-1 = none); empty unless asked for

    @property
    def centre(self) -> Any:
        return -self.rotation.T @ self.translation


def quaternion_matrix(qw: float, qx: float, qy: float, qz: float) -> Any:
    import numpy as np

    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def read_cameras(model: Path) -> dict[int, Camera]:
    data = (model / "cameras.bin").read_bytes()
    cameras = {}
    (count,), at = struct.unpack_from("<Q", data), 8
    for _ in range(count):
        camera_id, kind, width, height = struct.unpack_from("<iiQQ", data, at)
        params = struct.unpack_from(f"<{PARAM_COUNTS[kind]}d", data, at + 24)
        at += 24 + 8 * PARAM_COUNTS[kind]
        cameras[camera_id] = Camera(kind, width, height, params)
    return cameras


def read_images(model: Path, observations: bool = False) -> list[Image]:
    import numpy as np

    data = (model / "images.bin").read_bytes()
    images = []
    (count,), at = struct.unpack_from("<Q", data), 8
    for _ in range(count):
        _id, qw, qx, qy, qz, tx, ty, tz, camera_id = struct.unpack_from("<idddddddi", data, at)
        end = data.index(b"\0", at + 64)
        (points,) = struct.unpack_from("<Q", data, end + 1)
        rows = np.empty((0, 3))
        if observations and points:
            rows = np.frombuffer(data, dtype="<f8", count=points * 3, offset=end + 9).reshape(-1, 3).copy()
            rows[:, 2] = np.frombuffer(data, dtype="<i8", count=points * 3, offset=end + 9)[2::3]
        name = data[at + 64:end].decode(errors="replace")
        images.append(Image(name, camera_id, quaternion_matrix(qw, qx, qy, qz), np.array([tx, ty, tz]), rows))
        at = end + 9 + points * 24
    return images


@dataclass
class Points:
    ids: Any  # (N,) uint64
    xyz: Any  # (N, 3)
    rgb: Any  # (N, 3) uint8
    errors: Any  # (N,) reprojection error, px
    tracks: Any  # (N,) number of images that see each point


def read_points(model: Path) -> Points:
    import numpy as np

    data = (model / "points3D.bin").read_bytes()
    (count,), at = struct.unpack_from("<Q", data), 8
    ids, tracks = np.empty(count, dtype=np.uint64), np.empty(count, dtype=np.int64)
    xyz, rgb, errors = np.empty((count, 3)), np.empty((count, 3), dtype=np.uint8), np.empty(count)
    for i in range(count):
        ids[i], *xyz[i], r, g, b, errors[i], tracks[i] = struct.unpack_from("<QdddBBBdQ", data, at)
        rgb[i] = r, g, b
        at += 51 + int(tracks[i]) * 8
    return Points(ids, xyz, rgb, errors, tracks)


def write_points3d(path: Path, xyz: Any, rgb: Any) -> None:
    """Write a COLMAP points3D.bin with empty tracks (Brush reads only position and colour)."""
    import numpy as np

    record = np.dtype([("id", "<u8"), ("xyz", "<f8", 3), ("rgb", "u1", 3), ("error", "<f8"), ("track", "<u8")])
    points = np.zeros(len(xyz), dtype=record)
    points["id"] = np.arange(1, len(xyz) + 1)
    points["xyz"], points["rgb"] = xyz, rgb
    with path.open("wb") as stream:
        stream.write(np.uint64(len(xyz)).tobytes())
        stream.write(points.tobytes())


def split_cameras(database: Path, group: Callable[[str], int]) -> None:
    """Give each group of images (by name; group 0 keeps it) its own copy of the one camera
    `feature_extractor --ImageReader.single_camera 1` made; bundle adjustment then refines each.

    COLMAP 4 puts every image in a frame of a rig whose sensor is its camera, so each new camera
    also gets its own single-camera rig, and its images' frames move to it (the mapper checks this).
    """
    with closing(sqlite3.connect(database)) as db:
        ((camera,),) = db.execute("SELECT camera_id FROM cameras").fetchall()
        groups = {name: group(name) for (name,) in db.execute("SELECT name FROM images")}
        rigs = {}
        for index in sorted(set(groups.values()) - {0}):
            db.execute(
                "INSERT INTO cameras (camera_id, model, width, height, params, prior_focal_length) "
                "SELECT ?, model, width, height, params, prior_focal_length FROM cameras WHERE camera_id = ?",
                (camera + index, camera),
            )
            rigs[index] = db.execute("INSERT INTO rigs (ref_sensor_id, ref_sensor_type) VALUES (?, 0)", (camera + index,)).lastrowid
        for name, index in groups.items():
            if not index:
                continue
            db.execute("UPDATE images SET camera_id = ? WHERE name = ?", (camera + index, name))
            ((image,),) = db.execute("SELECT image_id FROM images WHERE name = ?", (name,)).fetchall()
            db.execute("UPDATE frames SET rig_id = ? WHERE frame_id IN "
                       "(SELECT frame_id FROM frame_data WHERE data_id = ? AND sensor_type = 0)", (rigs[index], image))
            db.execute("UPDATE frame_data SET sensor_id = ? WHERE data_id = ? AND sensor_type = 0", (camera + index, image))
        db.commit()
