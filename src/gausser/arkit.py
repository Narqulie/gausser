"""iPhone LiDAR scans from the Stray Scanner app: ARKit camera poses and depth.

A scan folder holds rgb.mp4 (frame n = row n of odometry.csv = depth/<n>.png), odometry.csv (per frame:
camera position x y z and rotation qx qy qz qw, world_from_camera with OpenCV camera axes, plus the
intrinsics fx fy cx cy in video pixels), depth/<n>.png (uint16 mm, e.g. 256×192) and confidence/<n>.png
(0, 1, 2 = high). ARKit's world is metric with +y up; gausser keeps gravity +y, so everything is turned
180° about x on import (FLIP).

`frames` writes arkit/poses.json (each image's camera centre, world_from_camera rotation and intrinsics)
and arkit/points.ply (the high-confidence LiDAR points of the selected frames, one per 1 cm voxel).
`solve` snaps the COLMAP model onto the poses; `train` can start Brush from the points.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from . import colmap_model, ply
from .scene import GausserError

POSES = "poses.json"
POINTS = "points.ply"
DEPTH = "depth"  # <image stem>.png: uint16 mm, high-confidence pixels only
VOXEL = 0.01  # metres: one LiDAR point per voxel
FLIP = (1.0, -1.0, -1.0)  # ARKit world (+y up) → gausser world (gravity +y)


def is_scan(path: Path) -> bool:
    return path.is_dir() and (path / "odometry.csv").is_file() and (path / "rgb.mp4").is_file()


def read_odometry(scan: Path) -> list[dict[str, Any]]:
    """Per video frame: centre, world_from_camera rotation (gausser world) and fx fy cx cy."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    flip = np.diag(FLIP)
    poses = []
    with (scan / "odometry.csv").open() as stream:
        for row in csv.DictReader(stream, skipinitialspace=True):
            rotation = Rotation.from_quat([float(row[key]) for key in ("qx", "qy", "qz", "qw")]).as_matrix()
            poses.append({
                "frame": int(row["frame"]),
                "centre": (flip @ [float(row[key]) for key in "xyz"]).tolist(),
                "rotation": (flip @ rotation).tolist(),
                "intrinsics": [float(row[key]) for key in ("fx", "fy", "cx", "cy")],
            })
    if [pose["frame"] for pose in poses] != list(range(len(poses))):
        raise GausserError(f"{scan / 'odometry.csv'} does not list frames 0, 1, 2, …")
    return poses


def import_scan(scan: Path, images: Path, items: list[dict[str, Any]], out: Path) -> int:
    """Write out/poses.json, out/points.ply and out/depth/<stem>.png for the frames `frames` selected into images/.

    `items` are frames.json's. The depth maps keep only high-confidence pixels (0 elsewhere).
    """
    import cv2
    import numpy as np

    poses = read_odometry(scan)
    selected, chunks, colours = {}, [], []
    (out / DEPTH).mkdir(parents=True)
    for item in items:
        if item["frame"] >= len(poses):
            raise GausserError(f"rgb.mp4 has more frames than odometry.csv ({item['name']})")
        pose = poses[item["frame"]]
        selected[item["name"]] = pose
        depth = cv2.imread(str(scan / "depth" / f"{item['frame']:06d}.png"), cv2.IMREAD_UNCHANGED)
        confidence = cv2.imread(str(scan / "confidence" / f"{item['frame']:06d}.png"), cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        if confidence is not None:
            depth[confidence != 2] = 0
        cv2.imwrite(str(out / DEPTH / f"{Path(item['name']).stem}.png"), depth)
        height, width = depth.shape
        rows, cols = np.nonzero(depth)
        z = depth[rows, cols] / 1000.0
        image = cv2.imread(str(images / item["name"]))
        pose["size"] = [image.shape[1], image.shape[0]]
        fx, fy, cx, cy = np.array(pose["intrinsics"]) * (width / image.shape[1])
        camera = np.stack([(cols + 0.5 - cx) / fx * z, (rows + 0.5 - cy) / fy * z, z], 1)
        # einsum, not @: numpy 1.26's Accelerate matmul emits spurious warnings on macOS (pano.py)
        chunks.append(np.einsum("ij,nj->ni", np.array(pose["rotation"]), camera) + pose["centre"])
        small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        colours.append(small[rows, cols, ::-1])
    if not chunks:
        raise GausserError(f"No depth maps in {scan / 'depth'}")
    xyz, rgb = np.concatenate(chunks), np.concatenate(colours)
    _, first = np.unique(np.floor(xyz / VOXEL).astype(np.int64), axis=0, return_index=True)
    (out / POSES).write_text(json.dumps(selected, indent=1))
    ply.write_points(out / POINTS, xyz[first], rgb[first])
    print(f"ARKit poses for {len(selected)} frames; {len(first):,} LiDAR points ({VOXEL * 100:g} cm voxels)", flush=True)
    return len(first)


def camera_params(poses: dict[str, Any]) -> str:
    """COLMAP SIMPLE_RADIAL "f,cx,cy,k" from ARKit's mean intrinsics (autofocus moves f by a few %)."""
    import numpy as np

    fx, fy, cx, cy = np.mean([pose["intrinsics"] for pose in poses.values()], axis=0)
    return f"{(fx + fy) / 2:.3f},{cx:.3f},{cy:.3f},0"


def snap_transform(model: Path, folder: Path, poses: dict[str, Any]) -> tuple[str, str]:
    """`colmap model_transformer` similarity that puts a COLMAP model onto ARKit's frame (metres, gravity +y).

    Rotation: the mean per-camera rotation between the models. Scale: the median ratio of LiDAR depth
    (folder/depth/) to COLMAP depth at the model's points. Translation: fits the camera centres.
    Camera centres alone fix neither rotation nor scale when the camera stays within a small space:
    on a 15 s room-corner scan (0.5 m of travel) they put the scale 12% off the LiDAR's.
    Returns the transform and a line for the log.
    """
    import cv2
    import numpy as np
    from scipy.spatial.transform import Rotation

    found_points = colmap_model.read_points(model)
    points = dict(zip(found_points.ids.tolist(), found_points.xyz, strict=True))
    rotations, sources, targets, ratios = [], [], [], []
    for image in colmap_model.read_images(model, observations=True):
        name, r, t, rows = image.name, image.rotation, image.translation, image.observations
        if name not in poses:
            continue
        rotations.append(np.einsum("ij,jk->ik", poses[name]["rotation"], r))  # world_arkit_from_cam · cam_from_world
        sources.append(-np.einsum("ji,j->i", r, t))
        targets.append(poses[name]["centre"])
        depth = cv2.imread(str(folder / DEPTH / f"{Path(name).stem}.png"), cv2.IMREAD_UNCHANGED)
        rows = rows[rows[:, 2] >= 0]
        if depth is None or not len(rows):
            continue
        xyz = np.array([points[int(i)] for i in rows[:, 2]])
        z = np.einsum("ij,nj->ni", r, xyz)[:, 2] + t[2]
        width, height = poses[name]["size"]
        u = np.clip((rows[:, 0] * depth.shape[1] / width).astype(int), 0, depth.shape[1] - 1)
        v = np.clip((rows[:, 1] * depth.shape[0] / height).astype(int), 0, depth.shape[0] - 1)
        lidar = depth[v, u] / 1000.0
        keep = (lidar > 0) & (z > 0)
        ratios.append(lidar[keep] / z[keep])
    found = np.concatenate(ratios) if ratios else np.empty(0)
    if len(targets) < 3 or len(found) < 100:
        raise GausserError("Too few solved images or LiDAR depths match the ARKit scan; cannot put the model in metres")
    rotation = Rotation.from_matrix(np.stack(rotations)).mean()
    s = float(np.median(found))
    source = s * np.einsum("ij,nj->ni", rotation.as_matrix(), np.array(sources))
    t = np.mean(targets, axis=0) - source.mean(0)
    x, y, z, w = rotation.as_quat()
    spread = np.percentile(found / s, [10, 90]) * 100 - 100
    note = f"{len(targets)} cameras, scale {s:.6g} from {len(found):,} LiDAR depths (10-90%: {spread[0]:+.1f}% … {spread[1]:+.1f}%)"
    return f"{s:.12g} {w:.12g} {x:.12g} {y:.12g} {z:.12g} {t[0]:.12g} {t[1]:.12g} {t[2]:.12g}\n", note


def snap_error(images_bin: Path, poses: dict[str, Any]) -> float:
    """RMS distance in metres between the (snapped) COLMAP camera centres and ARKit's."""
    import numpy as np

    from .orbit import read_cameras

    centres, _, _, names = read_cameras(images_bin)
    pairs = [(centre, poses[name]["centre"]) for centre, name in zip(centres, names, strict=True) if name in poses]
    return float(np.sqrt(np.mean([np.sum((np.array(a) - b) ** 2) for a, b in pairs])))
