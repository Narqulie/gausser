from __future__ import annotations

import json
import struct
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from gausser import arkit, steps
from gausser.scene import Scene

WIDTH, HEIGHT, F = 64, 48, 50.0  # video frames; the depth maps are a quarter of that
DEPTH_SHAPE = (12, 16)


def stray_scan(folder: Path, frames: int) -> None:
    """A Stray Scanner folder: camera at the ARKit origin looking along -z (ARKit), a wall 2 m away."""
    (folder / "depth").mkdir(parents=True)
    (folder / "confidence").mkdir()
    (folder / "rgb.mp4").write_bytes(b"")
    rows = ["timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y"]
    # OpenCV camera axes in ARKit's world: x right, y down (-y), z forward (-z) = 180° about x.
    qx, qy, qz, qw = Rotation.from_euler("x", 180, degrees=True).as_quat()
    for frame in range(frames):
        rows.append(f"{frame / 15}, {frame:06d}, {frame * 0.1}, 0, 0, {qx}, {qy}, {qz}, {qw}, {F}, {F}, 32, 24, , ")
        cv2.imwrite(str(folder / "depth" / f"{frame:06d}.png"), np.full(DEPTH_SHAPE, 2000, np.uint16))
        confidence = np.full(DEPTH_SHAPE, 2, np.uint8)
        confidence[0] = 1  # low confidence: left out
        cv2.imwrite(str(folder / "confidence" / f"{frame:06d}.png"), confidence)
    (folder / "odometry.csv").write_text("\n".join(rows) + "\n")


def test_a_lidar_scan_folder_gives_frames_poses_depth_and_points_in_the_gravity_plus_y_world(tmp_path, monkeypatch):
    scan = tmp_path / "scan"
    stray_scan(scan, 4)
    (tmp_path / "scene").mkdir()
    scene = Scene(tmp_path / "scene")

    def select(videos, ranges, staging, staged, rig, *rest):
        assert videos == [scan / "rgb.mp4"] and rig is None
        items = [{"name": f"frame_{n:06d}.jpg", "frame": n} for n in (1, 3)]
        for item in items:
            cv2.imwrite(str(staged / item["name"]), np.full((HEIGHT, WIDTH, 3), (0, 0, 255), np.uint8))  # red
        return items

    monkeypatch.setattr(steps, "_select_frames", select)
    steps.frames(scene, scan)

    poses = json.loads((scene.arkit / "poses.json").read_text())
    assert sorted(poses) == ["frame_000001.jpg", "frame_000003.jpg"]
    assert np.allclose(poses["frame_000003.jpg"]["centre"], [0.3, 0, 0])
    assert np.allclose(poses["frame_000001.jpg"]["rotation"], np.eye(3), atol=1e-9)  # looks along +z, image down = +y
    assert poses["frame_000001.jpg"]["size"] == [WIDTH, HEIGHT]
    assert (scene.arkit / "depth" / "frame_000001.png").is_file()
    xyz, rgb = arkit.read_points(scene.arkit / "points.ply", 10**6)
    assert np.allclose(xyz[:, 2], 2.0) and (rgb[:, 0] > 240).all() and (rgb[:, 2] < 15).all()  # red, through JPEG
    # The low-confidence top row (image up = -y) is left out: the points start one depth pixel lower.
    assert xyz[:, 1].min() > (0.5 + 1 - 24 / 4) / (F / 4) * 2 - 0.01

    (tmp_path / "photos").mkdir()  # other frames afterwards remove arkit/
    cv2.imwrite(str(tmp_path / "photos" / "a.jpg"), np.zeros((HEIGHT, WIDTH, 3), np.uint8))
    steps.frames(scene, tmp_path / "photos")
    assert not scene.arkit.exists()


def write_model(model: Path, cameras: list[tuple[str, np.ndarray, np.ndarray, list[tuple[float, float, int]]]], points: dict[int, np.ndarray]) -> None:
    """Binary images.bin (with observations) and points3D.bin."""
    model.mkdir(parents=True)
    records = []
    for index, (name, r, t, observed) in enumerate(cameras, start=1):
        x, y, z, w = Rotation.from_matrix(r).as_quat()
        record = struct.pack("<idddddddi", index, w, x, y, z, *t, 1) + name.encode() + b"\0" + struct.pack("<Q", len(observed))
        records.append(record + b"".join(struct.pack("<ddq", u, v, p) for u, v, p in observed))
    (model / "images.bin").write_bytes(struct.pack("<Q", len(records)) + b"".join(records))
    data = [struct.pack("<Qddd3BdQ", i, *p, 0, 0, 0, 0.5, 0) for i, p in points.items()]
    (model / "points3D.bin").write_bytes(struct.pack("<Q", len(data)) + b"".join(data))


def test_snap_puts_a_colmap_model_into_arkit_metres_by_camera_rotations_and_lidar_depth(tmp_path):
    rng = np.random.default_rng(1)
    truth = {i: np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(3, 4)]) for i in range(1, 41)}
    # The COLMAP model's frame: x_colmap = k · R0 · X + t0
    k, r0, t0 = 7.0, Rotation.from_euler("xyz", [30, -50, 100], degrees=True).as_matrix(), np.array([1.0, -2.0, 5.0])
    poses, cameras = {}, []
    (tmp_path / "arkit" / "depth").mkdir(parents=True)
    for index in range(5):
        name = f"frame_{index:06d}.jpg"
        world_from_cam = Rotation.from_euler("xy", [rng.uniform(-5, 5), rng.uniform(-5, 5)], degrees=True).as_matrix()
        centre = np.array([0.1 * index, 0.05 * index, 0.0])
        poses[name] = {"centre": centre.tolist(), "rotation": world_from_cam.tolist(), "size": [WIDTH * 10, HEIGHT * 10]}
        depth = np.zeros(DEPTH_SHAPE, np.uint16)
        observed = []
        for point, xyz in truth.items():
            cam = world_from_cam.T @ (xyz - centre)
            u, v = F * 10 * cam[0] / cam[2] + 320, F * 10 * cam[1] / cam[2] + 240
            observed.append((u, v, point))
            depth[int(v * DEPTH_SHAPE[0] / 480), int(u * DEPTH_SHAPE[1] / 640)] = round(cam[2] * 1000)
        cv2.imwrite(str(tmp_path / "arkit" / "depth" / f"frame_{index:06d}.png"), depth)
        r = world_from_cam.T @ r0.T  # cam_from_colmap
        cameras.append((name, r, -r @ (k * r0 @ centre + t0), observed))
    write_model(tmp_path / "model", cameras, {i: k * r0 @ p + t0 for i, p in truth.items()})

    text, note = arkit.snap_transform(tmp_path / "model", tmp_path / "arkit", poses)

    s, w, x, y, z, *t = map(float, text.split())
    r = Rotation.from_quat([x, y, z, w]).as_matrix()
    for xyz in truth.values():
        assert np.allclose(s * r @ (k * r0 @ xyz + t0) + t, xyz, atol=0.03)  # depth maps are mm, pixels coarse
    assert "5 cameras" in note
