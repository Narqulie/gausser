from __future__ import annotations

import struct
from pathlib import Path

import pytest

from gausser.colmap_model import read_camera_positions, read_model_stats, read_reconstruction_preview, read_sparse_points


def write_binary_model(path: Path, *, images: int = 3, points: int = 7) -> None:
    """Create a structurally valid miniature COLMAP binary model."""
    cameras = struct.pack("<QIiQQ4d", 1, 1, 1, 1, 1, 1.0, 1.0, 0.5, 0.5)
    image_records = b"".join(
        struct.pack("<I7dI", index, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1)
        + f"frame-{index}.jpg".encode() + b"\0" + struct.pack("<Q", 0)
        for index in range(1, images + 1)
    )
    point_records = b"".join(
        struct.pack("<Q3d3BdQ", index, 0.0, 0.0, 1.0, 255, 255, 255, 0.1, 0)
        for index in range(1, points + 1)
    )
    (path / "cameras.bin").write_bytes(cameras)
    (path / "images.bin").write_bytes(struct.pack("<Q", images) + image_records)
    (path / "points3D.bin").write_bytes(struct.pack("<Q", points) + point_records)


def test_read_model_stats_reads_binary_counts(tmp_path: Path):
    write_binary_model(tmp_path)

    stats = read_model_stats(tmp_path)

    assert stats is not None
    assert stats.format == "binary"
    assert (stats.images, stats.points) == (3, 7)


def test_read_model_stats_reads_text_model(tmp_path: Path):
    (tmp_path / "cameras.txt").write_text("# Camera list\n1 PINHOLE 1 1 1 1 1 1 1\n")
    (tmp_path / "images.txt").write_text(
        "# Image list\n"
        "1 1 0 0 0 0 0 0 1 frame-1.jpg\n"
        "\n"
        "2 1 0 0 0 0 0 0 1 frame-2.jpg\n"
        "\n"
    )
    (tmp_path / "points3D.txt").write_text("# Points\n1 0 0 0 1 2 3 0.1 1 1\n2 0 0 0 1 2 3 0.1 1 2\n")

    stats = read_model_stats(tmp_path)

    assert stats is not None
    assert stats.format == "text"
    assert (stats.images, stats.points) == (2, 2)


def test_text_model_parser_does_not_treat_points2d_rows_as_camera_records(tmp_path: Path):
    (tmp_path / "cameras.txt").write_text("# Camera list\n1 PINHOLE 1 1 1 1 1 1 1\n")
    (tmp_path / "images.txt").write_text(
        "# Image list\n"
        "1 1 0 0 0 0 0 0 1 frame-one.jpg\n"
        "1 100 -1 2 200 -1 3 300 -1 4 400 -1\n"
        "2 1 0 0 0 0 0 0 1 frame-two.jpg\n"
        "5 500 -1 6 600 -1 7 700 -1 8 800 -1\n"
    )
    (tmp_path / "points3D.txt").write_text("# Points\n1 0 0 0 1 2 3 0.1 1 1\n")

    stats = read_model_stats(tmp_path)
    cameras = read_camera_positions(tmp_path)

    assert stats is not None and stats.images == 2
    assert cameras is not None and [camera.name for camera in cameras] == ["frame-one.jpg", "frame-two.jpg"]


def test_text_model_requires_a_well_formed_points2d_row_for_every_image(tmp_path: Path):
    (tmp_path / "cameras.txt").write_text("# Camera list\n1 PINHOLE 1 1 1 1 1 1 1\n")
    (tmp_path / "points3D.txt").write_text("# Points\n")

    (tmp_path / "images.txt").write_text("1 1 0 0 0 0 0 0 1 frame.jpg\n")
    assert read_model_stats(tmp_path) is None

    (tmp_path / "images.txt").write_text("1 1 0 0 0 0 0 0 1 frame.jpg\n100 200 -1 trailing\n")
    assert read_model_stats(tmp_path) is None


def test_read_camera_positions_converts_colmap_world_to_camera_pose(tmp_path: Path):
    write_binary_model(tmp_path, images=1)
    # Rewrite the one image with an identity rotation and translation (1, 2, 3).
    (tmp_path / "images.bin").write_bytes(
        struct.pack("<Q", 1)
        + struct.pack("<I7dI", 1, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 1)
        + b"frame.jpg\0" + struct.pack("<Q", 0)
    )

    cameras = read_camera_positions(tmp_path)

    assert cameras is not None
    assert cameras[0].name == "frame.jpg"
    assert (cameras[0].x, cameras[0].y, cameras[0].z) == (-1.0, -2.0, -3.0)


def test_reconstruction_preview_uses_camera_optical_ray_focus_as_pivot(tmp_path: Path):
    write_binary_model(tmp_path, images=2, points=1)
    # Two non-parallel cameras look toward the origin. COLMAP stores
    # world-to-camera qvec/tvec, from which their optical rays are recovered.
    records = b"".join([
        struct.pack("<I7dI", 1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1)
        + b"near.jpg\0" + struct.pack("<Q", 0),
        struct.pack("<I7dI", 2, 2**-0.5, 2**-0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 1)
        + b"far.jpg\0" + struct.pack("<Q", 0),
    ])
    (tmp_path / "images.bin").write_bytes(struct.pack("<Q", 2) + records)

    preview = read_reconstruction_preview(tmp_path)

    assert preview is not None
    assert preview.focus is not None
    assert preview.focus == pytest.approx((0.0, 0.0, 0.0))


def test_read_reconstruction_preview_returns_a_bounded_stable_model_snapshot(tmp_path: Path):
    write_binary_model(tmp_path, images=3, points=7)

    preview = read_reconstruction_preview(tmp_path, maximum_cameras=2, maximum_points=2)

    assert preview is not None
    assert preview.camera_count == 3
    assert preview.point_count == 7
    assert len(preview.cameras) == 2
    assert len(preview.points) == 2


def test_read_reconstruction_preview_rejects_a_model_that_changes_during_read(tmp_path: Path, monkeypatch):
    import gausser.colmap_model as colmap_model

    write_binary_model(tmp_path)
    fingerprints = iter([(("before", 1),), (("after", 2),)])
    monkeypatch.setattr(colmap_model, "_model_fingerprint", lambda _: next(fingerprints))

    assert read_reconstruction_preview(tmp_path) is None


def test_read_sparse_points_reads_colmap_xyz_and_rgb_with_bounded_sampling(tmp_path: Path):
    write_binary_model(tmp_path, points=5)
    records = b"".join(
        struct.pack("<Q3d3BdQ", index, float(index), float(index + 1), float(index + 2), index, index + 1, index + 2, 0.1, 0)
        for index in range(1, 6)
    )
    (tmp_path / "points3D.bin").write_bytes(struct.pack("<Q", 5) + records)

    points = read_sparse_points(tmp_path, maximum=2)

    assert points is not None
    assert [(point.x, point.y, point.z, point.red, point.green, point.blue) for point in points] == [
        (1.0, 2.0, 3.0, 1, 2, 3),
        (4.0, 5.0, 6.0, 4, 5, 6),
    ]


def test_read_sparse_points_reads_text_model(tmp_path: Path):
    (tmp_path / "cameras.txt").write_text("# Camera list\n1 PINHOLE 1 1 1 1 1 1 1\n")
    (tmp_path / "images.txt").write_text("# Image list\n1 1 0 0 0 0 0 0 1 frame.jpg\n\n")
    (tmp_path / "points3D.txt").write_text("# Points\n1 1 2 3 10 20 30 0.1 1 2\n")

    points = read_sparse_points(tmp_path)

    assert points is not None
    assert [(point.x, point.y, point.z, point.red, point.green, point.blue) for point in points] == [(1.0, 2.0, 3.0, 10, 20, 30)]


def test_read_model_stats_rejects_partial_binary_output(tmp_path: Path):
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (tmp_path / name).write_bytes(b"\x01")

    assert read_model_stats(tmp_path) is None


def test_read_model_stats_rejects_binary_headers_without_records(tmp_path: Path):
    for name, count in (("cameras.bin", 1), ("images.bin", 3), ("points3D.bin", 7)):
        (tmp_path / name).write_bytes(struct.pack("<Q", count))

    assert read_model_stats(tmp_path) is None
