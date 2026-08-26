from __future__ import annotations

import struct
from pathlib import Path

from gausser.colmap_model import read_camera_positions, read_model_stats


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


def test_read_model_stats_rejects_partial_binary_output(tmp_path: Path):
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (tmp_path / name).write_bytes(b"\x01")

    assert read_model_stats(tmp_path) is None


def test_read_model_stats_rejects_binary_headers_without_records(tmp_path: Path):
    for name, count in (("cameras.bin", 1), ("images.bin", 3), ("points3D.bin", 7)):
        (tmp_path / name).write_bytes(struct.pack("<Q", count))

    assert read_model_stats(tmp_path) is None
