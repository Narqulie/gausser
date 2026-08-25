"""Safe inspection of complete COLMAP sparse-model directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Literal


ModelFormat = Literal["binary", "text"]


@dataclass(frozen=True)
class ModelStats:
    format: ModelFormat
    images: int
    points: int


_REQUIRED = ("cameras", "images", "points3D")
# COLMAP's CameraModelId enum. Unknown model IDs are rejected: their variable
# parameter count prevents a safe structural validation of cameras.bin.
_CAMERA_PARAMETERS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}


def model_format(model_dir: Path) -> ModelFormat | None:
    """Return the format only when all three model files agree and exist."""
    directory = Path(model_dir)
    if all((directory / f"{name}.bin").is_file() for name in _REQUIRED):
        return "binary"
    if all((directory / f"{name}.txt").is_file() for name in _REQUIRED):
        return "text"
    return None


def _binary_model_counts(model_dir: Path) -> tuple[int, int] | None:
    """Parse all COLMAP binary records far enough to reject truncated models."""
    try:
        cameras = (model_dir / "cameras.bin").read_bytes()
        images = (model_dir / "images.bin").read_bytes()
        points = (model_dir / "points3D.bin").read_bytes()
    except OSError:
        return None

    def records(data: bytes, parse_record) -> int | None:
        if len(data) < 8:
            return None
        (count,) = struct.unpack_from("<Q", data)
        position = 8
        for _ in range(count):
            position = parse_record(data, position)
            if position is None:
                return None
        return count if position == len(data) else None

    def camera(data: bytes, position: int) -> int | None:
        # camera_id, model_id, width, height, and model-specific doubles.
        if position + 24 > len(data):
            return None
        _, model_id, _, _ = struct.unpack_from("<IiQQ", data, position)
        parameters = _CAMERA_PARAMETERS.get(model_id)
        if parameters is None:
            return None
        end = position + 24 + parameters * 8
        return end if end <= len(data) else None

    def image(data: bytes, position: int) -> int | None:
        # image_id, qvec, tvec, camera_id, nul-terminated name, points2D count.
        fixed_end = position + 64
        if fixed_end > len(data):
            return None
        name_end = data.find(b"\0", fixed_end)
        if name_end < 0 or name_end + 9 > len(data):
            return None
        (point_count,) = struct.unpack_from("<Q", data, name_end + 1)
        end = name_end + 9 + point_count * 24
        return end if end <= len(data) else None

    def point(data: bytes, position: int) -> int | None:
        # point3D_id, xyz, rgb, error, track length, then (image_id, point2D_idx).
        fixed_end = position + 51
        if fixed_end > len(data):
            return None
        (track_length,) = struct.unpack_from("<Q", data, position + 43)
        end = fixed_end + track_length * 8
        return end if end <= len(data) else None

    camera_count = records(cameras, camera)
    image_count = records(images, image)
    point_count = records(points, point)
    if camera_count is None or image_count is None or point_count is None:
        return None
    return image_count, point_count


def _text_data_lines(path: Path) -> list[str] | None:
    try:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    except (OSError, UnicodeDecodeError):
        return None


def _text_image_count(path: Path) -> int | None:
    lines = _text_data_lines(path)
    if lines is None:
        return None
    count = 0
    for line in lines:
        fields = line.split()
        if len(fields) < 10:
            continue
        try:
            int(fields[0])
        except ValueError:
            continue
        count += 1
    return count


def read_model_stats(model_dir: Path) -> ModelStats | None:
    """Read counts only from a complete, structurally parseable sparse model."""
    fmt = model_format(model_dir)
    if fmt == "binary":
        counts = _binary_model_counts(Path(model_dir))
        if counts is None:
            return None
        images, points = counts
    elif fmt == "text":
        images = _text_image_count(Path(model_dir) / "images.txt")
        point_lines = _text_data_lines(Path(model_dir) / "points3D.txt")
        points = len(point_lines) if point_lines is not None else None
    else:
        return None

    if images is None or points is None:
        return None
    return ModelStats(fmt, images, points)
