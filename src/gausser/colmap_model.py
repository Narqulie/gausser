"""Safe inspection of complete COLMAP sparse-model directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import struct
from typing import Literal


ModelFormat = Literal["binary", "text"]


@dataclass(frozen=True)
class ModelStats:
    format: ModelFormat
    images: int
    points: int


@dataclass(frozen=True)
class CameraPosition:
    """One COLMAP camera centre and optical-axis direction in world coordinates."""

    name: str
    x: float
    y: float
    z: float
    forward_x: float = 0.0
    forward_y: float = 0.0
    forward_z: float = 1.0


@dataclass(frozen=True)
class SparsePoint:
    """One colour-bearing point from COLMAP's authoritative sparse model."""

    x: float
    y: float
    z: float
    red: int
    green: int
    blue: int


@dataclass(frozen=True)
class ReconstructionPreview:
    """A stable, bounded, view-only snapshot of a completed sparse model."""

    cameras: list[CameraPosition]
    points: list[SparsePoint]
    camera_count: int
    point_count: int
    focus: tuple[float, float, float] | None = None


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


def _text_image_records(path: Path) -> list[list[str]] | None:
    """Parse COLMAP's two-line text-image grammar without mistaking points2D for images."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None

    records: list[list[str]] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith("#"):
            index += 1
            continue
        fields = line.split()
        if len(fields) < 10:
            return None
        try:
            int(fields[0])
            _ = [float(value) for value in fields[1:8]]
            int(fields[8])
        except ValueError:
            return None
        records.append(fields)
        # Every image record is followed by exactly one points2D row. It may be
        # empty, but it must never be reinterpreted as another image record.
        index += 1
        if index >= len(lines):
            return None
        observations = lines[index].strip()
        if observations:
            values = observations.split()
            if len(values) % 3:
                return None
            try:
                for point_index in range(0, len(values), 3):
                    float(values[point_index])
                    float(values[point_index + 1])
                    int(values[point_index + 2])
            except ValueError:
                return None
        index += 1
    return records


def _text_image_count(path: Path) -> int | None:
    records = _text_image_records(path)
    return None if records is None else len(records)


def read_camera_positions(model_dir: Path, *, maximum: int | None = None) -> list[CameraPosition] | None:
    """Read complete-model camera centres without invoking an external COLMAP tool."""
    if maximum is not None and maximum < 1:
        raise ValueError("maximum must be positive")
    stats = read_model_stats(model_dir)
    if stats is None:
        return None
    stride = 1 if maximum is None else max(1, (stats.images + maximum - 1) // maximum)
    directory = Path(model_dir)
    fmt = model_format(directory)
    if fmt == "binary":
        try:
            data = (directory / "images.bin").read_bytes()
            (count,) = struct.unpack_from("<Q", data)
        except (OSError, struct.error):
            return None
        positions: list[CameraPosition] = []
        offset = 8
        try:
            for index in range(count):
                _, qw, qx, qy, qz, tx, ty, tz, _ = struct.unpack_from("<I7dI", data, offset)
                offset += 64
                end = data.index(b"\0", offset)
                name = data[offset:end].decode("utf-8")
                offset = end + 1
                (points2d,) = struct.unpack_from("<Q", data, offset)
                offset += 8 + points2d * 24
                if index % stride == 0:
                    positions.append(_camera_center(name, qw, qx, qy, qz, tx, ty, tz))
        except (ValueError, UnicodeDecodeError, struct.error):
            return None
        return positions if offset == len(data) else None
    if fmt == "text":
        records = _text_image_records(directory / "images.txt")
        if records is None or len(records) != stats.images:
            return None
        positions = []
        for index, fields in enumerate(records):
            try:
                values = [float(value) for value in fields[1:8]]
            except ValueError:
                return None
            if index % stride == 0:
                positions.append(_camera_center(fields[9], *values))
        return positions
    return None


def read_sparse_points(model_dir: Path, *, maximum: int = 15_000) -> list[SparsePoint] | None:
    """Read a deterministic bounded sample of the completed sparse point cloud.

    The native preview consumes this *view-only* sample rather than generating a
    duplicate PLY artifact. Sampling is evenly distributed over COLMAP's stable
    point-record order, keeping the JSON-lines response bounded for large scenes.
    """
    if maximum < 1:
        raise ValueError("maximum must be positive")
    stats = read_model_stats(model_dir)
    if stats is None:
        return None
    directory = Path(model_dir)
    stride = max(1, (stats.points + maximum - 1) // maximum)
    if stats.format == "binary":
        try:
            data = (directory / "points3D.bin").read_bytes()
            (count,) = struct.unpack_from("<Q", data)
        except (OSError, struct.error):
            return None
        if count != stats.points:
            return None
        points: list[SparsePoint] = []
        offset = 8
        try:
            for index in range(count):
                _, x, y, z, red, green, blue, _, track_length = struct.unpack_from("<Q3d3BdQ", data, offset)
                offset += 51 + track_length * 8
                if offset > len(data):
                    return None
                if index % stride == 0:
                    points.append(SparsePoint(x, y, z, red, green, blue))
        except struct.error:
            return None
        return points if offset == len(data) else None

    lines = _text_data_lines(directory / "points3D.txt")
    if lines is None or len(lines) != stats.points:
        return None
    points = []
    for index, line in enumerate(lines):
        fields = line.split()
        if len(fields) < 7:
            return None
        try:
            _, x, y, z = fields[:4]
            red, green, blue = (int(value) for value in fields[4:7])
            if not all(0 <= value <= 255 for value in (red, green, blue)):
                return None
            point = SparsePoint(float(x), float(y), float(z), red, green, blue)
        except ValueError:
            return None
        if index % stride == 0:
            points.append(point)
    return points


def _model_fingerprint(model_dir: Path) -> tuple[tuple[str, int, int, str], ...] | None:
    """Return file identity data sufficient to reject a changing COLMAP publish."""
    fmt = model_format(model_dir)
    if fmt is None:
        return None
    suffix = ".bin" if fmt == "binary" else ".txt"
    try:
        entries = []
        for name in _REQUIRED:
            path = model_dir / f"{name}{suffix}"
            metadata = path.stat()
            digest = hashlib.blake2b(digest_size=16)
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1_048_576), b""):
                    digest.update(chunk)
            entries.append((name, metadata.st_size, metadata.st_mtime_ns, digest.hexdigest()))
        return tuple(entries)
    except OSError:
        return None


def read_reconstruction_preview(
    model_dir: Path,
    *,
    maximum_cameras: int = 750,
    maximum_points: int = 4_000,
) -> ReconstructionPreview | None:
    """Read one stable bounded snapshot suitable for responsive interactive preview."""
    if maximum_cameras < 1 or maximum_points < 1:
        raise ValueError("preview maxima must be positive")
    directory = Path(model_dir)
    before = _model_fingerprint(directory)
    if before is None:
        return None
    stats = read_model_stats(directory)
    cameras = read_camera_positions(directory, maximum=maximum_cameras) if stats is not None else None
    points = read_sparse_points(directory, maximum=maximum_points) if stats is not None else None
    after = _model_fingerprint(directory)
    if before != after or stats is None or cameras is None or points is None:
        return None
    return ReconstructionPreview(cameras, points, stats.images, stats.points, _camera_ray_focus(cameras))


def _camera_ray_focus(cameras: list[CameraPosition]) -> tuple[float, float, float] | None:
    """Return the least-squares intersection of camera optical rays.

    Sparse-model world axes are arbitrary, but each COLMAP pose supplies an
    optical ray in that frame. The point closest to all sampled rays is a stable
    preview pivot and does not redefine or mutate the reconstruction origin.
    """
    if len(cameras) < 2:
        return None
    matrix = [[0.0] * 3 for _ in range(3)]
    vector = [0.0] * 3
    usable = 0
    for camera in cameras:
        direction = (camera.forward_x, camera.forward_y, camera.forward_z)
        length = sum(value * value for value in direction) ** 0.5
        if length < 1e-12:
            continue
        unit = tuple(value / length for value in direction)
        projector = [[(1.0 if row == column else 0.0) - unit[row] * unit[column] for column in range(3)] for row in range(3)]
        centre = (camera.x, camera.y, camera.z)
        for row in range(3):
            for column in range(3):
                matrix[row][column] += projector[row][column]
            vector[row] += sum(projector[row][column] * centre[column] for column in range(3))
        usable += 1
    if usable < 2:
        return None
    return _solve_3x3(matrix, vector)


def _solve_3x3(matrix: list[list[float]], vector: list[float]) -> tuple[float, float, float] | None:
    """Solve a small linear system, rejecting near-parallel/degenerate camera rays."""
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-10:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [value - factor * pivot_value for value, pivot_value in zip(augmented[row], augmented[column])]
    return tuple(augmented[row][3] for row in range(3))


def _camera_center(name: str, qw: float, qx: float, qy: float, qz: float, tx: float, ty: float, tz: float) -> CameraPosition:
    """Convert COLMAP world-to-camera qvec/tvec to a world-space camera centre."""
    # C = -R(q)^T t, expanded to avoid an optional numerical dependency.
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    # COLMAP stores world-to-camera R, t. The camera's forward optical axis in
    # world coordinates is Rᵀ·(0, 0, 1), the third row of R.
    return CameraPosition(
        name,
        -((1 - 2 * (yy + zz)) * tx + 2 * (xy + wz) * ty + 2 * (xz - wy) * tz),
        -(2 * (xy - wz) * tx + (1 - 2 * (xx + zz)) * ty + 2 * (yz + wx) * tz),
        -(2 * (xz + wy) * tx + 2 * (yz - wx) * ty + (1 - 2 * (xx + yy)) * tz),
        2 * (xz - wy),
        2 * (yz + wx),
        1 - 2 * (xx + yy),
    )


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
