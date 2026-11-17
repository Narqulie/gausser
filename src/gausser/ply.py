"""The PLY files gausser reads and writes: splat exports, xyz+rgb point clouds and OpenMVS dense clouds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .scene import GausserError

VERTICAL = "comment Vertical axis:"  # Brush's header line with its guess of the splat's up vector


def count(path: Path, element: str = "vertex") -> int:
    """The number of `element`s a PLY file declares."""
    with path.open("rb") as stream:
        for line in stream:
            if line.startswith(f"element {element} ".encode()):
                return int(line.split()[2])
            if line.strip() == b"end_header":
                break
    raise GausserError(f"{path.name} declares no {element} count")


def read(path: Path) -> tuple[list[str], Any]:
    """Header lines and the vertex array of a binary little-endian PLY with float properties only (a splat)."""
    import numpy as np

    with path.open("rb") as f:
        header = []
        while (line := f.readline().decode().strip()) != "end_header":
            if not line and not f.peek(1):
                raise GausserError(f"{path.name} is not a PLY file")
            header.append(line)
        vertices = next((int(h.split()[2]) for h in header if h.startswith("element vertex")), None)
        names = [h.split()[2] for h in header if h.startswith("property float")]
        if vertices is None or "format binary_little_endian 1.0" not in header or len(names) != sum(
            h.startswith("property") for h in header
        ):
            raise GausserError(f"{path.name}: only binary little-endian PLY with float properties is supported")
        return header, np.fromfile(f, dtype=np.dtype([(n, "<f4") for n in names]), count=vertices)


def write(path: Path, header: list[str], data: Any) -> None:
    lines = [f"element vertex {len(data)}" if h.startswith("element vertex") else h for h in header]
    with path.open("wb") as f:
        f.write(("\n".join(lines) + "\nend_header\n").encode())
        data.tofile(f)


def xyz(data: Any) -> Any:
    import numpy as np

    return np.stack([data["x"], data["y"], data["z"]], axis=1)


def opaque_centres(data: Any) -> Any:
    """Centres of the splats more than half opaque (stored opacity is a logit, so > 0)."""
    import numpy as np

    return xyz(data)[data["opacity"] > 0].astype(np.float64)


def write_points(path: Path, xyz: Any, rgb: Any) -> None:
    """A point cloud: float x, y, z and uchar red, green, blue."""
    import numpy as np

    points = np.zeros(len(xyz), dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    points["xyz"], points["rgb"] = xyz, rgb
    header = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    path.write_bytes(header.encode() + points.tobytes())


def read_points(path: Path, limit: int) -> tuple[Any, Any]:
    """Positions and colours of at most `limit` randomly chosen points of a `write_points` file."""
    import numpy as np

    data = path.read_bytes()
    start = data.index(b"end_header\n") + 11
    points = np.frombuffer(data, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)], offset=start)
    if len(points) > limit:
        points = points[np.sort(np.random.default_rng(0).choice(len(points), limit, replace=False))]
    return points["xyz"].astype(np.float64), points["rgb"].copy()


def read_dense_cloud(path: Path, limit: int) -> tuple[Any, Any]:
    """Positions and colours of at most `limit` randomly chosen points of an OpenMVS dense cloud.

    Its vertices end in variable-length lists (the views that saw them), so the records are walked
    once to find where each starts.
    """
    import mmap
    from array import array

    import numpy as np

    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        header = data[: data.find(b"end_header") + 11].decode()
        fields = [line.split()[-1] for line in header.splitlines() if line.startswith("property")]
        if fields[:6] != ["x", "y", "z", "red", "green", "blue"] or "binary_little_endian" not in header:
            raise GausserError(f"{path.name} is not an OpenMVS dense cloud")
        vertices = count(path)
        fixed = 15 + 12 * (fields[6:9] == ["nx", "ny", "nz"])  # xyz, rgb, optional normals
        lists = fields[9:] if fixed == 27 else fields[6:]
        starts = array("q", bytes(8 * vertices))
        offset = len(header)
        for index in range(vertices):
            starts[index] = offset
            offset += fixed
            for _ in lists:  # each list: a uint8 length and that many 4-byte values
                offset += 1 + 4 * data[offset]
        chosen = np.asarray(starts)
        if vertices > limit:
            chosen = np.sort(np.random.default_rng(0).choice(chosen, limit, replace=False))
        raw = np.frombuffer(data, dtype=np.uint8)[chosen[:, None] + np.arange(15)]
        xyz = raw[:, :12].copy().view(np.float32).astype(np.float64)
        rgb = raw[:, 12:15].copy()
        del raw
    return xyz, rgb
