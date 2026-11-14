"""`gausser crop`: a box around the subject that training and meshing keep to.

The box is axis-aligned in the model's level frame: COLMAP coordinates turned by the shortest rotation
that takes the model's up (`orbit.camera_up`) onto +y, the same turn the monitor's 3D view makes
(three.js `Quaternion.setFromUnitVectors`). It is stored with the model in sparse/crop.json, so a new
solve, which replaces sparse/, drops it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .scene import GausserError, Scene, read_json


def level_rotation(up: Any) -> Any:
    """3×3 rotation taking the unit vector `up` onto +y by the shortest arc."""
    import numpy as np

    u = np.asarray(up, dtype=float)
    u = u / np.linalg.norm(u)
    y = np.array([0.0, 1.0, 0.0])
    c = float(u @ y)
    if c < -1 + 1e-9:  # upside down: half a turn about x
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(u, y)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


@dataclass
class Crop:
    up: list[float]
    min: list[float]
    max: list[float]

    def inside(self, points: Any) -> Any:
        """Which of the N×3 model-coordinate `points` lie in the box."""
        import numpy as np

        level = np.asarray(points, dtype=float) @ level_rotation(self.up).T
        return ((level >= self.min) & (level <= self.max)).all(axis=1)


def load(scene: Scene) -> Crop | None:
    data = read_json(scene.crop)
    return Crop(data["up"], data["min"], data["max"]) if data else None


def save(scene: Scene, box: list[float], up: list[float] | None = None) -> Crop:
    """Store the box x,y,z,X,Y,Z (level frame); `up` defaults to the model's own."""
    from .orbit import camera_up

    model = scene.require_model()
    if len(box) != 6 or any(low >= high for low, high in zip(box[:3], box[3:])):
        raise GausserError("--box needs x,y,z,X,Y,Z with each minimum below its maximum")
    crop = Crop([float(v) for v in (up if up is not None else camera_up(model / "images.bin"))], box[:3], box[3:])
    scene.crop.write_text(json.dumps({"up": crop.up, "min": crop.min, "max": crop.max}) + "\n")
    return crop


def crop_splats(path: Path, crop: Crop) -> tuple[int, int]:
    """Keep only the splats of a PLY export whose centres lie in the box; returns (kept, total)."""
    from .composite import read_ply, write_ply, xyz

    header, data = read_ply(path)
    keep = crop.inside(xyz(data))
    write_ply(path, header, data[keep])
    return int(keep.sum()), len(data)


def crop_mesh(path: Path, crop: Crop) -> tuple[int, int]:
    """Keep the faces of an OpenMVS mesh PLY (float x,y,z; uchar/uint face lists) whose corners all lie in the box."""
    import numpy as np

    with path.open("rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise GausserError(f"{path.name} is not a PLY file")
            header += line
        text = header.decode()
        lines = text.splitlines()
        vertex_props = [line.split()[-1] for line in lines if line.startswith("property float")]
        if (
            "format binary_little_endian 1.0" not in text or vertex_props != ["x", "y", "z"]
            or "property list uint8 uint32 vertex_indices" not in text.replace("uchar", "uint8").replace(" uint ", " uint32 ")
        ):
            raise GausserError(f"{path.name}: expected an OpenMVS mesh (float x y z, uchar/uint face lists)")
        vertex_count = int(text.split("element vertex ")[1].split()[0])
        face_count = int(text.split("element face ")[1].split()[0])
        vertices = np.fromfile(f, dtype="<f4", count=vertex_count * 3).reshape(vertex_count, 3)
        faces = np.fromfile(f, dtype=np.dtype([("n", "u1"), ("i", "<u4", 3)]), count=face_count)
    if (faces["n"] != 3).any():
        raise GausserError(f"{path.name}: only triangle meshes are supported")
    triangles = faces["i"][crop.inside(vertices)[faces["i"]].all(axis=1)]
    used = np.unique(triangles)
    index = np.full(vertex_count, -1, dtype=np.int64)
    index[used] = np.arange(len(used))
    out = np.empty(len(triangles), dtype=faces.dtype)
    out["n"], out["i"] = 3, index[triangles]
    with path.open("wb") as f:
        f.write(
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(used)}\nproperty float x\nproperty float y\nproperty float z\n"
            f"element face {len(out)}\nproperty list uchar uint vertex_indices\nend_header\n".encode()
        )
        vertices[used].astype("<f4").tofile(f)
        out.tofile(f)
    return len(out), face_count
