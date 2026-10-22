"""`gausser composite`: put a masked statue splat into its unmasked context splat.

Both splats must come from the same COLMAP model (same coordinates). A masked Brush run on an
unmasked solve starts from the whole scene's COLMAP points and keeps opaque background splats, so
statue splats are first kept only if they project inside the SAM masks in most views (visual hull).
Context splats inside the statue's volume (its own blurrier copy) are then carved out and the
statue is appended.
"""

from __future__ import annotations

import shutil
import struct
from pathlib import Path
from typing import Any

from .scene import GausserError, Scene, replace


def read_colmap(model: Path) -> tuple[tuple[float, float, float, float], list[tuple[str, Any, Any]]]:
    """(fx, fy, cx, cy) of the single camera and [(image name, R, t)] from a binary COLMAP model."""
    import numpy as np

    with (model / "cameras.bin").open("rb") as f:
        f.read(8)
        _id, camera, _w, _h = struct.unpack("<iiQQ", f.read(24))
        count = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}.get(camera)  # SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV
        if count is None:
            raise GausserError(f"Unsupported COLMAP camera model {camera}")
        p = struct.unpack(f"<{count}d", f.read(8 * count))
        intrinsics = (p[0], p[0], p[1], p[2]) if camera in (0, 2, 3) else (p[0], p[1], p[2], p[3])
    images = []
    with (model / "images.bin").open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            _i, qw, qx, qy, qz, tx, ty, tz, _c = struct.unpack("<idddddddi", f.read(64))
            name = b""
            while (ch := f.read(1)) not in (b"\0", b""):
                name += ch
            (points,) = struct.unpack("<Q", f.read(8))
            f.seek(points * 24, 1)
            r = np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ])
            images.append((name.decode(), r, np.array([tx, ty, tz])))
    return intrinsics, images


def hull_keep(
    points: Any, scene: Scene, min_share: float = 0.8, min_views: int = 5, scale: int = 4, slack: int = 6,
) -> Any:
    """True for points inside the (dilated) Brush mask in >= ``min_share`` of the views that see them.

    All-black masks (frames dropped with `mask --miss drop`) say nothing and are skipped.
    Masks are shrunk by ``scale`` and dilated by ``slack`` px (at that size).
    """
    import numpy as np
    from PIL import Image, ImageFilter

    (fx, fy, cx, cy), images = read_colmap(scene.require_model())
    seen = np.zeros(len(points), int)
    inside = np.zeros(len(points), int)
    for name, r, t in images:
        mask_path = scene.masks / f"{Path(name).stem}.png"
        if not mask_path.is_file():
            continue
        small = Image.open(mask_path).convert("L").reduce(scale).point(lambda v: 255 if v > 127 else 0)
        if not small.getextrema()[1]:
            continue
        mask = np.asarray(small.filter(ImageFilter.MaxFilter(2 * slack + 1))) > 127
        cam = np.einsum("ij,nj->ni", r, points.astype(np.float64)) + t  # einsum: no spurious matmul warnings
        front = cam[:, 2] > 0.05
        z = np.where(front, cam[:, 2], 1.0)
        u = (fx * cam[:, 0] / z + cx) / scale
        v = (fy * cam[:, 1] / z + cy) / scale
        ok = front & (u >= 0) & (v >= 0) & (u < mask.shape[1]) & (v < mask.shape[0])
        seen += ok
        hit = np.zeros(len(points), bool)
        hit[ok] = mask[v[ok].astype(int), u[ok].astype(int)]
        inside += hit
    if not seen.any():
        raise GausserError("No usable masks: the visual hull needs masks/ from `gausser mask` (not all dropped)")
    return (seen >= min_views) & (inside >= min_share * seen)


def read_ply(path: Path) -> tuple[list[str], Any]:
    """Header lines and the vertex array of a binary little-endian PLY with float properties only."""
    import numpy as np

    with path.open("rb") as f:
        header = []
        while (line := f.readline().decode().strip()) != "end_header":
            if not line and not f.peek(1):
                raise GausserError(f"{path.name} is not a PLY file")
            header.append(line)
        count = next((int(h.split()[2]) for h in header if h.startswith("element vertex")), None)
        names = [h.split()[2] for h in header if h.startswith("property float")]
        if count is None or "format binary_little_endian 1.0" not in header or len(names) != sum(
            h.startswith("property") for h in header
        ):
            raise GausserError(f"{path.name}: only binary little-endian PLY with float properties is supported")
        return header, np.fromfile(f, dtype=np.dtype([(n, "<f4") for n in names]), count=count)


def write_ply(path: Path, header: list[str], data: Any) -> None:
    lines = [f"element vertex {len(data)}" if h.startswith("element vertex") else h for h in header]
    with path.open("wb") as f:
        f.write(("\n".join(lines) + "\nend_header\n").encode())
        data.tofile(f)


def xyz(data: Any) -> Any:
    import numpy as np

    return np.stack([data["x"], data["y"], data["z"]], axis=1)


def carve(
    statue: Any, context: Any, cells: int = 256, grow: int = 1, radius: float | None = None,
) -> Any:
    """Boolean keep-mask over ``context``: False inside the statue's voxel hull (and beyond ``radius`` heights)."""
    import numpy as np

    solid = xyz(statue)[1 / (1 + np.exp(-statue["opacity"])) > 0.5]
    if not len(solid):
        raise GausserError("The statue has no opaque splats to carve with")
    lo, hi = np.percentile(solid, 0.5, axis=0), np.percentile(solid, 99.5, axis=0)
    size = (hi - lo).max() / cells
    lo = lo - size * (grow + 2)
    shape = np.ceil((hi - lo) / size).astype(int) + grow + 3
    grid = np.zeros(shape, bool)
    grid[tuple(((solid - lo) / size).astype(int).clip(0, shape - 1).T)] = True
    for _ in range(grow):  # 6-neighbour dilation, one voxel per pass
        grown = grid.copy()
        for axis in range(3):
            for step in (1, -1):
                grown |= np.roll(grid, step, axis)  # the grid has a margin, so wrap-around never matters
        grid = grown

    points = xyz(context)
    idx = ((points - lo) / size).astype(int)
    within = np.all((idx >= 0) & (idx < shape), axis=1)
    inside = np.zeros(len(context), bool)
    inside[within] = grid[tuple(idx[within].T)]
    keep = ~inside
    if radius:
        centre, height = (lo + hi) / 2, (hi - lo).max()
        keep &= np.linalg.norm(points - centre, axis=1) < radius * height
    return keep


def newest_export(scene: Scene, marker: str) -> Path:
    """Newest export of the newest Brush run marked ``masked`` or ``unmasked`` by `gausser train`."""
    for run in sorted(scene.runs.glob("brush/*"), reverse=True):
        exports = sorted(
            (p for p in run.glob("exports/*.ply") if not p.name.startswith("._")),  # exFAT AppleDouble sidecars
            key=lambda path: path.stat().st_mtime,
        )
        if (run / marker).is_file() and exports:
            return exports[-1]
    role = "statue" if marker == "masked" else "context"
    flag = "" if marker == "masked" else " --no-masks"
    raise GausserError(
        f"No {marker} Brush run with an export found (runs trained before this version have no marker). "
        f"Train one with `gausser train{flag}` or pass --{role} PLY"
    )


def composite(
    scene: Scene, statue: Path | None, context: Path | None, *, radius: float | None, out: Path | None,
) -> Path:
    """Write runs/composite/<time>/composite.ply (or ``out``); read-only on the scene's other files."""
    import numpy as np

    statue = (statue or newest_export(scene, "masked")).expanduser().resolve()
    context = (context or newest_export(scene, "unmasked")).expanduser().resolve()
    for ply in (statue, context):
        if not ply.is_file():
            raise GausserError(f"No such file: {ply}")
    scene.require_model()
    if not scene.masks.is_dir():
        raise GausserError("No masks/ folder; the statue splat is filtered by the masks (`gausser mask`)")
    print(f"statue  {statue}\ncontext {context}")
    _, statue_data = read_ply(statue)
    header, context_data = read_ply(context)
    if statue_data.dtype != context_data.dtype:
        raise GausserError("The statue and context splats have different PLY layouts")
    hull = hull_keep(xyz(statue_data), scene)
    print(f"visual hull keeps {int(hull.sum()):,}/{len(hull):,} statue splats")
    statue_data = statue_data[hull]
    keep = carve(statue_data, context_data, radius=radius)
    print(f"context keeps {int(keep.sum()):,}/{len(keep):,} splats (carved {int((~keep).sum()):,})")

    staged = scene.staging("composite")
    write_ply(staged / "composite.ply", header, np.concatenate([context_data[keep], statue_data]))
    if out:
        out = out.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(staged / "composite.ply", out)  # not replace(): -o may be on another filesystem than the scene
        shutil.rmtree(staged)
    else:
        run = scene.new_run("composite")
        replace(staged, run)
        out = run / "composite.ply"
    print(f"Composite: {out}")
    return out
