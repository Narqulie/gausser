"""`gausser align`: put a re-trained splat into the coordinate frame of an older, published one.

A fresh COLMAP solve lands anywhere (position, rotation, scale). Camera presets and hand-placed
geometry made for the published splat only fit if the new splat shares its frame. We estimate the
similarity x' = s·R·x + t that maps the opaque splat centres of MOVING onto REFERENCE, then let
`splat-transform` apply it (it also rotates quaternions and spherical harmonics).

The reference may be cropped or cleaned (a part of the moving scene) and neither frame has a known up
axis, so: try many rotations and scales, run a few trimmed ICP steps from each, keep the best starts by
truncated loss and refine them on more points. Reference points look up their nearest moving point
(never the other way round), so a moving cloud much larger than the reference cannot win by shrinking.
The Brush `Vertical axis` header of MOVING is not used for the search: the reference's up is unknown anyway.
"""

from __future__ import annotations

import itertools
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import ply
from .scene import GausserError, require_tool, run_tool

COARSE_POINTS, FINE_POINTS = (1_000, 20_000), (50_000, 150_000)  # (reference, moving) samples per stage
SCALE_STEPS = (0.6, 1.0, 1.7)  # around the ratio of the clouds' median radii


@dataclass
class Fit:
    scale: float
    rotation: Any  # 3x3
    translation: Any  # 3
    rms: float  # best 70% of reference points, reference units
    inliers: float  # share of reference points within 2% of the reference extent


def umeyama(src: Any, dst: Any, with_scale: bool = True) -> tuple[float, Any, Any]:
    """Least-squares (s, R, t) with dst ≈ s·R·src + t."""
    import numpy as np

    mean_src, mean_dst = src.mean(0), dst.mean(0)
    a, b = src - mean_src, dst - mean_dst
    u, sing, vt = np.linalg.svd(b.T @ a / len(src))
    flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(u) * np.linalg.det(vt))])
    r = u @ flip @ vt
    s = float((sing * flip.diagonal()).sum() / (a**2).sum(1).mean()) if with_scale else 1.0
    return s, r, mean_dst - s * r @ mean_src


def rotation_starts(random_count: int = 300) -> Any:
    """The 24 axis-aligned rotations plus seeded random ones, as an (n,3,3) array."""
    import numpy as np

    cube = [
        np.diag(signs) @ np.eye(3)[list(order)]
        for order in itertools.permutations(range(3)) for signs in itertools.product((1, -1), repeat=3)
    ]
    q = np.random.default_rng(0).normal(size=(random_count, 4))
    w, x, y, z = (q / np.linalg.norm(q, axis=1, keepdims=True)).T
    random = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], axis=1).reshape(-1, 3, 3)
    return np.concatenate([np.array([c for c in cube if np.linalg.det(c) > 0]), random])


class Matcher:
    """Residuals of REFERENCE points against a moving cloud under x' = s·R·x + t, in reference units."""

    def __init__(self, moving: Any, reference: Any, cap: float, with_scale: bool, reach: float = math.inf) -> None:
        from scipy.spatial import KDTree

        self.moving, self.reference, self.cap, self.with_scale, self.reach = moving, reference, cap, with_scale, reach
        self.tree = KDTree(moving, leafsize=64, balanced_tree=False, compact_nodes=False)  # builds fast, queries ~4x faster here

    def residuals(self, s: float, r: Any, t: Any) -> tuple[Any, Any]:
        # (q - t) @ R = Rᵀ(q - t). Far-away queries are slow, so ``reach`` caps the search; misses are inf.
        distance, index = self.tree.query((self.reference - t) @ r / s, distance_upper_bound=self.reach / s, workers=1 if len(self.reference) < 5000 else -1)
        return s * distance, index

    def loss(self, res: Any) -> float:
        import numpy as np

        return float((np.minimum(res, self.cap) ** 2).mean())

    def icp(self, start: tuple[float, Any, Any], steps: int, trim: float) -> tuple[float, tuple[float, Any, Any]]:
        """Trimmed ICP: each step refits to the best ``trim`` share of matches. Returns the lowest loss seen.

        Stops early once a step improves the loss by less than 0.1%.
        """
        import numpy as np

        pose, best = start, (math.inf, start)
        for _ in range(steps + 1):
            res, index = self.residuals(*pose)
            loss = self.loss(res)
            done = best[0] - loss < 1e-3 * loss
            if loss < best[0]:
                best = (loss, pose)
            if done:
                break
            keep = (res <= np.quantile(res, trim)) & np.isfinite(res)
            if keep.sum() < 10:
                break
            s, r, t = umeyama(self.moving[index[keep]], self.reference[keep], self.with_scale)
            if not (math.isfinite(s) and start[0] / 4 < s < start[0] * 4):  # a collapsing fit is a dead end
                break
            pose = (s, r, t)
        return best


def estimate(moving: Any, reference: Any, with_scale: bool = True, random_count: int = 300) -> Fit:
    """Fit x' = s·R·x + t from MOVING to REFERENCE centres (n,3 arrays; REFERENCE is a part of MOVING)."""
    import numpy as np

    if len(moving) < 100 or len(reference) < 100:
        raise GausserError("Too few opaque splats to align (need at least 100 in each)")
    with np.errstate(all="ignore"):  # Accelerate's matmul raises spurious FP warnings
        return _estimate(moving, reference, with_scale, random_count)


def _estimate(moving: Any, reference: Any, with_scale: bool, random_count: int) -> Fit:
    import numpy as np

    rng = np.random.default_rng(0)

    def sample(points: Any, count: int) -> Any:
        return points if len(points) <= count else points[rng.choice(len(points), count, replace=False)]

    centre_m, centre_r = np.median(moving, 0), np.median(reference, 0)
    extent = float(np.linalg.norm(np.percentile(reference, 99, 0) - np.percentile(reference, 1, 0)))
    cap = 0.05 * extent
    ratio = float(np.median(np.linalg.norm(reference - centre_r, axis=1)) / np.median(np.linalg.norm(moving - centre_m, axis=1)))
    scales = [ratio * f for f in SCALE_STEPS] if with_scale else [1.0]

    coarse = Matcher(sample(moving, COARSE_POINTS[1]), sample(reference, COARSE_POINTS[0]), cap, with_scale, 0.3 * extent)
    ranked: list[tuple[float, int, tuple[float, Any, Any]]] = []
    for r in rotation_starts(random_count):
        for s in scales:
            start = (s, r, centre_r - s * r @ centre_m)
            loss, pose = coarse.icp(start, steps=6, trim=0.8)
            ranked.append((loss, len(ranked), pose))
    ranked.sort(key=lambda item: item[:2])

    fine = Matcher(sample(moving, FINE_POINTS[1]), sample(reference, FINE_POINTS[0]), cap, with_scale)
    finalists = sorted((fine.icp(pose, steps=40, trim=0.8) for _, _, pose in ranked[:5]), key=lambda item: item[0])
    loss, pose = fine.icp(finalists[0][1], steps=150, trim=0.9)

    # Report on every reference point and all opaque moving splats.
    report = Matcher(moving, reference, cap, with_scale)
    res, _ = report.residuals(*pose)
    best = np.sort(res)[: int(0.7 * len(res))]
    return Fit(pose[0], pose[1], pose[2], float(np.sqrt((best**2).mean())), float((res < 0.02 * extent).mean()))


def euler_for_splat_transform(r: Any) -> tuple[float, float, float]:
    """`splat-transform -r x,y,z` degrees for the rotation ``r``.

    The CLI works in a frame turned 180° about z: it applies Rz(z)·Ry(y)·Rx(x) there, which in file
    coordinates is D·M·D with D = diag(-1,-1,1). Translations are negated in x and y the same way.
    """
    import numpy as np

    d = np.diag([-1.0, -1.0, 1.0])
    m = d @ r @ d
    y = -math.asin(max(-1.0, min(1.0, m[2, 0])))
    if abs(m[2, 0]) > 1 - 1e-9:  # gimbal lock: only x and z together matter
        return 0.0, math.degrees(y), math.degrees(math.atan2(-m[0, 1], m[1, 1]))
    return math.degrees(math.atan2(m[2, 1], m[2, 2])), math.degrees(y), math.degrees(math.atan2(m[1, 0], m[0, 0]))


def transform_args(fit: Fit) -> list[str]:
    """splat-transform actions for x' = s·R·x + t (actions run in order: scale, rotate, translate)."""
    x, y, z = euler_for_splat_transform(fit.rotation)
    tx, ty, tz = fit.translation
    return ["-s", f"{fit.scale:.9g}", "-r", f"{x:.9g},{y:.9g},{z:.9g}", "-t", f"{-tx:.9g},{-ty:.9g},{tz:.9g}"]


def align(moving: Path, references: list[Path], out: Path, *, with_scale: bool = True) -> Fit:
    """Write MOVING moved into the REFERENCE frame to ``out``; inputs stay untouched."""
    import numpy as np

    tool = require_tool("splat-transform")
    moving, out = moving.expanduser().resolve(), out.expanduser().resolve()
    references = [ref.expanduser().resolve() for ref in references]
    for path in (moving, *references):
        if not path.is_file():
            raise GausserError(f"No such file: {path}")
    if out in (moving, *references):
        raise GausserError("-o must not be one of the inputs")
    header, data = ply.read(moving)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".gausser-align-", dir=out.parent))  # same filesystem as ``out`` for the final move
    try:
        # splat-transform decodes SOG (log-encoded means) and LOD children itself; never decode them here.
        run_tool([tool, "-w", *map(str, references), str(work / "reference.ply")], cwd=work)
        _, reference = ply.read(work / "reference.ply")
        fit = estimate(ply.opaque_centres(data), ply.opaque_centres(reference), with_scale)
        print(f"scale {fit.scale:.6g}\nrotation\n{np.array2string(fit.rotation, precision=6, suppress_small=True)}")
        print(f"translation {np.array2string(fit.translation, precision=6)}")
        print(f"trimmed RMS {fit.rms:.4g}, inliers {fit.inliers:.1%} of reference splats (within 2% of its extent)")
        run_tool([tool, "-w", str(moving), *transform_args(fit), str(work / "aligned.ply")], cwd=work)
        aligned_header, aligned = ply.read(work / "aligned.ply")
        expected = fit.scale * np.einsum("ij,nj->ni", fit.rotation, ply.xyz(data).astype(np.float64)) + fit.translation  # einsum: no spurious matmul warnings
        error = float(np.abs(ply.xyz(aligned) - expected).max())
        if len(aligned) != len(data) or error > 1e-4 * max(1.0, float(np.abs(expected).max())):
            raise GausserError(f"splat-transform moved the splats differently than planned (max error {error:.3g})")
        print(f"splat-transform matches the fit (max error {error:.2g})")
        up = next((h for h in header if h.startswith(ply.VERTICAL)), None)
        if up:  # splat-transform drops comments; `gausser orbit` needs the rotated up vector
            vector = fit.rotation @ np.array(up[len(ply.VERTICAL):].split(), dtype=float)
            after_format = 1 + next(i for i, h in enumerate(aligned_header) if h.startswith("format"))
            aligned_header.insert(after_format, f"{ply.VERTICAL} {' '.join(f'{v:g}' for v in vector)}")
        ply.write(work / "final.ply", aligned_header, aligned)
        shutil.move(work / "final.ply", out)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"Aligned: {out}")
    return fit
