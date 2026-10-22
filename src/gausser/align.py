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

from .composite import read_ply, write_ply, xyz
from .scene import GausserError, require_tool, run_tool

SPLAT_TRANSFORM_HINT = "npm install -g @playcanvas/splat-transform"
VERTICAL = "comment Vertical axis:"
COARSE_POINTS, FINE_POINTS = (1_000, 20_000), (50_000, 150_000)  # (reference, moving) samples per stage
SCALE_STEPS = (0.6, 1.0, 1.7)  # around the ratio of the clouds' median radii
PART_SCALE_STEPS = (0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.1, 3.9, 4.9, 6.1, 7.6, 9.5, 12.0)  # the part is smaller than the whole, so true scale is >= the radii ratio
ANCHORS, PROBES, WHOLE_POINTS = 600, 40, 30_000  # part-mode coarse search: candidate whole points, part points scored, whole sample


@dataclass
class Fit:
    scale: float
    rotation: Any  # 3x3
    translation: Any  # 3
    rms: float  # best 70% of reference points, reference units
    inliers: float  # share of reference points within 2% of the reference extent


def opaque_centres(data: Any) -> Any:
    import numpy as np

    return xyz(data)[1 / (1 + np.exp(-data["opacity"])) > 0.5].astype(np.float64)


def umeyama(src: Any, dst: Any, with_scale: bool = True, fixed_scale: float | None = None) -> tuple[float, Any, Any]:
    """Least-squares (s, R, t) with dst ≈ s·R·src + t; ``fixed_scale`` keeps s as given."""
    import numpy as np

    mean_src, mean_dst = src.mean(0), dst.mean(0)
    a, b = src - mean_src, dst - mean_dst
    u, sing, vt = np.linalg.svd(b.T @ a / len(src))
    flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(u) * np.linalg.det(vt))])
    r = u @ flip @ vt
    s = fixed_scale or (float((sing * flip.diagonal()).sum() / (a**2).sum(1).mean()) if with_scale else 1.0)
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

    def __init__(self, moving: Any, reference: Any, cap: float, with_scale: bool, reach: float = math.inf, fixed_scale: bool = False) -> None:
        from scipy.spatial import cKDTree

        self.moving, self.reference, self.cap, self.with_scale, self.reach = moving, reference, cap, with_scale, reach
        self.fixed_scale = fixed_scale  # ICP keeps each start's scale: a free scale drifts into wrong basins on a dense whole
        self.tree = cKDTree(moving, leafsize=64, balanced_tree=False, compact_nodes=False)  # builds fast, queries ~4x faster here

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
            s, r, t = umeyama(self.moving[index[keep]], self.reference[keep], self.with_scale, start[0] if self.fixed_scale else None)
            if not (math.isfinite(s) and start[0] / 4 < s < start[0] * 4):  # a collapsing fit is a dead end
                break
            pose = (s, r, t)
        return best


def _anchored_starts(coarse: Matcher, reference: Any, centre_r: Any, scales: list[float], rotations: Any, cap: float, rng: Any, keep: int = 150) -> list[tuple[float, Any, Any]]:
    """Starts for a reference much smaller than the moving cloud.

    For each (rotation, scale), put the reference point nearest its median on each of ANCHORS moving points and score
    PROBES reference points with a loose cap; the best anchor per (rotation, scale) gives a start, the best ``keep`` are returned.
    """
    import numpy as np

    anchors = coarse.moving[rng.choice(len(coarse.moving), min(ANCHORS, len(coarse.moving)), replace=False)]
    probes = reference[rng.choice(len(reference), min(PROBES, len(reference)), replace=False)]
    p0 = reference[np.argmin(np.linalg.norm(reference - centre_r, axis=1))]
    scored = []
    for r in rotations:
        for s in scales:
            queries = anchors[:, None, :] + (probes - p0) @ r / s  # Rᵀ(p - p0)/s around each anchor, in moving units
            distance, _ = coarse.tree.query(queries.reshape(-1, 3), distance_upper_bound=cap / s, workers=-1)
            loss = (np.minimum(s * distance.reshape(len(anchors), -1), cap) ** 2).mean(1)
            best = int(loss.argmin())
            scored.append((float(loss[best]), len(scored), (s, r, p0 - s * r @ anchors[best])))
    scored.sort(key=lambda item: item[:2])
    return [item[2] for item in scored[:keep]]


def estimate(moving: Any, reference: Any, with_scale: bool = True, random_count: int = 300, anchored: bool = False) -> Fit:
    """Fit x' = s·R·x + t from MOVING to REFERENCE centres (n,3 arrays; REFERENCE is a part of MOVING).

    ``anchored``: REFERENCE may be much smaller than MOVING, so the median-to-median start is useless;
    also try putting one reference point on each of many MOVING points (see `_anchored_starts`).
    """
    import numpy as np

    if len(moving) < 100 or len(reference) < 100:
        raise GausserError("Too few opaque splats to align (need at least 100 in each)")
    with np.errstate(all="ignore"):  # Accelerate's matmul raises spurious FP warnings
        return _estimate(moving, reference, with_scale, random_count, anchored)


def _estimate(moving: Any, reference: Any, with_scale: bool, random_count: int, anchored: bool) -> Fit:
    import numpy as np

    rng = np.random.default_rng(0)

    def sample(points: Any, count: int) -> Any:
        return points if len(points) <= count else points[rng.choice(len(points), count, replace=False)]

    centre_m, centre_r = np.median(moving, 0), np.median(reference, 0)
    extent = float(np.linalg.norm(np.percentile(reference, 99, 0) - np.percentile(reference, 1, 0)))
    cap = 0.05 * extent
    ratio = float(np.median(np.linalg.norm(reference - centre_r, axis=1)) / np.median(np.linalg.norm(moving - centre_m, axis=1)))
    scales = [ratio * f for f in (PART_SCALE_STEPS if anchored else SCALE_STEPS)] if with_scale else [1.0]

    coarse = Matcher(sample(moving, WHOLE_POINTS if anchored else COARSE_POINTS[1]), sample(reference, COARSE_POINTS[0]), cap, with_scale, 0.3 * extent, fixed_scale=anchored)
    ranked: list[tuple[float, int, tuple[float, Any, Any]]] = []
    starts = _anchored_starts(coarse, reference, centre_r, scales, rotation_starts(random_count), 0.2 * extent, rng) if anchored else (
        (s, r, centre_r - s * r @ centre_m) for r in rotation_starts(random_count) for s in scales
    )
    for start in starts:
        loss, pose = coarse.icp(start, steps=20 if anchored else 6, trim=0.8)
        ranked.append((loss, len(ranked), pose))
    ranked.sort(key=lambda item: item[:2])

    fine = Matcher(sample(moving, FINE_POINTS[1]), sample(reference, FINE_POINTS[0]), cap, with_scale)
    finalists = sorted((fine.icp(pose, steps=40, trim=0.8) for _, _, pose in ranked[:15 if anchored else 5]), key=lambda item: item[0])
    loss, pose = fine.icp(finalists[0][1], steps=150, trim=0.9)

    # Report on every reference point and all opaque moving splats.
    report = Matcher(moving, reference, cap, with_scale)
    res, _ = report.residuals(*pose)
    best = np.sort(res)[: int(0.7 * len(res))]
    return Fit(pose[0], pose[1], pose[2], float(np.sqrt((best**2).mean())), float((res < 0.02 * extent).mean()))


def estimate_part(part: Any, whole: Any, with_scale: bool = True, random_count: int = 300) -> Fit:
    """Fit x' = s·R·x + t from PART to WHOLE when PART is a piece of WHOLE (the reverse of `estimate`).

    The fit runs whole→part, so the part's points look up their nearest whole point, then inverts it.
    ``rms`` is converted to whole units; ``inliers`` counts the part's points within 2% of the part's extent.
    """
    inverse = estimate(whole, part, with_scale, random_count, anchored=True)
    s = 1 / inverse.scale
    r = inverse.rotation.T
    return Fit(s, r, -s * r @ inverse.translation, inverse.rms * s, inverse.inliers)


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


def align(moving: Path, references: list[Path], out: Path, *, with_scale: bool = True, moving_is_part: bool = False) -> Fit:
    """Write MOVING moved into the REFERENCE frame to ``out``; inputs stay untouched."""
    import numpy as np

    tool = require_tool("splat-transform", SPLAT_TRANSFORM_HINT)
    moving, out = moving.expanduser().resolve(), out.expanduser().resolve()
    references = [ref.expanduser().resolve() for ref in references]
    for path in (moving, *references):
        if not path.is_file():
            raise GausserError(f"No such file: {path}")
    if out in (moving, *references):
        raise GausserError("-o must not be one of the inputs")
    header, data = read_ply(moving)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".gausser-align-", dir=out.parent))  # same filesystem as ``out`` for the final move
    try:
        # splat-transform decodes SOG (log-encoded means) and LOD children itself; never decode them here.
        run_tool([tool, "-w", *map(str, references), str(work / "reference.ply")], cwd=work)
        _, reference = read_ply(work / "reference.ply")
        fit = (estimate_part if moving_is_part else estimate)(opaque_centres(data), opaque_centres(reference), with_scale)
        print(f"scale {fit.scale:.6g}\nrotation\n{np.array2string(fit.rotation, precision=6, suppress_small=True)}")
        print(f"translation {np.array2string(fit.translation, precision=6)}")
        print(f"trimmed RMS {fit.rms:.4g}, inliers {fit.inliers:.1%} of {'moving' if moving_is_part else 'reference'} splats (within 2% of its extent)")
        run_tool([tool, "-w", str(moving), *transform_args(fit), str(work / "aligned.ply")], cwd=work)
        aligned_header, aligned = read_ply(work / "aligned.ply")
        expected = fit.scale * np.einsum("ij,nj->ni", fit.rotation, xyz(data).astype(np.float64)) + fit.translation  # einsum: no spurious matmul warnings
        error = float(np.abs(xyz(aligned) - expected).max())
        if len(aligned) != len(data) or error > 1e-4 * max(1.0, float(np.abs(expected).max())):
            raise GausserError(f"splat-transform moved the splats differently than planned (max error {error:.3g})")
        print(f"splat-transform matches the fit (max error {error:.2g})")
        up = next((h for h in header if h.startswith(VERTICAL)), None)
        if up:  # splat-transform drops comments; `gausser orbit` needs the rotated up vector
            vector = fit.rotation @ np.array(up[len(VERTICAL):].split(), dtype=float)
            after_format = 1 + next(i for i, h in enumerate(aligned_header) if h.startswith("format"))
            aligned_header.insert(after_format, f"{VERTICAL} {' '.join(f'{v:g}' for v in vector)}")
        write_ply(work / "final.ply", aligned_header, aligned)
        shutil.move(work / "final.ply", out)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"Aligned: {out}")
    return fit
