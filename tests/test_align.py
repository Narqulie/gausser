from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from gausser import align, cli  # noqa: E402
from gausser.composite import read_ply, write_ply  # noqa: E402
from gausser.scene import GausserError  # noqa: E402

NAMES = ("x", "y", "z", "opacity")
HEADER = ["ply", "format binary_little_endian 1.0", "comment Vertical axis: 0 -1 0", "element vertex 0"] + [
    f"property float {name}" for name in NAMES
]


def random_rotation(rng) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.linalg.det(q)


def scene_points(rng) -> np.ndarray:
    """An asymmetric scene: a wide ground, a tall column, a lopsided block and a small ball."""
    ground = rng.uniform(-10, 10, (6000, 3)) * [1, 0.02, 1]
    column = rng.normal(size=(2500, 3)) * [0.5, 2.0, 0.5] + [2, 2, 1]
    block = rng.uniform(0, 1, (2000, 3)) * [4, 1, 2] + [-5, 0, -3]
    ball = rng.normal(size=(800, 3)) * 0.4 + [-1, 3, 4]
    return np.concatenate([ground, column, block, ball])


def splat(path: Path, points: np.ndarray, header: list[str] = HEADER) -> None:
    data = np.zeros(len(points), dtype=[(name, "<f4") for name in NAMES])
    for axis, name in enumerate("xyz"):
        data[name] = points[:, axis]
    data["opacity"] = 5.0
    write_ply(path, header, data)


def test_estimate_recovers_a_similarity_for_a_cropped_reference_in_an_arbitrary_frame():
    rng = np.random.default_rng(3)
    moving = scene_points(rng)
    r, s, t = random_rotation(rng), 0.37, np.array([4.0, -2.0, 7.5])
    # The reference is the part of the scene around the column and block (inside 7 units), minus 20% of those points.
    part = moving[np.linalg.norm(moving - [0, 1, 0], axis=1) < 7]
    part = part[rng.uniform(size=len(part)) > 0.2]
    reference = s * np.einsum("ij,nj->ni", r, part) + t + rng.normal(scale=0.002, size=part.shape)

    fit = align.estimate(moving, reference, random_count=100)

    assert fit.scale == pytest.approx(s, rel=0.01)
    assert np.abs(fit.rotation - r).max() < 0.01
    assert np.abs(fit.translation - t).max() < 0.05
    assert fit.inliers > 0.95


def test_estimate_without_scale_keeps_the_scale_at_one():
    rng = np.random.default_rng(5)
    moving = scene_points(rng)
    r, t = random_rotation(rng), np.array([-3.0, 1.0, 2.0])
    fit = align.estimate(moving, np.einsum("ij,nj->ni", r, moving[::2]) + t, with_scale=False, random_count=100)
    assert fit.scale == 1.0
    assert np.abs(fit.rotation - r).max() < 0.01


def test_euler_for_splat_transform_round_trips_through_the_cli_convention():
    rng = np.random.default_rng(1)
    d = np.diag([-1.0, -1.0, 1.0])
    for r in [random_rotation(rng) for _ in range(20)] + [np.eye(3), np.diag([1.0, -1, -1])]:
        assert np.abs(cli_rotation(*align.euler_for_splat_transform(r)) - d @ r @ d).max() < 1e-9


def cli_rotation(x: float, y: float, z: float) -> np.ndarray:
    """What `splat-transform -r x,y,z` does in its own frame: Rz·Ry·Rx."""
    a, b, c = np.radians([x, y, z])
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx


def fake_splat_transform(command: list[str], *, cwd: Path, env=None) -> None:
    """Stands in for splat-transform: a reference decode copies a prepared PLY, an action run applies -s -r -t."""
    *inputs, output = [a for a in command[1:] if a != "-w"]
    if "-s" not in inputs:
        Path(output).write_bytes(Path(cwd / "reference.source").read_bytes())
        return
    scale = float(inputs[inputs.index("-s") + 1])
    rotation = cli_rotation(*map(float, inputs[inputs.index("-r") + 1].split(",")))
    shift = np.array(inputs[inputs.index("-t") + 1].split(","), dtype=float)
    header, data = read_ply(Path(inputs[0]))
    points = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(float)
    d = np.diag([-1.0, -1.0, 1.0])
    moved = scale * np.einsum("ij,nj->ni", d @ rotation @ d, points) + d @ shift
    for axis, name in enumerate("xyz"):
        data[name] = moved[:, axis]
    write_ply(Path(output), [h for h in header if not h.startswith("comment")], data)


def test_align_command_writes_the_moved_splat_with_a_rotated_vertical_axis(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    moving = scene_points(rng)[::2]
    r, s, t = random_rotation(rng), 1.8, np.array([1.0, 2.0, -3.0])
    splat(tmp_path / "moving.ply", moving)
    splat(tmp_path / "ref.ply", s * np.einsum("ij,nj->ni", r, moving[np.linalg.norm(moving - [0, 1, 0], axis=1) < 7]) + t)
    monkeypatch.setattr(align, "require_tool", lambda name, hint: name)
    real_run = fake_splat_transform

    def run_tool(command: list[str], *, cwd: Path, env=None) -> None:
        (cwd / "reference.source").write_bytes((tmp_path / "ref.ply").read_bytes())
        real_run(command, cwd=cwd, env=env)

    monkeypatch.setattr(align, "run_tool", run_tool)
    out = tmp_path / "out" / "aligned.ply"

    assert cli.main(["align", str(tmp_path / "moving.ply"), "--to", str(tmp_path / "ref.ply"), "-o", str(out)]) == 0

    header, aligned = read_ply(out)
    moved = s * np.einsum("ij,nj->ni", r, moving) + t
    assert np.abs(np.stack([aligned["x"], aligned["y"], aligned["z"]], axis=1) - moved).max() < 0.01
    up = next(h for h in header if h.startswith("comment Vertical axis:"))
    assert np.allclose([float(v) for v in up.split(":")[1].split()], r @ [0, -1, 0], atol=0.01)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["moving.ply", "out", "ref.ply"]  # no staging left behind


def test_align_refuses_an_output_that_is_an_input(tmp_path, monkeypatch):
    splat(tmp_path / "a.ply", np.zeros((3, 3)))
    monkeypatch.setattr(align, "require_tool", lambda name, hint: name)
    with pytest.raises(GausserError, match="must not be one of the inputs"):
        align.align(tmp_path / "a.ply", [tmp_path / "a.ply"], tmp_path / "a.ply")
