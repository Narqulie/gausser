from __future__ import annotations

import struct
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
image_module = pytest.importorskip("PIL.Image")

from gausser import composite  # noqa: E402
from gausser.scene import GausserError, Scene  # noqa: E402

SIZE, FOCAL, DISTANCE, VIEWS = 256, 400.0, 5.0, 12
HEADER = ["ply", "format binary_little_endian 1.0", "comment Vertical axis: 0 -1 0", "element vertex 0"] + [
    f"property float {name}" for name in ("x", "y", "z", "opacity")
]


def ring_scene(root: Path) -> Scene:
    """A camera ring around the origin, each looking at it, with a disc mask around the image centre."""
    root.mkdir(exist_ok=True)
    scene = Scene(root)
    scene.masks.mkdir(parents=True)
    model = scene.sparse / "0"
    model.mkdir(parents=True)
    (model / "points3D.bin").write_bytes(b"")
    (model / "cameras.bin").write_bytes(
        struct.pack("<Q", 1) + struct.pack("<iiQQ", 1, 0, SIZE, SIZE) + struct.pack("<3d", FOCAL, SIZE / 2, SIZE / 2)
    )
    images = struct.pack("<Q", VIEWS)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    disc = ((xx - SIZE / 2) ** 2 + (yy - SIZE / 2) ** 2 < 20**2).astype(np.uint8) * 255
    for index in range(VIEWS):
        angle = 2 * np.pi * index / VIEWS  # rotation about y only: the camera stays DISTANCE from the origin
        images += struct.pack("<idddddddi", index, np.cos(angle / 2), 0, np.sin(angle / 2), 0, 0, 0, DISTANCE, 1)
        images += f"img{index}.jpg".encode() + b"\0" + struct.pack("<Q", 0)
        image_module.fromarray(disc, mode="L").save(scene.masks / f"img{index}.png")
    (model / "images.bin").write_bytes(images)
    return scene


def splat(path: Path, points: list[list[float]], opacity: float = 5.0) -> None:
    data = np.zeros(len(points), dtype=[(name, "<f4") for name in ("x", "y", "z", "opacity")])
    for axis, name in enumerate("xyz"):
        data[name] = [point[axis] for point in points]
    data["opacity"] = opacity
    composite.write_ply(path, HEADER, data)


def cube() -> list[list[float]]:
    """343 points on a lattice that includes the origin."""
    ticks = np.linspace(-0.3, 0.3, 7)
    return np.stack(np.meshgrid(ticks, ticks, ticks), axis=-1).reshape(-1, 3).tolist()


def test_hull_keep_keeps_the_centre_and_drops_a_point_outside_the_masks(tmp_path):
    scene = ring_scene(tmp_path)
    keep = composite.hull_keep(np.array([[0.0, 0, 0], [0, 1.0, 0]]), scene)
    assert keep.tolist() == [True, False]


def test_carve_drops_context_inside_the_statue_and_keeps_the_rest(tmp_path):
    splat(tmp_path / "s.ply", cube())
    splat(tmp_path / "c.ply", [[0, 0, 0], [10, 0, 0]])
    _, statue = composite.read_ply(tmp_path / "s.ply")
    _, context = composite.read_ply(tmp_path / "c.ply")
    assert composite.carve(statue, context, cells=16).tolist() == [False, True]


def test_newest_export_picks_runs_by_their_marker(tmp_path):
    scene = Scene(tmp_path)
    for name, marker in (("1", "masked"), ("2", "unmasked"), ("3", "masked"), ("4", "unmasked"), ("5", None)):
        run = scene.runs / "brush" / name
        (run / "exports").mkdir(parents=True)
        (run / "exports" / "export_2.ply").write_bytes(b"ply")
        if marker:
            (run / marker).touch()
    assert composite.newest_export(scene, "masked") == scene.runs / "brush" / "3" / "exports" / "export_2.ply"
    assert composite.newest_export(scene, "unmasked") == scene.runs / "brush" / "4" / "exports" / "export_2.ply"
    (tmp_path / "other").mkdir()
    with pytest.raises(GausserError, match="No masked Brush run"):
        composite.newest_export(Scene(tmp_path / "other"), "masked")


def test_composite_selects_runs_filters_the_statue_and_keeps_the_context_header(tmp_path):
    scene = ring_scene(tmp_path / "scene")
    for name, marker, points in (
        ("1", "masked", [*cube(), [0, 1.0, 0]]),  # the last statue splat is background garbage
        ("2", "unmasked", [[0, 0, 0], [10, 0, 0]]),
    ):
        run = scene.runs / "brush" / name
        (run / "exports").mkdir(parents=True)
        splat(run / "exports" / "export_1.ply", points)
        (run / marker).touch()

    out = composite.composite(scene, None, None, radius=None, out=tmp_path / "out" / "o.ply")

    header, data = composite.read_ply(out)
    assert header[2] == "comment Vertical axis: 0 -1 0"
    assert len(data) == 1 + 343  # the context splat outside the statue, plus the statue without its garbage splat
    assert not list(scene.runs.glob("composite/*"))  # -o writes only there
