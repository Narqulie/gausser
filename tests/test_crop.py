from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from gausser import cli, crop, steps
from gausser.composite import read_ply, write_ply
from gausser.scene import Scene

from test_gausser import make_scene, tools  # noqa: F401  (fixture)

UP = [0.0, 0.0, -1.0]  # a model whose up is -z: the level frame turns -z onto +y


def splat_ply(path: Path, points: list[list[float]]) -> None:
    names = ("x", "y", "z", "opacity")
    data = np.zeros(len(points), dtype=[(name, "<f4") for name in names])
    data["x"], data["y"], data["z"] = np.asarray(points, dtype=np.float32).T
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(points)}", *(f"property float {n}" for n in names)]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_ply(path, header, data)


def cropped_scene(root: Path) -> Scene:
    """A solved scene with three COLMAP points and a crop holding only the first (level y = -z)."""
    scene = make_scene(root)
    model = scene.sparse / "0"
    model.mkdir(parents=True)
    (model / "images.bin").write_bytes(struct.pack("<Q", 3))
    (model / "cameras.bin").write_bytes(b"")
    steps.write_points3d(model / "points3D.bin", np.array([[0.0, 0, -1], [0, 0, 5], [9, 0, 0]]), np.zeros((3, 3), np.uint8))
    assert cli.main(["crop", "-s", str(root), "--box=-1,0,-1,1,2,1", "--up=" + ",".join(map(str, UP))]) == 0
    return scene


def test_level_rotation_turns_up_onto_y_like_three_js():
    for up in ([0, 0, -1], [0.3, -0.5, 0.8], [0, 1, 0], [0, -1, 0]):
        r = crop.level_rotation(up)
        assert np.allclose(r @ r.T, np.eye(3)) and np.isclose(np.linalg.det(r), 1)
        assert np.allclose(r @ (np.array(up) / np.linalg.norm(up)), [0, 1, 0])
    # three.js setFromUnitVectors((0,0,-1), (0,1,0)) is a quarter turn about x: (0,1,0) goes to (0,0,1)
    assert np.allclose(crop.level_rotation([0, 0, -1]) @ [0, 1, 0], [0, 0, 1])


def test_crop_command_sets_shows_and_clears_and_makes_train_out_of_date(tmp_path, capsys):
    scene = cropped_scene(tmp_path)
    export = scene.runs / "brush" / "a" / "exports" / "export_10.ply"
    export.parent.mkdir(parents=True)
    export.write_bytes(b"ply")
    earlier = os.stat(scene.crop).st_mtime - 5
    os.utime(export, (earlier, earlier))

    assert json.loads(scene.crop.read_text()) == {"up": UP, "min": [-1.0, 0.0, -1.0], "max": [1.0, 2.0, 1.0]}
    assert not scene.train_done()
    assert cli.main(["crop", "-s", str(tmp_path)]) == 0
    assert "[-1.0, 0.0, -1.0] to [1.0, 2.0, 1.0]" in capsys.readouterr().out
    assert cli.main(["crop", "-s", str(tmp_path), "--clear"]) == 0
    assert not scene.crop.exists()
    assert cli.main(["crop", "-s", str(tmp_path), "--box=1,0,0,0,1,1", "--up=0,1,0"]) == 1  # min above max


def test_brush_starts_inside_the_crop_and_its_exports_keep_only_splats_in_it(tmp_path, tools):  # noqa: F811
    scene = cropped_scene(tmp_path)
    seen = {}

    def brush(command):
        data = (Path(command[1]) / "sparse" / "0" / "points3D.bin").read_bytes()
        seen["count"] = struct.unpack_from("<Q", data)[0]
        splat_ply(Path(command[command.index("--export-path") + 1]) / "export_10.ply", [[0, 0, -1], [0, 0, -5], [0, 3, 0]])

    tools.on_run(brush)
    run = steps.train_brush(scene, iterations=10)

    assert seen["count"] == 1  # (0, 0, -1) is level (0, 1, 0): inside; the others are not
    assert xyz_of(run / "exports" / "export_10.ply") == [[0, 0, -1]]
    assert scene.train_done()


def xyz_of(path: Path) -> list[list[float]]:
    _, data = read_ply(path)
    return np.stack([data["x"], data["y"], data["z"]], axis=1).tolist()


def test_crop_mesh_keeps_faces_with_every_corner_inside(tmp_path):
    vertices = np.array([[0, 0, -0.5], [0.5, 0, -0.5], [0, 0.5, -0.5], [0, 0, -9]], dtype="<f4")
    faces = np.zeros(2, dtype=[("n", "u1"), ("i", "<u4", 3)])
    faces["n"], faces["i"] = 3, [[0, 1, 2], [0, 1, 3]]
    path = tmp_path / "mesh.ply"
    with path.open("wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
                b"element face 2\nproperty list uchar uint vertex_indices\nend_header\n")
        vertices.tofile(f)
        faces.tofile(f)

    assert crop.crop_mesh(path, crop.Crop(UP, [-1, 0, -1], [1, 2, 1])) == (1, 2)
    assert steps.ply_count(path) == 3 and steps.ply_count(path, "face") == 1


@pytest.mark.parametrize("scores, final", [({10: 20.0, 15: 21.0, 20: 21.05}, 20), ({10: 20.0, 15: 19.5}, 10)])
def test_adaptive_training_continues_from_the_last_export_while_held_out_psnr_gains(tmp_path, tools, monkeypatch, scores, final):  # noqa: F811
    scene = make_scene(tmp_path)
    model = scene.sparse / "0"
    model.mkdir(parents=True)
    for name in ("images.bin", "cameras.bin", "points3D.bin"):
        (model / name).write_bytes(b"")
    monkeypatch.setattr(steps, "EXTEND_STEPS", 5)
    monkeypatch.setattr(steps, "eval_psnr", lambda scene, exports, step: scores[step])
    inits = []

    def brush(command):
        step = int(command[command.index("--total-train-iters") + 1])
        init = Path(command[1]) / "init.ply"
        inits.append(init.resolve().name if init.exists() else None)
        splat_ply(Path(command[command.index("--export-path") + 1]) / f"export_{step}.ply", [[0, 0, 0]])

    tools.on_run(brush)
    run = steps.train_brush(scene, iterations=10, adaptive=True, max_iterations=100)

    assert inits == [None, "export_10.ply", "export_15.ply"][: len(scores)]
    assert tools.commands[0][tools.commands[0].index("--eval-split-every") + 1] == "20"
    assert tools.commands[1][2:4] == ["--start-iter", "10"] and tools.commands[1][-4:] == ["--lr-mean", "2e-7", "--lr-mean-end", "1e-7"]
    assert scene.exports()[-1] == run / "exports" / f"export_{final}.ply"
