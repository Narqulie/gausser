from __future__ import annotations

import json
import struct
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from gausser import cli, orbit
from gausser.scene import GausserError, Scene

PLY_HEADER = b"ply\nformat binary_little_endian 1.0\ncomment Vertical axis: 0 0 1\nelement vertex 0\nend_header\n"


def quaternion(r):
    """(w, x, y, z) of a rotation matrix, via the largest-diagonal branch."""
    t = np.trace(r)
    if t > 0:
        s = 2 * np.sqrt(t + 1)
        return s / 4, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2 * np.sqrt(1 + r[i, i] - r[j, j] - r[k, k])
    q = np.zeros(4)
    q[1 + i] = s / 4
    q[0] = (r[k, j] - r[j, k]) / s
    q[1 + j] = (r[j, i] + r[i, j]) / s
    q[1 + k] = (r[k, i] + r[i, k]) / s
    return tuple(q)


def write_ring(images_bin: Path, target, radius: float, height: float, count: int = 24) -> None:
    """Cameras on a circle round `target` (z up), each looking at it, in COLMAP's images.bin layout."""
    target = np.array(target, dtype=float)
    with images_bin.open("wb") as f:
        f.write(struct.pack("<Q", count))
        for i in range(count):
            angle = 2 * np.pi * i / count
            centre = target + [radius * np.cos(angle), radius * np.sin(angle), height]
            z = target - centre
            z /= np.linalg.norm(z)
            x = np.cross(z, [0, 0, 1.0])
            x /= np.linalg.norm(x)
            r = np.array([x, np.cross(z, x), z])  # world → camera rotation, rows are camera axes
            t = -r @ centre
            f.write(struct.pack("<idddddddi", i, *quaternion(r), *t, 1))
            f.write(f"img{i}.jpg\0".encode() + struct.pack("<Q", 2) + bytes(48))


def test_orbit_params_recover_centre_radius_and_height(tmp_path):
    write_ring(tmp_path / "images.bin", [1.0, -2.0, 0.5], radius=4.0, height=1.5)

    params = orbit.orbit_params(tmp_path / "images.bin", np.array([0.0, 0.0, 1.0]), zoom=1.0, drop=0.0)

    assert params["target"] == pytest.approx([1.0, -2.0, 0.5], abs=1e-6)
    assert params["radius"] == pytest.approx(4.0)
    assert params["height"] == pytest.approx(1.5)


def test_orbit_params_zoom_and_drop_keep_the_camera_height(tmp_path):
    write_ring(tmp_path / "images.bin", [0.0, 0.0, 0.0], radius=4.0, height=1.5)

    params = orbit.orbit_params(tmp_path / "images.bin", np.array([0.0, 0.0, 1.0]), zoom=1.5, drop=0.5)

    assert params["radius"] == pytest.approx(6.0)
    assert params["target"][2] == pytest.approx(-0.5)
    assert params["target"][2] + params["height"] == pytest.approx(1.5)  # camera stays where it was


def test_vertical_axis_needs_the_brush_header(tmp_path):
    (tmp_path / "a.ply").write_bytes(PLY_HEADER)
    (tmp_path / "b.ply").write_bytes(b"ply\nend_header\n")

    assert list(orbit.vertical_axis(tmp_path / "a.ply")) == [0, 0, 1]
    with pytest.raises(GausserError, match="Vertical axis"):
        orbit.vertical_axis(tmp_path / "b.ply")


@pytest.fixture
def orbit_scene(tmp_path, monkeypatch):
    (tmp_path / "scene").mkdir()
    scene = Scene(tmp_path / "scene")
    model = scene.sparse / "0"
    model.mkdir(parents=True)
    write_ring(model / "images.bin", [0, 0, 0], radius=4.0, height=1.5)
    for name in ("cameras.bin", "points3D.bin"):
        (model / name).write_bytes(b"")
    export = scene.runs / "brush" / "a" / "exports" / "export_1.ply"
    export.parent.mkdir(parents=True)
    export.write_bytes(PLY_HEADER)

    dist, node_dir = tmp_path / "dist", tmp_path / "node"
    dist.mkdir()
    (dist / "index.html").write_text("<html>")
    (node_dir / "node_modules" / "playwright-core").mkdir(parents=True)
    monkeypatch.setenv("GAUSSER_SUPERSPLAT", str(dist))
    monkeypatch.setenv("GAUSSER_NODE_DIR", str(node_dir))
    monkeypatch.setattr(orbit, "require_tool", lambda name, hint: name)
    return scene, export


def test_orbit_command_runs_the_driver_and_publishes_the_video_in_a_run(orbit_scene, monkeypatch):
    scene, export = orbit_scene
    seen: dict = {}

    def run_tool(command, *, cwd, env=None):
        seen.update(command=command, cwd=cwd, env=env)
        base = command[2].split("/?load=")[0]
        seen["served"] = urllib.request.urlopen(base + "/_gausser/export_1.ply").read()
        seen["params"] = json.loads(Path(command[3]).read_text())
        Path(command[4]).write_bytes(b"mp4")

    monkeypatch.setattr(orbit, "run_tool", run_tool)

    assert cli.main(["orbit", "-s", str(scene.root), "--seconds", "5", "--size", "1280x720"]) == 0

    command = seen["command"]
    assert command[:2] == ["node", str(orbit.DRIVER)]
    assert command[2].startswith("http://127.0.0.1:") and command[2].endswith("/?load=/_gausser/export_1.ply")
    assert command[5:] == ["5.0", "1280", "720"]
    assert seen["env"] == {"GAUSSER_NODE_DIR": str(orbit.node_dir())}
    assert seen["served"] == PLY_HEADER
    assert seen["params"]["radius"] == pytest.approx(4.0 * 1.3)
    (run,) = (scene.runs / "orbit").iterdir()
    assert (run / "orbit.mp4").read_bytes() == b"mp4"
    assert not list(scene.tmp.glob("orbit-*"))
    with pytest.raises(urllib.error.URLError):  # the server stopped with the command
        urllib.request.urlopen(seen["command"][2])


def test_orbit_output_option_writes_only_that_file(orbit_scene, monkeypatch, tmp_path):
    scene, export = orbit_scene
    monkeypatch.setattr(orbit, "run_tool", lambda command, *, cwd, env=None: Path(command[4]).write_bytes(b"mp4"))

    assert cli.main(["orbit", str(export), "-s", str(scene.root), "-o", str(tmp_path / "out" / "x.mp4")]) == 0

    assert (tmp_path / "out" / "x.mp4").read_bytes() == b"mp4"
    assert not (scene.runs / "orbit").exists()
    assert not list(scene.tmp.glob("orbit-*"))


def test_orbit_without_a_model_or_playwright_fails_clearly(orbit_scene, monkeypatch, capsys):
    scene, _ = orbit_scene
    monkeypatch.setattr(orbit, "run_tool", lambda *args, **kwargs: pytest.fail("must not run"))

    monkeypatch.setenv("GAUSSER_NODE_DIR", str(scene.root / "nowhere"))
    assert cli.main(["orbit", "-s", str(scene.root)]) == 1
    assert "playwright-core" in capsys.readouterr().err

    monkeypatch.undo()
    monkeypatch.setattr(orbit, "require_tool", lambda name, hint: name)
    monkeypatch.setenv("GAUSSER_NODE_DIR", str(scene.root.parent / "node"))
    (scene.sparse / "0" / "images.bin").unlink()
    assert cli.main(["orbit", "-s", str(scene.root)]) == 1
    assert "gausser solve" in capsys.readouterr().err
