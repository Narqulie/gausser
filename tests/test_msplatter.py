from __future__ import annotations

import struct
import sys
from types import SimpleNamespace

import pytest

from gausser.scene import ScenePaths


def test_msplat_requires_explicit_unmasked_consent(monkeypatch, scene):
    from msplatter import cli

    (scene / "masks").mkdir()
    (scene / "masks" / "frame.png").write_bytes(b"mask")
    monkeypatch.setitem(sys.modules, "msplat", object())

    with pytest.raises(SystemExit) as exc_info:
        cli.main([str(scene)])

    assert exc_info.value.code == 2


def test_msplat_writes_isolated_export_and_manifest(monkeypatch, scene):
    from msplatter import cli

    calls: list[object] = []

    class Trainer:
        def __init__(self, dataset, config):
            calls.extend([dataset, config])

        def train(self, callback, *, callback_every):
            callback(SimpleNamespace(iteration=10, splat_count=42))

        def export_ply(self, path):
            path = __import__("pathlib").Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("ply")

        def save_checkpoint(self, path):
            __import__("pathlib").Path(path).write_text("checkpoint")

        def evaluate(self):
            return {"psnr": 24.5, "ssim": 0.8}

    model = scene / "sparse" / "0"
    model.mkdir(parents=True)
    (model / "cameras.bin").write_bytes(struct.pack("<QIiQQ4d", 1, 1, 1, 1, 1, 1.0, 1.0, 0.5, 0.5))
    (model / "images.bin").write_bytes(
        struct.pack("<Q", 1)
        + struct.pack("<I7dI", 1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1)
        + b"frame_0001.jpg\0" + struct.pack("<Q", 0)
    )
    (model / "points3D.bin").write_bytes(
        struct.pack("<Q", 1) + struct.pack("<Q3d3BdQ", 1, 0.0, 0.0, 1.0, 255, 255, 255, 0.1, 0)
    )

    load_calls: list[object] = []

    def load_dataset(path, *, downscale_factor, eval_mode):
        load_calls.extend([path, downscale_factor, eval_mode])
        return (path, downscale_factor, eval_mode)

    fake_msplat = SimpleNamespace(
        load_dataset=load_dataset,
        TrainingConfig=lambda **kwargs: kwargs,
        GaussianTrainer=Trainer,
    )
    monkeypatch.setitem(sys.modules, "msplat", fake_msplat)
    monkeypatch.setattr(ScenePaths, "new_run_id", lambda _: "test-run")

    cli.main([str(scene), "--iterations", "10", "--downscale-factor", "4", "--allow-unmasked"])

    assert (scene / "runs" / "msplat" / "test-run" / "exports" / "scene.ply").is_file()
    run_dir = scene / "runs" / "msplat" / "test-run"
    assert (run_dir / "exports" / "scene.ply").is_file()
    assert (run_dir / "checkpoint.msplat").is_file()
    assert calls[0][0] == str(run_dir / "dataset")
    assert load_calls == [str(run_dir / "dataset"), 4.0, True]
    assert (run_dir / "dataset" / "frame_0001.jpg").exists()
    assert calls[1]["iterations"] == 10
