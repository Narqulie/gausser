from __future__ import annotations

import struct
import sys
from types import SimpleNamespace

import pytest

from gausser.scene import ScenePaths


_PARAMETER_SHAPES = (
    (1, 3),
    (1, 3),
    (1, 4),
    (1, 3),
    (1, 15, 3),
    (1, 1),
)


def _checkpoint_bytes(iteration: int) -> bytes:
    payload = bytearray(
        struct.pack(
            "<IIIIII8f",
            0x4C50534D,
            1,
            iteration,
            1,
            0,
            iteration,
            *([0.001] * 8),
        )
    )
    for shape in _PARAMETER_SHAPES * 3:
        byte_count = 4
        for dimension in shape:
            byte_count *= dimension
        payload.extend(struct.pack("<I", len(shape)))
        payload.extend(struct.pack(f"<{len(shape)}q", *shape))
        payload.extend(struct.pack("<Q", byte_count))
        payload.extend(b"\0" * byte_count)
    return bytes(payload)


def test_msplat_runtime_check_uses_current_interpreter_without_scene_mutation(
    monkeypatch, tmp_path, capsys
):
    from msplatter import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(
        sys.modules, "msplat", SimpleNamespace(__version__="1.1.4")
    )

    cli.main(["--check-runtime"])

    assert "runtime import is ready" in capsys.readouterr().out
    assert not (tmp_path / ".gausser").exists()


@pytest.mark.parametrize("runtime_error", [OSError("linker failed"), RuntimeError("load failed")])
def test_msplat_runtime_check_reports_native_import_failures(
    monkeypatch, tmp_path, capsys, runtime_error
):
    from msplatter import cli

    monkeypatch.chdir(tmp_path)

    def fail_runtime_check():
        raise runtime_error

    monkeypatch.setattr(cli, "_load_msplat", fail_runtime_check)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--check-runtime"])

    assert exc_info.value.code == 1
    assert str(runtime_error) in capsys.readouterr().out
    assert not (tmp_path / ".gausser").exists()


def test_msplat_requires_explicit_unmasked_consent(monkeypatch, scene):
    from msplatter import cli

    (scene / "masks").mkdir()
    (scene / "masks" / "frame.png").write_bytes(b"mask")
    monkeypatch.setitem(sys.modules, "msplat", object())

    with pytest.raises(SystemExit) as exc_info:
        cli.main([str(scene)])

    assert exc_info.value.code == 2


def test_msplat_writes_isolated_export_and_manifest(monkeypatch, scene, capsys):
    from msplatter import cli

    calls: list[object] = []

    class Trainer:
        def __init__(self, dataset, config):
            calls.extend([dataset, config])
            self.iteration = 0
            self.splat_count = 1

        def step(self):
            self.iteration += 1
            self.splat_count = 42
            return SimpleNamespace(iteration=self.iteration, splat_count=self.splat_count)

        def export_ply(self, path):
            path = __import__("pathlib").Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("ply")

        def save_checkpoint(self, path):
            __import__("pathlib").Path(path).write_bytes(
                _checkpoint_bytes(self.iteration)
            )

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
        __version__="1.1.4",
        load_dataset=load_dataset,
        TrainingConfig=lambda **kwargs: kwargs,
        GaussianTrainer=Trainer,
    )
    monkeypatch.setitem(sys.modules, "msplat", fake_msplat)
    monkeypatch.setattr(ScenePaths, "new_run_id", lambda _: "test-run")

    cli.main([str(scene), "--iterations", "10", "--downscale-factor", "4", "--allow-unmasked"])

    output = capsys.readouterr().out
    assert "◈ MSPLATTER // EXPERIMENTAL UNMASKED TRAINING SIGNAL" in output
    assert "◆ UNMASKED TRAINING COMPLETE" in output
    assert (scene / "runs" / "msplat" / "test-run" / "exports" / "scene.ply").is_file()
    run_dir = scene / "runs" / "msplat" / "test-run"
    assert (run_dir / "exports" / "scene.ply").is_file()
    assert (run_dir / "checkpoint.msplat").is_file()
    staged_dataset = scene / ".gausser" / "staging" / "msplat-test-run" / "run" / "dataset"
    assert calls[0][0] == str(staged_dataset)
    assert load_calls == [str(staged_dataset), 4.0, True]
    assert (run_dir / "dataset" / "frame_0001.jpg").exists()
    assert calls[1] == {
        "iterations": 10,
        "num_downscales": 0,
        "downscale_factor": 4.0,
    }
