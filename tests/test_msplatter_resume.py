from __future__ import annotations

import json
from pathlib import Path
import signal
import struct
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from gausser.scene import ScenePaths
from msplatter.checkpoint import CheckpointFormatError, inspect_checkpoint


_MAGIC = 0x4C50534D
_VERSION = 1
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
            _MAGIC,
            _VERSION,
            iteration,
            1,
            min(3, iteration // 1000),
            iteration,
            *([0.001] * 8),
        )
    )
    for shape in _PARAMETER_SHAPES * 3:
        size = 4
        for dimension in shape:
            size *= dimension
        payload.extend(struct.pack("<I", len(shape)))
        payload.extend(struct.pack(f"<{len(shape)}q", *shape))
        payload.extend(struct.pack("<Q", size))
        payload.extend(b"\0" * size)
    return bytes(payload)


def _write_checkpoint(path: Path, iteration: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_checkpoint_bytes(iteration))


def _prepare_binary_model(scene: Path) -> None:
    model = scene / "sparse" / "0"
    model.mkdir(parents=True, exist_ok=True)
    (model / "cameras.bin").write_bytes(
        struct.pack("<QIiQQ4d", 1, 1, 1, 1, 1, 1.0, 1.0, 0.5, 0.5)
    )
    (model / "images.bin").write_bytes(
        struct.pack("<Q", 1)
        + struct.pack("<I7dI", 1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1)
        + b"frame_0001.jpg\0"
        + struct.pack("<Q", 0)
    )
    (model / "points3D.bin").write_bytes(
        struct.pack("<Q", 1)
        + struct.pack(
            "<Q3d3BdQ", 1, 0.0, 0.0, 1.0, 255, 255, 255, 0.1, 0
        )
    )


class FakeTrainer:
    fail_at: int | None = None
    signal_at: int | None = None
    loaded_iterations: list[int] = []
    stepped_iterations: list[int] = []

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.iteration = 0
        self.splat_count = 1

    def step(self):
        next_iteration = self.iteration + 1
        if self.fail_at == next_iteration:
            raise RuntimeError("interrupted step")
        self.iteration = next_iteration
        self.stepped_iterations.append(self.iteration)
        if self.signal_at == self.iteration:
            signal.raise_signal(signal.SIGTERM)
        return SimpleNamespace(iteration=self.iteration, splat_count=42)

    def save_checkpoint(self, path: str) -> None:
        _write_checkpoint(Path(path), self.iteration)

    def load_checkpoint(self, path: str) -> None:
        summary = inspect_checkpoint(Path(path))
        self.iteration = summary.iteration
        self.loaded_iterations.append(self.iteration)

    def export_ply(self, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"ply\nnonempty\n")

    def evaluate(self):
        return {"psnr": 24.5, "ssim": 0.8}


@pytest.fixture(autouse=True)
def reset_fake_trainer():
    FakeTrainer.fail_at = None
    FakeTrainer.signal_at = None
    FakeTrainer.loaded_iterations = []
    FakeTrainer.stepped_iterations = []


def _install_fake_msplat(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_msplat = SimpleNamespace(
        __version__="1.1.4",
        load_dataset=lambda path, *, downscale_factor, eval_mode: (
            path,
            downscale_factor,
            eval_mode,
        ),
        TrainingConfig=lambda **values: values,
        GaussianTrainer=FakeTrainer,
    )
    monkeypatch.setitem(sys.modules, "msplat", fake_msplat)


def _run(
    scene: Path,
    *,
    iterations: int = 4,
    checkpoint_every: int = 2,
) -> list[str]:
    from msplatter import cli

    arguments = [
        str(scene),
        "--run-id",
        "test-run",
        "--iterations",
        str(iterations),
        "--num-downscales",
        "0",
        "--downscale-factor",
        "4",
        "--checkpoint-every",
        str(checkpoint_every),
        "--allow-unmasked",
    ]
    cli.main(arguments)
    return arguments


def _attempt(scene: Path) -> Path:
    return scene / ".gausser" / "staging" / "msplat-test-run"


def _resume_document(scene: Path) -> dict[str, object]:
    return json.loads((_attempt(scene) / "resume.json").read_text())


def test_checkpoint_parser_requires_complete_bounded_v114_structure(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.msplat"
    _write_checkpoint(checkpoint, 23)

    summary = inspect_checkpoint(checkpoint)

    assert summary.iteration == 23
    assert summary.num_active == 1
    assert len(summary.tensor_shapes) == 18

    checkpoint.write_bytes(checkpoint.read_bytes() + b"trailing")
    with pytest.raises(CheckpointFormatError, match="trailing"):
        inspect_checkpoint(checkpoint)


def test_step_failure_preserves_attempt_and_last_checkpoint(
    monkeypatch: pytest.MonkeyPatch, scene: Path
):
    _prepare_binary_model(scene)
    _install_fake_msplat(monkeypatch)
    monkeypatch.setattr(ScenePaths, "new_run_id", lambda _self: "test-run")
    FakeTrainer.fail_at = 3

    with pytest.raises(SystemExit) as exc_info:
        _run(scene, checkpoint_every=2)

    assert exc_info.value.code == 1
    assert _attempt(scene).is_dir()
    state = _resume_document(scene)
    assert state["current"]["iteration"] == 2
    assert not (scene / "runs" / "msplat").exists()


def test_corrupt_latest_generation_falls_back_to_previous(
    monkeypatch: pytest.MonkeyPatch, scene: Path
):
    _prepare_binary_model(scene)
    _install_fake_msplat(monkeypatch)
    monkeypatch.setattr(ScenePaths, "new_run_id", lambda _self: "test-run")
    FakeTrainer.signal_at = 2

    with pytest.raises(SystemExit) as exc_info:
        _run(scene, checkpoint_every=1)
    assert exc_info.value.code == 128 + signal.SIGTERM

    state = _resume_document(scene)
    assert state["current"]["iteration"] == 2
    assert state["previous"]["iteration"] == 1
    latest = _attempt(scene) / "checkpoints" / state["current"]["filename"]
    latest.write_bytes(b"truncated")

    FakeTrainer.signal_at = None
    FakeTrainer.stepped_iterations = []
    _run(scene, checkpoint_every=1)

    assert FakeTrainer.loaded_iterations == [1]
    assert FakeTrainer.stepped_iterations == [2, 3, 4]
    assert (scene / "runs" / "msplat" / "test-run" / "exports" / "scene.ply").is_file()


@pytest.mark.parametrize("mismatch", ["input", "sparse", "config"])
def test_resume_rejects_input_sparse_or_config_mismatch_without_overwrite(
    monkeypatch: pytest.MonkeyPatch, scene: Path, mismatch: str
):
    _prepare_binary_model(scene)
    _install_fake_msplat(monkeypatch)
    monkeypatch.setattr(ScenePaths, "new_run_id", lambda _self: "test-run")
    FakeTrainer.signal_at = 2

    with pytest.raises(SystemExit):
        _run(scene, checkpoint_every=2)
    before = (_attempt(scene) / "resume.json").read_bytes()

    FakeTrainer.signal_at = None
    if mismatch == "input":
        Image.new("RGB", (1, 1), color="black").save(
            scene / "images" / "frame_0001.jpg"
        )
        arguments = {"iterations": 4, "checkpoint_every": 2}
    elif mismatch == "sparse":
        (scene / "sparse" / "0" / "cameras.bin").write_bytes(b"changed")
        arguments = {"iterations": 4, "checkpoint_every": 2}
    else:
        arguments = {"iterations": 5, "checkpoint_every": 2}

    with pytest.raises(SystemExit) as exc_info:
        _run(scene, **arguments)

    assert exc_info.value.code == 2
    assert (_attempt(scene) / "resume.json").read_bytes() == before
    assert not (scene / "runs" / "msplat").exists()


def test_resume_rejects_corrupted_materialized_dataset(
    monkeypatch: pytest.MonkeyPatch, scene: Path
):
    _prepare_binary_model(scene)
    _install_fake_msplat(monkeypatch)
    monkeypatch.setattr(ScenePaths, "new_run_id", lambda _self: "test-run")
    FakeTrainer.signal_at = 2
    with pytest.raises(SystemExit):
        _run(scene, checkpoint_every=2)

    staged_model = _attempt(scene) / "run" / "dataset" / "cameras.bin"
    staged_model.write_bytes(b"corrupt")
    FakeTrainer.signal_at = None

    with pytest.raises(SystemExit) as exc_info:
        _run(scene, checkpoint_every=2)

    assert exc_info.value.code == 2


def test_resume_continues_from_exact_saved_iteration(
    monkeypatch: pytest.MonkeyPatch, scene: Path
):
    _prepare_binary_model(scene)
    _install_fake_msplat(monkeypatch)
    monkeypatch.setattr(ScenePaths, "new_run_id", lambda _self: "test-run")
    FakeTrainer.signal_at = 2

    with pytest.raises(SystemExit):
        _run(scene, checkpoint_every=2)

    FakeTrainer.signal_at = None
    FakeTrainer.stepped_iterations = []
    _run(scene, checkpoint_every=2)

    assert FakeTrainer.loaded_iterations == [2]
    assert FakeTrainer.stepped_iterations == [3, 4]


def test_signal_is_deferred_to_a_completed_step_and_checkpointed(
    monkeypatch: pytest.MonkeyPatch, scene: Path
):
    _prepare_binary_model(scene)
    _install_fake_msplat(monkeypatch)
    monkeypatch.setattr(ScenePaths, "new_run_id", lambda _self: "test-run")
    FakeTrainer.signal_at = 2

    with pytest.raises(SystemExit) as exc_info:
        _run(scene, checkpoint_every=1000)

    assert exc_info.value.code == 128 + signal.SIGTERM
    assert _resume_document(scene)["current"]["iteration"] == 2
    assert FakeTrainer.stepped_iterations == [1, 2]


def test_final_publication_requires_nonempty_export_and_validated_checkpoint(
    monkeypatch: pytest.MonkeyPatch, scene: Path
):
    _prepare_binary_model(scene)
    _install_fake_msplat(monkeypatch)
    monkeypatch.setattr(ScenePaths, "new_run_id", lambda _self: "test-run")

    _run(scene, checkpoint_every=2)

    run = scene / "runs" / "msplat" / "test-run"
    assert (run / "exports" / "scene.ply").stat().st_size > 0
    final = inspect_checkpoint(run / "checkpoint.msplat")
    assert final.iteration == 4
    receipt = json.loads((run / "checkpoint-receipt.json").read_text())
    assert receipt["completed"] == {
        "checkpoint_sha256": receipt["current"]["sha256"],
        "checkpoint_size_bytes": receipt["current"]["size_bytes"],
        "iteration": 4,
        "num_active": 1,
    }
    assert [item["iteration"] for item in receipt["history"]] == [2, 4]
    assert receipt["current"]["num_active"] == 1
    manifest = json.loads((scene / ".gausser" / "manifest.json").read_text())
    outputs = manifest["stages"]["msplat"]["outputs"]
    assert outputs["checkpoint_receipt"].endswith("checkpoint-receipt.json")
    assert len(outputs["checkpoint_receipt_sha256"]) == 64
    assert outputs["terminal_iteration"] == 4
    assert outputs["num_active"] == 1
    assert not _attempt(scene).exists()


def test_new_publication_preserves_previous_successful_runs(
    monkeypatch: pytest.MonkeyPatch, scene: Path
):
    _prepare_binary_model(scene)
    _install_fake_msplat(monkeypatch)
    monkeypatch.setattr(ScenePaths, "new_run_id", lambda _self: "test-run")
    previous = scene / "runs" / "msplat" / "previous" / "exports" / "scene.ply"
    previous.parent.mkdir(parents=True)
    previous.write_bytes(b"previous")

    _run(scene, checkpoint_every=2)

    assert previous.read_bytes() == b"previous"
    assert (scene / "runs" / "msplat" / "test-run" / "exports" / "scene.ply").is_file()
