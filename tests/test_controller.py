from __future__ import annotations

import json
from pathlib import Path
import threading

import pytest

from gausser.controller import SceneController, StageRequest, StageUnavailable
from gausser.process import CommandResult
from gausser.scene import ScenePaths


def test_controller_uses_scene_owned_command_and_cancellation(scene: Path):
    calls: list[tuple[list[str], dict]] = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs))
        return CommandResult(tuple(command), kwargs["cwd"], 0, 0.0, (), False)

    controller = SceneController(ScenePaths.from_root(scene), runner=fake_runner)
    cancel = threading.Event()
    result = controller.run(StageRequest("sammer", ("--prompt", "statue")), cancel_event=cancel)

    assert result.succeeded
    assert calls[0][0] == ["sammer", "--scene", str(scene), "--prompt", "statue"]
    assert calls[0][1]["cwd"] == scene
    assert calls[0][1]["cancel_event"] is cancel
    assert calls[0][1]["env"] is None
    assert calls[0][1]["inherited_fds"] == ()
    assert calls[0][1]["interrupt_grace_seconds"] == 2.0


def test_controller_allows_msplat_time_to_finish_a_large_checkpoint(
    scene: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[dict] = []

    def fake_runner(command, **kwargs):
        calls.append(kwargs)
        return CommandResult(tuple(command), kwargs["cwd"], 0, 0.0, (), False)

    controller = SceneController(ScenePaths.from_root(scene), runner=fake_runner)
    monkeypatch.setattr(controller, "readiness", lambda: {"msplat": "ready"})

    controller.run(StageRequest("msplat", ("--allow-unmasked",)))

    assert calls[0]["interrupt_grace_seconds"] == 300.0


def test_controller_passes_the_scene_lock_lease_to_a_stage_cli(scene: Path):
    calls: list[dict] = []

    def fake_runner(_command, **kwargs):
        calls.append(kwargs)
        return CommandResult(("sammer",), kwargs["cwd"], 0, 0.0, (), False)

    controller = SceneController(ScenePaths.from_root(scene), runner=fake_runner)
    controller.run(StageRequest("sammer", ("--prompt", "statue")), scene_lock_fd=42)

    assert calls[0]["inherited_fds"] == (42,)
    assert calls[0]["env"] == {"GAUSSER_SCENE_LOCK_FD": "42"}


def test_controller_builds_scene_owned_sharpframes_command(scene: Path):
    controller = SceneController(ScenePaths.from_root(scene))

    assert controller.command_for(StageRequest("sharpframes", ("source.mov", "--quality", "80"))) == [
        "sharpframes", "source.mov", "--quality", "80", "--scene", str(scene)
    ]


def test_controller_passes_resolved_studio_settings_to_sharpframes(scene: Path):
    controller = SceneController(ScenePaths.from_root(scene))

    command = controller.command_for(StageRequest("sharpframes", ("source.mov",), {
        "specification_version": 1, "preset": "custom", "values": {"fps": 2},
    }))

    assert command[:-2] == ["sharpframes", "source.mov", "--scene", str(scene)]
    assert command[-2] == "--studio-settings"
    assert json.loads(command[-1]) == {"specification_version": 1, "preset": "custom", "values": {"fps": 2}}


def test_controller_streams_known_stage_telemetry(scene: Path):
    events = []

    def fake_runner(command, **kwargs):
        kwargs["on_line"]("Processed file [2/3]")
        return CommandResult(tuple(command), kwargs["cwd"], 0, 0.0, (), False)

    controller = SceneController(ScenePaths.from_root(scene), runner=fake_runner)
    controller.run(StageRequest("colmap"), on_event=events.append)

    assert events[0].kind == "progress"
    assert events[0].data["current"] == 2
    assert events[0].data["total"] == 3


def test_controller_does_not_invent_msplat_acknowledgement(scene: Path):
    controller = SceneController(ScenePaths.from_root(scene))

    assert controller.command_for(StageRequest("msplat")) == ["msplatter", str(scene)]


def test_controller_preserves_one_resolved_msplat_acknowledgement(scene: Path):
    controller = SceneController(ScenePaths.from_root(scene))

    command = controller.command_for(StageRequest("msplat", ("--allow-unmasked",)))

    assert command == ["msplatter", str(scene), "--allow-unmasked"]
    assert command.count("--allow-unmasked") == 1


def test_controller_builds_scene_owned_lod_command(scene: Path):
    controller = SceneController(ScenePaths.from_root(scene))

    assert controller.command_for(StageRequest("lod", ("runs/brush/test/exports/export_15000.ply",))) == [
        "lodder", "--scene", str(scene), "runs/brush/test/exports/export_15000.ply"
    ]


def test_controller_gates_brush_without_a_valid_colmap_model(scene: Path):
    controller = SceneController(ScenePaths.from_root(scene))

    with pytest.raises(StageUnavailable, match="brush is unavailable"):
        controller.run(StageRequest("brush"))


def test_controller_allows_an_explicitly_completed_stage_to_run(scene: Path):
    calls: list[list[str]] = []

    def fake_runner(command, **kwargs):
        calls.append(command)
        return CommandResult(tuple(command), kwargs["cwd"], 0, 0.0, (), False)

    controller = SceneController(ScenePaths.from_root(scene), runner=fake_runner)
    controller.readiness = lambda: {"colmap": "completed"}  # type: ignore[method-assign]

    result = controller.run(StageRequest("colmap", ("--existing", "replace")))

    assert result.succeeded
    assert calls == [["colmapper", "--scene", str(scene), "--existing", "replace"]]


def test_controller_can_launch_sharpframes_before_images_directory_exists(tmp_path: Path):
    root = tmp_path / "scene"
    root.mkdir()
    calls: list[list[str]] = []

    def fake_runner(command, **kwargs):
        calls.append(command)
        return CommandResult(tuple(command), kwargs["cwd"], 0, 0.0, (), False)

    controller = SceneController(ScenePaths.from_root(root), runner=fake_runner)
    result = controller.run(StageRequest("sharpframes", ("/tmp/source.mov",)))

    assert result.succeeded
    assert calls == [["sharpframes", "/tmp/source.mov", "--scene", str(root)]]
