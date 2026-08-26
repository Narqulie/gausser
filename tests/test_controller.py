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
    assert calls[0][1]["inherited_fds"] == ()


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


def test_controller_builds_explicit_unmasked_msplat_command(scene: Path):
    controller = SceneController(ScenePaths.from_root(scene))

    assert controller.command_for(StageRequest("msplat")) == [
        "msplatter", str(scene), "--allow-unmasked"
    ]


def test_controller_builds_scene_owned_lod_command(scene: Path):
    controller = SceneController(ScenePaths.from_root(scene))

    assert controller.command_for(StageRequest("lod", ("runs/brush/test/exports/export_15000.ply",))) == [
        "lodder", "--scene", str(scene), "runs/brush/test/exports/export_15000.ply"
    ]


def test_controller_gates_brush_without_a_valid_colmap_model(scene: Path):
    controller = SceneController(ScenePaths.from_root(scene))

    with pytest.raises(StageUnavailable, match="brush is unavailable"):
        controller.run(StageRequest("brush"))
