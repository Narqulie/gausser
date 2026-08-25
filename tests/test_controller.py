from __future__ import annotations

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


def test_controller_builds_explicit_unmasked_msplat_command(scene: Path):
    controller = SceneController(ScenePaths.from_root(scene))

    assert controller.command_for(StageRequest("msplat")) == [
        "msplatter", str(scene), "--allow-unmasked"
    ]


def test_controller_gates_brush_without_a_valid_colmap_model(scene: Path):
    controller = SceneController(ScenePaths.from_root(scene))

    with pytest.raises(StageUnavailable, match="brush is unavailable"):
        controller.run(StageRequest("brush"))
