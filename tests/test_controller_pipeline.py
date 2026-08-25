from __future__ import annotations

from gausser.controller import SceneController, StageRequest
from gausser.process import CommandResult
from gausser.scene import ScenePaths


def test_pipeline_stops_after_failed_colmap(scene):
    calls: list[str] = []

    def fake_runner(command, **kwargs):
        calls.append(command[0])
        return CommandResult(tuple(command), kwargs["cwd"], 1 if command[0] == "colmapper" else 0, 0.0, ("failure",), False)

    controller = SceneController(ScenePaths.from_root(scene), runner=fake_runner)
    requests = {
        "sammer": StageRequest("sammer"),
        "colmap": StageRequest("colmap"),
        "brush": StageRequest("brush"),
    }
    results = controller.run_pipeline(requests)

    assert [result.returncode for result in results] == [0, 1]
    assert calls == ["sammer", "colmapper"]
