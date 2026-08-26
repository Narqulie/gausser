from __future__ import annotations

import io
import json
from pathlib import Path
import threading
import time

from gausser.process import CommandResult
from gausser.scene import ScenePaths


def _wait_for(events: list[dict], kind: str, timeout: float = 1.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for event in events:
            if event.get("event") == kind:
                return event
        time.sleep(0.01)
    raise AssertionError(f"did not receive {kind}; got {events!r}")


def test_engine_stdio_preserves_json_lines_after_a_malformed_message():
    from gausser.engine import run_stdio

    output = io.StringIO()
    run_stdio(io.StringIO("not-json\n{\"id\":\"status\",\"method\":\"status\"}\n"), output)

    messages = [json.loads(line) for line in output.getvalue().splitlines()]
    assert messages[0]["id"] is None
    assert messages[0]["error"]["code"] == "invalid_json"
    assert messages[1] == {"id": "status", "result": {"active_runs": []}}


def test_engine_starts_stage_after_acknowledging_and_streams_result(scene: Path):
    from gausser.engine import GausserEngine

    events: list[dict] = []

    class Controller:
        def __init__(self, paths: ScenePaths) -> None:
            self.scene = paths

        def run(self, request, *, on_line, cancel_event):
            on_line("working")
            return CommandResult((request.stage,), self.scene.root, 0, 0.1, ("working",), False)

    engine = GausserEngine(controller_factory=Controller)
    engine.dispatch(
        {
            "id": "start-1",
            "method": "start",
            "params": {"scene": str(scene), "stage": "sammer", "arguments": ["--prompt", "statue"]},
        },
        events.append,
    )

    finished = _wait_for(events, "finished")

    assert events[0] == {"id": "start-1", "result": {"run_id": finished["run"]["id"]}}
    assert [event["event"] for event in events[1:]] == ["started", "log", "finished"]
    assert finished["run"]["state"] == "succeeded"
    assert finished["run"]["result"]["command"] == ["sammer"]


def test_engine_rejects_malformed_requests_without_stopping(scene: Path):
    from gausser.engine import GausserEngine

    events: list[dict] = []
    engine = GausserEngine()

    engine.dispatch({"id": "bad", "method": "start", "params": {"scene": str(scene), "stage": "bad"}}, events.append)
    engine.dispatch({"id": "status", "method": "status", "params": {}}, events.append)

    assert events[0]["id"] == "bad"
    assert events[0]["error"]["code"] == "invalid_request"
    assert events[1] == {"id": "status", "result": {"active_runs": []}}


def test_engine_rejects_a_second_active_run_for_the_same_scene(scene: Path):
    from gausser.engine import GausserEngine

    events: list[dict] = []
    release = threading.Event()

    class Controller:
        def __init__(self, paths: ScenePaths) -> None:
            self.scene = paths

        def run(self, request, *, on_line, cancel_event):
            release.wait(timeout=1)
            return CommandResult((request.stage,), self.scene.root, 0, 0.0, (), False)

    engine = GausserEngine(controller_factory=Controller)
    start = {"id": "first", "method": "start", "params": {"scene": str(scene), "stage": "sammer"}}
    engine.dispatch(start, events.append)
    _wait_for(events, "started")
    engine.dispatch({**start, "id": "second"}, events.append)
    release.set()
    _wait_for(events, "finished")

    assert events[2] == {"id": "second", "error": {"code": "scene_busy", "message": f"a run is already active for {scene}"}}


def test_engine_cancel_sets_the_stage_cancellation_event(scene: Path):
    from gausser.engine import GausserEngine

    events: list[dict] = []
    observed_cancel = threading.Event()

    class Controller:
        def __init__(self, paths: ScenePaths) -> None:
            self.scene = paths

        def run(self, request, *, on_line, cancel_event):
            assert cancel_event is not None
            assert cancel_event.wait(timeout=1)
            observed_cancel.set()
            return CommandResult((request.stage,), self.scene.root, 130, 0.0, (), True)

    engine = GausserEngine(controller_factory=Controller)
    engine.dispatch({"id": "start", "method": "start", "params": {"scene": str(scene), "stage": "sammer"}}, events.append)
    run_id = _wait_for(events, "started")["run"]["id"]
    engine.dispatch({"id": "cancel", "method": "cancel", "params": {"run_id": run_id}}, events.append)
    finished = _wait_for(events, "finished")

    assert observed_cancel.is_set()
    assert events[-2] == {"id": "cancel", "result": {"run_id": run_id, "cancellation_requested": True}}
    assert finished["run"]["state"] == "cancelled"
