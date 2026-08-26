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

        def run(self, request, *, on_line, cancel_event, inherited_fds):
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


def test_engine_catalog_and_typed_sharpframes_start_use_engine_resolved_options(scene: Path):
    from gausser.engine import GausserEngine

    events: list[dict] = []
    received: list[tuple[str, ...]] = []

    class Controller:
        def __init__(self, paths: ScenePaths) -> None:
            self.scene = paths

        def command_for(self, request):
            return ["sharpframes", *request.arguments, "--scene", str(self.scene.root)]

        def run(self, request, *, on_line, cancel_event, inherited_fds):
            received.append(request.arguments)
            return CommandResult((request.stage,), self.scene.root, 0, 0.0, (), False)

    engine = GausserEngine(controller_factory=Controller)
    engine.dispatch({"id": "catalog", "method": "stage_catalog", "params": {"scene": str(scene)}}, events.append)
    assert events.pop()["result"]["stages"]["sharpframes"]["version"] == 1

    engine.dispatch(
        {
            "id": "resolve", "method": "resolve_stage",
            "params": {"scene": str(scene), "stage": "sharpframes", "options": {"source": "/tmp/clips", "fps": 2}},
        },
        events.append,
    )
    assert events.pop()["result"]["command"][:3] == ["sharpframes", "/tmp/clips", "--fps"]

    engine.dispatch(
        {
            "id": "start", "method": "start",
            "params": {"scene": str(scene), "stage": "sharpframes", "options": {"source": "/tmp/clips", "fps": 2}},
        },
        events.append,
    )
    _wait_for(events, "finished")

    assert received == [
        ("/tmp/clips", "--fps", "2", "--num-frames", "300", "--min-buffer", "3", "--format", "jpg",
         "--selection-method", "best-n", "--batch-size", "5", "--batch-buffer", "2", "--outlier-window-size", "15",
         "--outlier-sensitivity", "60", "--width", "0"),
    ]


def test_engine_initializes_explicit_empty_scene(tmp_path: Path):
    from gausser.engine import GausserEngine

    root = tmp_path / "new-scene"
    root.mkdir()
    emitted: list[dict] = []

    GausserEngine().dispatch(
        {"id": "create-1", "method": "initialize", "params": {"scene": str(root)}},
        emitted.append,
    )

    assert emitted == [
        {
            "id": "create-1",
            "result": {
                "scene": str(root),
                "manifest": str(root / ".gausser" / "manifest.json"),
            },
        }
    ]
    assert (root / "images").is_dir()


def test_engine_reports_nonempty_project_initialization_as_a_domain_error(tmp_path: Path):
    from gausser.engine import GausserEngine

    root = tmp_path / "not-empty"
    root.mkdir()
    (root / "source.mov").write_bytes(b"video")
    emitted: list[dict] = []

    GausserEngine().dispatch(
        {"id": "create-2", "method": "initialize", "params": {"scene": str(root)}},
        emitted.append,
    )

    assert emitted == [{
        "id": "create-2",
        "error": {"code": "scene_initialization_failed", "message": f"New scene directory must be empty: {root}"},
    }]


def test_engine_ingests_photo_directory_with_explicit_policy(scene: Path, tmp_path: Path):
    from PIL import Image
    from gausser.engine import GausserEngine
    from gausser.scene import initialize_scene

    scene = tmp_path / "initialized"
    scene.mkdir()
    initialize_scene(scene)
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (3, 2)).save(source / "photo.jpg")
    emitted: list[dict] = []

    GausserEngine().dispatch(
        {
            "id": "ingest-1",
            "method": "ingest_photos",
            "params": {"scene": str(scene), "source": str(source), "policy": "copy"},
        },
        emitted.append,
    )

    assert emitted[0]["id"] == "ingest-1"
    assert emitted[0]["result"]["policy"] == "copy"
    assert emitted[0]["result"]["image_count"] == 1
    assert (scene / "images" / "photo.jpg").is_file()


def test_engine_reimports_edited_supersplat_export(scene: Path, tmp_path: Path):
    from gausser.engine import GausserEngine

    source = scene / "runs" / "brush" / "source-run" / "exports" / "scene.ply"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    edited = tmp_path / "edited.ply"
    edited.write_bytes(b"edited")
    emitted: list[dict] = []

    GausserEngine().dispatch(
        {
            "id": "reimport",
            "method": "reimport_supersplat",
            "params": {"scene": str(scene), "source": "runs/brush/source-run/exports/scene.ply", "edited": str(edited)},
        },
        emitted.append,
    )

    output = scene / "runs" / "supersplat" / emitted[0]["result"]["run_id"] / "exports" / "edited.ply"
    assert output.read_bytes() == b"edited"


def test_engine_rejects_photo_ingest_while_a_stage_owns_the_scene(scene: Path):
    from gausser.engine import GausserEngine

    events: list[dict] = []
    release = threading.Event()

    class Controller:
        def __init__(self, paths: ScenePaths) -> None:
            self.scene = paths

        def run(self, request, *, on_line, cancel_event, inherited_fds):
            release.wait(timeout=1)
            return CommandResult((request.stage,), self.scene.root, 0, 0.0, (), False)

    engine = GausserEngine(controller_factory=Controller)
    engine.dispatch(
        {"id": "start", "method": "start", "params": {"scene": str(scene), "stage": "sharpframes"}},
        events.append,
    )
    _wait_for(events, "started")
    engine.dispatch(
        {
            "id": "ingest",
            "method": "ingest_photos",
            "params": {"scene": str(scene), "source": str(scene), "policy": "copy"},
        },
        events.append,
    )
    release.set()
    _wait_for(events, "finished")

    assert events[2] == {"id": "ingest", "error": {"code": "scene_busy", "message": f"a run is already active for {scene}"}}


def test_engine_lists_scene_owned_artifacts(scene: Path):
    from gausser.engine import GausserEngine

    export = scene / "runs" / "brush" / "run-a" / "exports" / "scene.ply"
    export.parent.mkdir(parents=True)
    export.write_bytes(b"ply")
    emitted: list[dict] = []

    GausserEngine().dispatch(
        {"id": "artifacts-1", "method": "artifacts", "params": {"scene": str(scene)}},
        emitted.append,
    )

    assert emitted == [{"id": "artifacts-1", "result": {"artifacts": [{
        "relative_path": "runs/brush/run-a/exports/scene.ply",
        "kind": "brush",
        "run_id": "run-a",
        "format": "ply",
        "size_bytes": 3,
    }]}}]


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

        def run(self, request, *, on_line, cancel_event, inherited_fds):
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


def test_engine_shutdown_cancels_active_work_and_releases_scene(scene: Path):
    from gausser.engine import GausserEngine

    events: list[dict] = []
    observed_cancel = threading.Event()

    class Controller:
        def __init__(self, paths: ScenePaths) -> None:
            self.scene = paths

        def run(self, request, *, on_line, cancel_event, inherited_fds):
            assert cancel_event is not None
            assert cancel_event.wait(timeout=1)
            observed_cancel.set()
            return CommandResult((request.stage,), self.scene.root, 130, 0.0, (), True)

    engine = GausserEngine(controller_factory=Controller)
    engine.dispatch(
        {"id": "start", "method": "start", "params": {"scene": str(scene), "stage": "sammer"}},
        events.append,
    )
    _wait_for(events, "started")

    engine.shutdown()

    assert observed_cancel.is_set()
    assert _wait_for(events, "finished")["run"]["state"] == "cancelled"
    status: list[dict] = []
    engine.dispatch({"id": "status", "method": "status", "params": {}}, status.append)
    assert status == [{"id": "status", "result": {"active_runs": []}}]


def test_engine_cancel_sets_the_stage_cancellation_event(scene: Path):
    from gausser.engine import GausserEngine

    events: list[dict] = []
    observed_cancel = threading.Event()

    class Controller:
        def __init__(self, paths: ScenePaths) -> None:
            self.scene = paths

        def run(self, request, *, on_line, cancel_event, inherited_fds):
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
