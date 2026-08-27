from __future__ import annotations

from gausser.preflight import inspect_scene, inspect_tools
from gausser.scene import ScenePaths


def test_preflight_blocks_sharpframes_when_images_are_already_owned(scene):
    (scene / "images" / "photo.jpg").write_bytes(b"image")

    statuses = {item.stage: item for item in inspect_scene(ScenePaths.from_root(scene))}

    assert statuses["sharpframes"].state == "unavailable"
    assert "replace" in statuses["sharpframes"].detail.lower()


def test_tool_preflight_checks_the_actual_stage_commands(monkeypatch):
    commands = {"sharpframes", "sharp-frames", "sammer", "colmap", "brush", "msplatter", "splat-transform"}
    monkeypatch.setattr("gausser.preflight.shutil.which", lambda command: "/tools/" + command if command in commands else None)
    monkeypatch.setattr("gausser.preflight.importlib.util.find_spec", lambda name: object() if name == "msplat" else None)
    monkeypatch.setattr("gausser.preflight._sammer_runtime_status", lambda _: type("Status", (), {"stage": "sammer", "state": "ready", "detail": "runtime"})())

    assert {item.stage: item.state for item in inspect_tools()} == {
        "sharpframes": "ready", "sammer": "ready", "colmap": "ready", "brush": "ready", "msplat": "ready", "lod": "ready",
    }


def test_tool_preflight_reports_missing_sharp_frames_backend(monkeypatch):
    commands = {"sharpframes", "sammer", "colmap", "brush", "msplatter", "splat-transform"}
    monkeypatch.setattr("gausser.preflight.shutil.which", lambda command: "/tools/" + command if command in commands else None)
    monkeypatch.setattr("gausser.preflight.importlib.util.find_spec", lambda name: object() if name == "msplat" else None)
    monkeypatch.setattr("gausser.preflight._sammer_runtime_status", lambda _: type("Status", (), {"stage": "sammer", "state": "ready", "detail": "runtime"})())

    status = {item.stage: item for item in inspect_tools()}["sharpframes"]
    assert status.state == "unavailable"
    assert "sharp-frames" in status.detail


def test_preflight_guides_a_scene_without_colmap(scene):
    statuses = {item.stage: item for item in inspect_scene(ScenePaths.from_root(scene))}

    assert statuses["images"].state == "ready"
    assert statuses["sammer"].state == "ready"
    assert statuses["colmap"].state == "ready"
    assert statuses["brush"].state == "unavailable"
    assert statuses["msplat"].state == "unavailable"
    assert "unmasked" in statuses["msplat"].detail
    assert statuses["lod"].state == "ready"
    assert "explicit" in statuses["lod"].detail


def test_sammer_runtime_preflight_reports_success(monkeypatch):
    from types import SimpleNamespace
    from gausser.preflight import _sammer_runtime_status

    captured = {}
    monkeypatch.setattr("gausser.preflight.subprocess.run", lambda command, **kwargs: (captured.update(command=command, kwargs=kwargs) or SimpleNamespace(returncode=0, stdout="ready", stderr="")))

    status = _sammer_runtime_status("/tools/sammer")

    assert status.state == "ready"
    assert captured["command"] == ["/tools/sammer", "--check-runtime"]
    assert captured["kwargs"]["timeout"] == 20


def test_sammer_runtime_preflight_reports_failed_or_unavailable_runtime(monkeypatch):
    from types import SimpleNamespace
    from gausser.preflight import _sammer_runtime_status
    import subprocess

    monkeypatch.setattr("gausser.preflight.subprocess.run", lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr="SAM3 is missing"))
    assert _sammer_runtime_status("/tools/sammer").detail == "SAM3 is missing"
    assert _sammer_runtime_status(None).state == "unavailable"

    def time_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("sammer", 20)

    monkeypatch.setattr("gausser.preflight.subprocess.run", time_out)
    assert "timed out" in _sammer_runtime_status("/tools/sammer").detail.lower()

    monkeypatch.setattr("gausser.preflight.subprocess.run", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("not executable")))
    assert "not executable" in _sammer_runtime_status("/tools/sammer").detail
