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

    assert {item.stage: item.state for item in inspect_tools()} == {
        "sharpframes": "ready", "sammer": "ready", "colmap": "ready", "brush": "ready", "msplat": "ready", "lod": "ready",
    }


def test_tool_preflight_reports_missing_sharp_frames_backend(monkeypatch):
    commands = {"sharpframes", "sammer", "colmap", "brush", "msplatter", "splat-transform"}
    monkeypatch.setattr("gausser.preflight.shutil.which", lambda command: "/tools/" + command if command in commands else None)
    monkeypatch.setattr("gausser.preflight.importlib.util.find_spec", lambda name: object() if name == "msplat" else None)

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
