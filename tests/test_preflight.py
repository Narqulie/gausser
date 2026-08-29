from __future__ import annotations

import pytest

from gausser.artifacts import Artifact
from gausser.preflight import (
    inspect_direct_tool,
    inspect_launch_tool,
    inspect_scene,
    inspect_tools,
)
from gausser.scene import ScenePaths


def test_preflight_marks_owned_images_as_an_explicit_sharpframes_replacement(scene):
    (scene / "images" / "photo.jpg").write_bytes(b"image")

    statuses = {item.stage: item for item in inspect_scene(ScenePaths.from_root(scene))}

    assert statuses["sharpframes"].state == "completed"
    assert statuses["sharpframes"].required_action == "force"
    assert "replace" in statuses["sharpframes"].detail.lower()


def test_scene_readiness_ignores_metadata_sidecars_in_images_directory(
    tmp_path,
):
    paths = ScenePaths.from_root(tmp_path)
    paths.images.mkdir()
    (paths.images / ".DS_Store").write_bytes(b"metadata")
    (paths.images / "._frame.jpg").write_bytes(b"sidecar")

    statuses = {status.stage: status for status in inspect_scene(paths)}

    assert statuses["images"].state == "unavailable"
    assert statuses["sharpframes"].state == "ready"
    assert statuses["sharpframes"].required_action == "none"


def test_preflight_detects_a_complete_mask_generation(scene):
    (scene / "masks").mkdir()
    (scene / "masks" / "frame_0001.png").write_bytes(b"mask")
    (scene / "colmap" / "masks").mkdir(parents=True)
    (scene / "colmap" / "masks" / "frame_0001.jpg.png").write_bytes(b"mask")

    status = {item.stage: item for item in inspect_scene(ScenePaths.from_root(scene))}["sammer"]

    assert status.state == "completed"
    assert status.required_action == "force"
    assert "complete" in status.detail.lower()


def test_preflight_detects_partial_masks_that_require_replacement(scene):
    (scene / "masks").mkdir()
    (scene / "masks" / "frame_0001.png").write_bytes(b"mask")

    status = {item.stage: item for item in inspect_scene(ScenePaths.from_root(scene))}["sammer"]

    assert status.state == "ready"
    assert status.required_action == "force"
    assert "incomplete" in status.detail.lower()


def test_launch_tool_preflight_checks_wrappers_and_upstream_commands(monkeypatch):
    commands = {
        "sharpframes",
        "sharp-frames",
        "ffmpeg",
        "ffprobe",
        "sammer",
        "colmapper",
        "colmap",
        "brusher",
        "brush",
        "msplatter",
        "lodder",
        "splat-transform",
    }
    observed: list[str] = []

    def find(command, **_kwargs):
        observed.append(command)
        return "/tools/" + command if command in commands else None

    monkeypatch.setattr("gausser.preflight.resolve_command", find)
    monkeypatch.setattr(
        "gausser.preflight._command_runtime_status",
        lambda stage, command, **_kwargs: type(
            "Status", (), {"stage": stage, "state": "ready", "detail": command}
        )(),
    )

    assert {item.stage: item.state for item in inspect_tools()} == {
        "sharpframes": "ready",
        "sammer": "ready",
        "colmap": "ready",
        "brush": "ready",
        "msplat": "ready",
        "lod": "ready",
    }
    assert set(observed) == commands


def test_direct_tool_preflight_checks_only_current_runtime_dependencies(monkeypatch):
    observed: list[str] = []

    def find(command, **_kwargs):
        observed.append(command)
        return "/tools/" + command

    monkeypatch.setattr("gausser.preflight.resolve_command", find)
    monkeypatch.setattr(
        "gausser.preflight._current_sammer_runtime_status",
        lambda: type(
            "Status",
            (),
            {"stage": "sammer", "state": "ready", "detail": "current runtime"},
        )(),
    )
    monkeypatch.setattr(
        "gausser.preflight._current_msplat_runtime_status",
        lambda: type(
            "Status",
            (),
            {"stage": "msplat", "state": "ready", "detail": "current runtime"},
        )(),
    )

    assert inspect_direct_tool("sammer").state == "ready"
    assert inspect_direct_tool("msplat").state == "ready"
    assert inspect_direct_tool("sharpframes").state == "ready"
    assert observed == ["sharp-frames", "ffmpeg", "ffprobe"]


def test_single_launch_tool_preflight_does_not_probe_unrelated_runtimes(monkeypatch):
    commands: list[str] = []

    def find(command, **_kwargs):
        commands.append(command)
        return "/tools/" + command

    monkeypatch.setattr("gausser.preflight.resolve_command", find)
    monkeypatch.setattr(
        "gausser.preflight._command_runtime_status",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Python runtimes must not be inspected")
        ),
    )

    assert inspect_launch_tool("brush").state == "ready"
    assert commands == ["brusher", "brush"]


def test_launch_tool_preflight_reports_missing_stage_wrapper(monkeypatch):
    commands = {"colmap"}
    monkeypatch.setattr(
        "gausser.preflight.resolve_command",
        lambda command, **_kwargs: "/tools/" + command if command in commands else None,
    )

    status = inspect_launch_tool("colmap")

    assert status.state == "unavailable"
    assert "colmapper" in status.detail


def test_tool_preflight_reports_missing_sharp_frames_backend(monkeypatch):
    commands = {"sharpframes", "ffmpeg", "ffprobe"}
    monkeypatch.setattr(
        "gausser.preflight.resolve_command",
        lambda command, **_kwargs: "/tools/" + command if command in commands else None,
    )

    status = {item.stage: item for item in inspect_tools()}["sharpframes"]
    assert status.state == "unavailable"
    assert "sharp-frames" in status.detail


def test_direct_tool_preflight_reports_missing_sharp_frames_media_tools(monkeypatch):
    commands = {"sharp-frames"}
    monkeypatch.setattr(
        "gausser.preflight.resolve_command",
        lambda command, **_kwargs: "/tools/" + command if command in commands else None,
    )

    status = inspect_direct_tool("sharpframes")

    assert status.state == "unavailable"
    assert "ffmpeg" in status.detail
    assert "ffprobe" in status.detail


@pytest.mark.parametrize("artifact", ["database.db", "sparse/1/cameras.bin"])
def test_preflight_requires_policy_for_incomplete_colmap_artifacts(scene, artifact):
    path = scene / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"incomplete")

    status = {item.stage: item for item in inspect_scene(ScenePaths.from_root(scene))}[
        "colmap"
    ]

    assert status.state == "ready"
    assert status.required_action == "existing_policy"
    assert "incomplete" in status.detail.lower()


def test_preflight_guides_a_scene_without_colmap(scene):
    statuses = {item.stage: item for item in inspect_scene(ScenePaths.from_root(scene))}

    assert statuses["images"].state == "ready"
    assert statuses["sammer"].state == "ready"
    assert statuses["colmap"].state == "ready"
    assert statuses["brush"].state == "unavailable"
    assert statuses["msplat"].state == "unavailable"
    assert "unmasked" in statuses["msplat"].detail
    assert statuses["lod"].state == "unavailable"
    assert "ply or sog" in statuses["lod"].detail.lower()


def test_preflight_lod_readiness_follows_scene_owned_artifacts(scene):
    export = scene / "runs" / "brush" / "run-1" / "exports" / "model.ply"
    export.parent.mkdir(parents=True)
    export.write_bytes(b"ply")

    statuses = {item.stage: item for item in inspect_scene(ScenePaths.from_root(scene))}
    assert statuses["lod"].state == "ready"
    assert "model.ply" in statuses["lod"].detail

    metadata = scene / "runs" / "lod" / "run-2" / "lod-meta.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("{}")
    statuses = {item.stage: item for item in inspect_scene(ScenePaths.from_root(scene))}
    assert statuses["lod"].state == "completed"


def test_preflight_does_not_treat_published_lod_chunks_as_new_lod_sources(scene):
    artifacts = [Artifact("runs/lod/run-1/chunks/0.sog", "lod", "run-1", "sog", 100)]

    statuses = {
        item.stage: item
        for item in inspect_scene(ScenePaths.from_root(scene), artifacts)
    }

    assert statuses["lod"].state == "unavailable"


def test_spawned_runtime_preflight_reports_success(monkeypatch):
    from types import SimpleNamespace

    from gausser.preflight import _command_runtime_status

    captured = {}
    expected_environment = {"PATH": "/normalized/path"}
    monkeypatch.setattr(
        "gausser.preflight.command_environment",
        lambda: expected_environment,
    )
    monkeypatch.setattr(
        "gausser.preflight.subprocess.run",
        lambda command, **kwargs: (
            captured.update(command=command, kwargs=kwargs)
            or SimpleNamespace(returncode=0, stdout="ready", stderr="")
        ),
    )

    status = _command_runtime_status(
        "sammer",
        "/tools/sammer",
        success_detail="SAM3 runtime",
        missing_detail="Install Sammer",
    )

    assert status.state == "ready"
    assert captured["command"] == ["/tools/sammer", "--check-runtime"]
    assert captured["kwargs"]["timeout"] == 20
    assert captured["kwargs"]["env"] is expected_environment


def test_spawned_runtime_preflight_reports_failed_or_unavailable_runtime(monkeypatch):
    import subprocess
    from types import SimpleNamespace

    from gausser.preflight import _command_runtime_status

    def inspect(command):
        return _command_runtime_status(
            "sammer",
            command,
            success_detail="SAM3 runtime",
            missing_detail="Install Sammer",
        )

    monkeypatch.setattr(
        "gausser.preflight.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="SAM3 is missing"
        ),
    )
    assert inspect("/tools/sammer").detail == "SAM3 is missing"
    assert inspect(None).state == "unavailable"

    def time_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("sammer", 20)

    monkeypatch.setattr("gausser.preflight.subprocess.run", time_out)
    assert "timed out" in inspect("/tools/sammer").detail.lower()

    monkeypatch.setattr(
        "gausser.preflight.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("not executable")),
    )
    assert "not executable" in inspect("/tools/sammer").detail


def test_current_sammer_runtime_preflight_uses_current_interpreter(monkeypatch):
    import sys
    from types import SimpleNamespace

    calls: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "sammer.engine",
        SimpleNamespace(validate_sam3_runtime=lambda: calls.append("validated")),
    )

    from gausser.preflight import _current_sammer_runtime_status

    status = _current_sammer_runtime_status()

    assert status.state == "ready"
    assert calls == ["validated"]
