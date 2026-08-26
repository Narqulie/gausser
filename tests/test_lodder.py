from __future__ import annotations

import json
from pathlib import Path

import pytest

from gausser.process import CommandResult
from gausser.scene import ScenePaths


def _success(command, cwd, output=()):
    return CommandResult(tuple(command), cwd, 0, 0.0, tuple(output), False)


def test_lodder_publishes_validated_scene_owned_lod(monkeypatch, scene):
    from lodder import cli

    source = scene / "runs" / "brush" / "brush-run" / "exports" / "export_15000.ply"
    source.parent.mkdir(parents=True)
    source.write_text("ply source", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command, *, cwd, on_line=None, **kwargs):
        calls.append(command)
        if command[-1] == "--version":
            return _success(command, cwd, ("splat-transform v2.0.3",))
        output = Path(command[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        if "--decimate" in command:
            output.write_text("ply level", encoding="utf-8")
        else:
            (output.parent / "0_0").mkdir()
            (output.parent / "0_0" / "meta.json").write_text("{}", encoding="utf-8")
            output.write_text(json.dumps({"lodLevels": 3, "filenames": ["0_0/meta.json"]}), encoding="utf-8")
        return _success(command, cwd)

    monkeypatch.setattr(cli.shutil, "which", lambda _: "/usr/local/bin/splat-transform")
    monkeypatch.setattr(cli, "run_command", fake_run)
    monkeypatch.setattr(ScenePaths, "new_run_id", lambda _: "lod-run")

    cli.main([str(source), "--scene", str(scene), "--chunk-count", "1024", "--chunk-extent", "32"])

    output = scene / "runs" / "lod" / "lod-run"
    assert (output / "lod-meta.json").is_file()
    assert (output / "0_0" / "meta.json").is_file()
    staged = scene / ".gausser" / "staging" / "lod-lod-run"
    assert calls == [
        ["splat-transform", "--version"],
        ["splat-transform", str(source), "--decimate", "50%", str(staged / "levels" / "lod-1.ply")],
        ["splat-transform", str(source), "--decimate", "25%", str(staged / "levels" / "lod-2.ply")],
        [
            "splat-transform", "-C", "1024", "-X", "32", str(source), "-l", "0",
            str(staged / "levels" / "lod-1.ply"), "-l", "1",
            str(staged / "levels" / "lod-2.ply"), "-l", "2",
            str(staged / "lod-meta.json"),
        ],
    ]
    manifest = json.loads((scene / ".gausser" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["lod"]["source"] == "runs/brush/brush-run/exports/export_15000.ply"
    assert manifest["stages"]["lod"]["tool_version"] == "splat-transform v2.0.3"
    assert manifest["stages"]["lod"]["output"] == "runs/lod/lod-run/lod-meta.json"
    assert manifest["stages"]["lod"]["lod_percentages"] == [100, 50, 25]


def test_lodder_refuses_input_outside_the_scene(monkeypatch, scene, tmp_path):
    from lodder import cli

    source = tmp_path / "external.ply"
    source.write_text("ply", encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        cli.main([str(source), "--scene", str(scene)])


def test_lodder_refuses_incomplete_output(monkeypatch, scene):
    from lodder import cli

    source = scene / "runs" / "brush" / "brush-run" / "exports" / "export_15000.ply"
    source.parent.mkdir(parents=True)
    source.write_text("ply source", encoding="utf-8")

    def fake_run(command, *, cwd, on_line=None, **kwargs):
        if command[-1] == "--version":
            return _success(command, cwd, ("splat-transform v2.0.3",))
        output = Path(command[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        if "--decimate" in command:
            output.write_text("ply level", encoding="utf-8")
        else:
            (output.parent / "0_0").mkdir()
            (output.parent / "0_0" / "meta.json").write_text(
                json.dumps({"means": {"files": ["missing.webp"]}}), encoding="utf-8"
            )
            output.write_text(json.dumps({"lodLevels": 3, "filenames": ["0_0/meta.json"]}), encoding="utf-8")
        return _success(command, cwd)

    monkeypatch.setattr(cli.shutil, "which", lambda _: "/usr/local/bin/splat-transform")
    monkeypatch.setattr(cli, "run_command", fake_run)

    with pytest.raises(SystemExit, match="1"):
        cli.main([str(source), "--scene", str(scene)])

    assert not list((scene / "runs" / "lod").glob("*")) if (scene / "runs" / "lod").exists() else True
