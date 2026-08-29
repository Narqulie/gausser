from __future__ import annotations

import json
from pathlib import Path

import pytest

from brusher import cli
from gausser.process import CommandResult


def test_default_export_path_belongs_to_explicit_scene(monkeypatch, scene, tmp_path: Path):
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    monkeypatch.setattr(cli, "resolve_command", lambda _: "/fake/brush")

    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs) -> CommandResult:
        commands.append(cmd)
        export_path = Path(cmd[cmd.index("--export-path") + 1])
        export_path.mkdir(parents=True)
        (export_path / "result.ply").write_text("ply")
        return CommandResult(tuple(cmd), kwargs["cwd"], 0, 0.0, (), False)

    monkeypatch.setattr(cli, "run_brush", fake_run)

    with pytest.raises(SystemExit) as exc_info:
        cli.main([str(scene), "--total-train-iters", "10", "--export-every", "5"])

    assert exc_info.value.code == 0
    export_index = commands[0].index("--export-path") + 1
    staged_export = Path(commands[0][export_index])
    assert staged_export.is_relative_to(scene / ".gausser" / "staging")
    manifest = json.loads((scene / ".gausser" / "manifest.json").read_text(encoding="utf-8"))
    published_run = scene / manifest["stages"]["brush"]["outputs"]["run"]
    assert published_run.is_relative_to(scene / "runs" / "brush")
    assert (published_run / "exports" / "result.ply").is_file()
    assert not staged_export.exists()


def test_brusher_rejects_an_image_directory_containing_only_sidecars(
    monkeypatch, tmp_path: Path, capsys
):
    scene = tmp_path / "scene"
    images = scene / "images"
    images.mkdir(parents=True)
    (images / "._frame.jpg").write_bytes(b"sidecar")
    monkeypatch.setattr(cli, "resolve_command", lambda _: "/fake/brush")
    monkeypatch.setattr(
        cli,
        "run_brush",
        lambda *args, **kwargs: pytest.fail("runner should not start"),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main([str(scene), "--total-train-iters", "10", "--export-every", "5"])

    assert exc_info.value.code == 1
    assert "No recognisable dataset" in capsys.readouterr().out


def test_external_export_path_is_rejected_before_training(monkeypatch, scene, tmp_path: Path):
    monkeypatch.setattr(cli, "resolve_command", lambda _: "/fake/brush")
    monkeypatch.setattr(cli, "run_brush", lambda *args, **kwargs: pytest.fail("runner should not start"))

    with pytest.raises(SystemExit) as exc_info:
        cli.main([
            str(scene),
            "--total-train-iters", "10",
            "--export-every", "5",
            "--export-path", str(tmp_path / "external"),
        ])

    assert exc_info.value.code == 2


def test_export_interval_must_produce_an_export(monkeypatch, scene):
    monkeypatch.setattr(cli, "resolve_command", lambda _: "/fake/brush")
    monkeypatch.setattr(cli, "run_brush", lambda *args, **kwargs: pytest.fail("runner should not start"))

    with pytest.raises(SystemExit) as exc_info:
        cli.main([str(scene), "--total-train-iters", "10", "--export-every", "10"])

    assert exc_info.value.code == 2
