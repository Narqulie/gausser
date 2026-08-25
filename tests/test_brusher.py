from __future__ import annotations

from pathlib import Path

import pytest

from brusher import cli
from gausser.process import CommandResult


def test_default_export_path_belongs_to_explicit_scene(monkeypatch, scene, tmp_path: Path):
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    monkeypatch.setattr(cli.shutil, "which", lambda _: "/fake/brush")

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
    export_path = Path(commands[0][export_index])
    assert export_path.is_relative_to(scene / "runs" / "brush")


def test_export_interval_must_produce_an_export(monkeypatch, scene):
    monkeypatch.setattr(cli.shutil, "which", lambda _: "/fake/brush")
    monkeypatch.setattr(cli, "run_brush", lambda *args, **kwargs: pytest.fail("runner should not start"))

    with pytest.raises(SystemExit) as exc_info:
        cli.main([str(scene), "--total-train-iters", "10", "--export-every", "10"])

    assert exc_info.value.code == 2
