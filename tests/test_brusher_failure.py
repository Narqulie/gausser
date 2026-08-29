from __future__ import annotations

import pytest

from brusher import cli
from gausser.process import CommandResult


def test_successful_brush_exit_without_export_is_rejected(monkeypatch, scene):
    monkeypatch.setattr(cli, "resolve_command", lambda _: "/fake/brush")
    monkeypatch.setattr(
        cli,
        "run_brush",
        lambda command, **kwargs: CommandResult(tuple(command), kwargs["cwd"], 0, 0.0, (), False),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main([str(scene), "--total-train-iters", "10", "--export-every", "5"])

    assert exc_info.value.code == 1
