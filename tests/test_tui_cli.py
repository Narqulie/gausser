from __future__ import annotations

import pytest

from gausser import tui


def test_tui_entrypoint_has_non_interactive_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        tui.main(["--help"])

    assert exc_info.value.code == 0
    assert "Textual scene controller" in capsys.readouterr().out
