from __future__ import annotations

import pytest

from gausser.preflight import Readiness
from gausser.scene import ScenePaths
from gausser import tui


@pytest.mark.asyncio
async def test_tui_renders_preflight_and_requires_second_launch_press(monkeypatch, scene):
    monkeypatch.setattr(
        tui,
        "inspect_tools",
        lambda: [Readiness(stage, "ready", stage) for stage in ("sammer", "colmap", "brush")],
    )
    app = tui.create_app(ScenePaths.from_root(scene))

    async with app.run_test() as pilot:
        await pilot.pause()
        overview = app.query_one("#overview")
        assert "images" in str(overview.render())
        assert "brush" in str(overview.render())
        assert app.query_one("#brush").disabled

        prompt = app.query_one("#prompt")
        prompt.value = "statue"
        await pilot.click("#sammer")
        assert app.pending_request is not None
        await pilot.press("escape")
        assert app.pending_request is None
