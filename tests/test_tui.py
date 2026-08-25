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


@pytest.mark.asyncio
async def test_tui_requires_a_second_explicit_msplat_acknowledgement(monkeypatch, scene):
    ready = lambda stage: Readiness(stage, "ready", stage)
    monkeypatch.setattr(tui, "inspect_scene", lambda _: [ready(stage) for stage in ("images", "sammer", "colmap", "brush", "msplat")])
    monkeypatch.setattr(tui, "inspect_tools", lambda: [ready(stage) for stage in ("sammer", "colmap", "brush", "msplat")])
    app = tui.create_app(ScenePaths.from_root(scene))

    async with app.run_test() as pilot:
        await pilot.pause()
        assert not app.query_one("#msplat").disabled
        await pilot.click("#msplat")
        assert app.pending_request is not None
        assert app.pending_request.stage == "msplat"
