from __future__ import annotations

from gausser.preflight import inspect_scene
from gausser.scene import ScenePaths


def test_preflight_guides_a_scene_without_colmap(scene):
    statuses = {item.stage: item for item in inspect_scene(ScenePaths.from_root(scene))}

    assert statuses["images"].state == "ready"
    assert statuses["sammer"].state == "ready"
    assert statuses["colmap"].state == "ready"
    assert statuses["brush"].state == "unavailable"
    assert statuses["msplat"].state == "unavailable"
    assert "unmasked" in statuses["msplat"].detail
