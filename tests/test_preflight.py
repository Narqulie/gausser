from __future__ import annotations

from gausser.preflight import inspect_scene
from gausser.scene import ScenePaths


def test_preflight_blocks_sharpframes_when_images_are_already_owned(scene):
    (scene / "images" / "photo.jpg").write_bytes(b"image")

    statuses = {item.stage: item for item in inspect_scene(ScenePaths.from_root(scene))}

    assert statuses["sharpframes"].state == "unavailable"
    assert "replace" in statuses["sharpframes"].detail.lower()


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
