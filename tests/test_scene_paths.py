from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from gausser.scene import SceneError, ScenePaths


def test_paths_are_derived_only_from_explicit_scene(scene, monkeypatch, tmp_path: Path):
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)

    paths = ScenePaths.from_root(scene)

    assert paths.database == scene / "database.db"
    assert paths.sparse == scene / "sparse"
    assert paths.manifest == scene / ".gausser" / "manifest.json"
    assert paths.training_run("brush", "run-1") == scene / "runs" / "brush" / "run-1"


def test_images_reject_duplicate_stems(scene):
    Image.new("RGB", (1, 1)).save(scene / "images" / "frame_0001.png")

    with pytest.raises(SceneError, match="Duplicate image stem"):
        ScenePaths.from_root(scene).image_files()


def test_images_reject_nested_files(scene):
    nested = scene / "images" / "subset"
    nested.mkdir()
    Image.new("RGB", (1, 1)).save(nested / "frame_0002.jpg")

    with pytest.raises(SceneError, match="must be flat"):
        ScenePaths.from_root(scene).image_files()
