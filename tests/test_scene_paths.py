from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from gausser.scene import SceneError, ScenePaths, initialize_scene
from gausser.ui import count_images, list_images


def test_paths_are_derived_only_from_explicit_scene(scene, monkeypatch, tmp_path: Path):
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)

    paths = ScenePaths.from_root(scene)

    assert paths.database == scene / "database.db"
    assert paths.sparse == scene / "sparse"
    assert paths.manifest == scene / ".gausser" / "manifest.json"
    assert paths.training_run("brush", "run-1") == scene / "runs" / "brush" / "run-1"


def test_initialize_scene_creates_only_canonical_empty_scene_layout(tmp_path: Path):
    root = tmp_path / "new-scene"
    root.mkdir()

    paths = initialize_scene(root)

    assert paths.images.is_dir()
    assert paths.runs.is_dir()
    assert paths.staging.is_dir()
    assert paths.manifest.read_text(encoding="utf-8") == '{\n  "stages": {},\n  "version": 1\n}\n'
    assert {path.name for path in root.iterdir()} == {"images", "runs", ".gausser"}


def test_initialize_scene_refuses_nonempty_directory(tmp_path: Path):
    root = tmp_path / "not-empty"
    root.mkdir()
    (root / "source.mov").write_bytes(b"video")

    with pytest.raises(SceneError, match="must be empty"):
        initialize_scene(root)

    assert not (root / "images").exists()


def test_images_ignore_macos_appledouble_sidecars(scene):
    sidecar = scene / "images" / "._frame_0001.jpg"
    sidecar.write_bytes(b"AppleDouble metadata")

    images = ScenePaths.from_root(scene).image_files()

    assert images == [scene / "images" / "frame_0001.jpg"]
    assert list_images(scene / "images") == images
    assert count_images(scene / "images") == 1


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
