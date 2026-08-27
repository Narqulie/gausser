from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from gausser.dataset import list_dataset_images
from gausser.scene import ScenePaths, initialize_scene


def _image(path: Path, name: str, color: str) -> None:
    Image.new("RGB", (4, 3), color=color).save(path / name)


def test_dataset_images_are_scene_owned_paginated_and_ignore_appledouble(scene: Path):
    _image(scene / "images", "c.jpg", "red")
    _image(scene / "images", "a.png", "blue")
    _image(scene / "images", "b.jpg", "green")
    (scene / "images" / "._a.png").write_bytes(b"metadata")

    page = list_dataset_images(ScenePaths.from_root(scene), offset=1, limit=1)

    assert page.total_count == 4
    assert page.revision
    assert page.next_offset == 2
    assert [(image.relative_path, image.name) for image in page.images] == [("images/b.jpg", "b.jpg")]

    assert list_dataset_images(ScenePaths.from_root(scene), offset=2, limit=1, revision=page.revision).revision == page.revision
    with pytest.raises(ValueError, match="dataset changed"):
        list_dataset_images(ScenePaths.from_root(scene), offset=2, limit=1, revision="obsolete")


def test_dataset_images_reject_invalid_pagination(scene: Path):
    paths = ScenePaths.from_root(scene)

    for offset, limit in ((-1, 1), (0, 0), (0, 501)):
        with pytest.raises(ValueError):
            list_dataset_images(paths, offset=offset, limit=limit)


def test_dataset_images_expose_only_matching_scene_owned_masks_and_revision_changes(tmp_path: Path):
    root = tmp_path / "scene"
    root.mkdir()
    paths = initialize_scene(root)
    Image.new("RGB", (2, 2)).save(paths.images / "frame.jpg")

    initial = list_dataset_images(paths)
    assert initial.images[0].mask_relative_path is None

    paths.brush_masks.mkdir()
    Image.new("L", (2, 2), color=255).save(paths.brush_masks / "frame.png")
    with_mask = list_dataset_images(paths)

    assert with_mask.images[0].mask_relative_path == "masks/frame.png"
    assert with_mask.revision != initial.revision
