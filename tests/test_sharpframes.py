from pathlib import Path

from sharpframes.cli import can_publish_into_scene


def test_sharpframes_allows_initialized_empty_images_directory(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()

    assert can_publish_into_scene(images)


def test_sharpframes_requires_force_for_existing_images(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "photo.jpg").write_bytes(b"image")

    assert not can_publish_into_scene(images)
