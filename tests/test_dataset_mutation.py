from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from gausser.scene import SceneError, ScenePaths, initialize_scene


def _scene(tmp_path: Path) -> ScenePaths:
    root = tmp_path / "scene"
    root.mkdir()
    return initialize_scene(root)


def _image(directory: Path, name: str, color: str = "white") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    Image.new("RGB", (4, 3), color=color).save(path)
    return path


def test_dataset_mutation_adds_and_removes_images_transactionally_then_archives_dependents(tmp_path: Path):
    from gausser.ingest import IngestPolicy, mutate_photo_dataset

    paths = _scene(tmp_path)
    _image(paths.images, "first.jpg")
    _image(paths.images, "second.jpg")
    _image(paths.brush_masks, "first.png")
    _image(paths.colmap_masks, "first.jpg.png")
    paths.database.write_bytes(b"database")
    (paths.sparse / "0").mkdir(parents=True)
    (paths.sparse / "0" / "cameras.bin").write_bytes(b"model")
    export = paths.runs / "brush" / "old" / "exports" / "scene.ply"
    export.parent.mkdir(parents=True)
    export.write_bytes(b"old export")

    additions = tmp_path / "additions"
    additions.mkdir()
    third = _image(additions, "third.jpg", "red")

    result = mutate_photo_dataset(
        paths,
        additions=[third],
        removals=["first.jpg"],
        policy=IngestPolicy.COPY,
    )

    assert [image.name for image in paths.image_files()] == ["second.jpg", "third.jpg"]
    assert result.added == ("third.jpg",)
    assert result.removed == ("first.jpg",)
    assert not paths.brush_masks.exists()
    assert not paths.colmap.exists()
    assert not paths.database.exists()
    assert not paths.sparse.exists()
    assert list(paths.runs.iterdir()) == []
    assert (result.archive / "masks" / "first.png").is_file()
    assert (result.archive / "runs" / "brush" / "old" / "exports" / "scene.ply").read_bytes() == b"old export"

    document = json.loads(paths.manifest.read_text())
    mutation = document["stages"]["dataset"]
    assert mutation["added"] == ["third.jpg"]
    assert mutation["removed"] == ["first.jpg"]
    assert mutation["invalidated_archive"] == str(paths.relative_path(result.archive))


def test_dataset_mutation_rejects_colliding_names_or_stems_without_change(tmp_path: Path):
    from gausser.ingest import IngestPolicy, mutate_photo_dataset

    paths = _scene(tmp_path)
    _image(paths.images, "image.jpg")
    original_manifest = paths.manifest.read_bytes()
    source = tmp_path / "source"
    source.mkdir()
    duplicate = _image(source, "image.png")

    with pytest.raises(SceneError, match="Duplicate image stem"):
        mutate_photo_dataset(paths, additions=[duplicate], removals=[], policy=IngestPolicy.COPY)

    assert [image.name for image in paths.image_files()] == ["image.jpg"]
    assert paths.manifest.read_bytes() == original_manifest


def test_dataset_mutation_rolls_back_images_and_dependents_when_manifest_write_fails(tmp_path: Path, monkeypatch):
    import gausser.ingest as ingest

    paths = _scene(tmp_path)
    _image(paths.images, "before.jpg")
    _image(paths.brush_masks, "before.png")
    source = tmp_path / "source"
    source.mkdir()
    added = _image(source, "after.jpg")
    original_manifest = paths.manifest.read_bytes()

    def fail_to_record(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("gausser.publish.transition_stage", fail_to_record)

    with pytest.raises(OSError, match="disk full"):
        ingest.mutate_photo_dataset(paths, additions=[added], removals=["before.jpg"], policy="copy")

    assert [image.name for image in paths.image_files()] == ["before.jpg"]
    assert (paths.brush_masks / "before.png").is_file()
    assert paths.manifest.read_bytes() == original_manifest
    assert not (paths.metadata / "invalidated").exists()
    attempts = list(paths.staging.iterdir())
    assert len(attempts) == 1
    assert (attempts[0] / "images" / "after.jpg").is_file()


def test_dataset_mutation_requires_direct_supported_additions_and_known_removals(tmp_path: Path):
    from gausser.ingest import mutate_photo_dataset

    paths = _scene(tmp_path)
    existing = _image(paths.images, "existing.jpg")
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("not an image")

    with pytest.raises(SceneError, match="not a supported image"):
        mutate_photo_dataset(paths, additions=[unsupported], removals=[])
    with pytest.raises(SceneError, match="not in this scene"):
        mutate_photo_dataset(paths, additions=[], removals=["missing.jpg"])
    with pytest.raises(SceneError, match="direct filename"):
        mutate_photo_dataset(paths, additions=[], removals=["images/existing.jpg"])

    assert paths.image_files() == [existing]
