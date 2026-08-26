from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from gausser.scene import SceneError, ScenePaths, initialize_scene


def _initialized_scene(tmp_path: Path) -> ScenePaths:
    root = tmp_path / "scene"
    root.mkdir()
    return initialize_scene(root)


def _photo(directory: Path, name: str, color: str = "white") -> Path:
    path = directory / name
    Image.new("RGB", (1, 1), color=color).save(path)
    return path


def test_ingest_copies_supported_source_images_and_records_provenance(tmp_path: Path):
    from gausser.ingest import IngestPolicy, ingest_photo_directory

    paths = _initialized_scene(tmp_path)
    source = tmp_path / "camera"
    source.mkdir()
    first = _photo(source, "frame_0001.jpg")
    _photo(source, "frame_0002.PNG")
    (source / "._frame_0001.jpg").write_bytes(b"AppleDouble")
    (source / "notes.txt").write_text("not a photo")
    nested = source / "nested"
    nested.mkdir()
    _photo(nested, "frame_0003.jpg")

    result = ingest_photo_directory(paths, source, policy=IngestPolicy.COPY)

    assert result.source == source.resolve()
    assert result.policy is IngestPolicy.COPY
    assert [image.name for image in result.images] == ["frame_0001.jpg", "frame_0002.PNG"]
    assert [image.name for image in paths.image_files()] == ["frame_0001.jpg", "frame_0002.PNG"]
    assert not (paths.images / "frame_0001.jpg").is_symlink()
    first.write_bytes(b"changed source")
    assert (paths.images / "frame_0001.jpg").read_bytes() != first.read_bytes()

    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    ingest = manifest["stages"]["ingest"]
    assert ingest["source"] == str(source.resolve())
    assert ingest["policy"] == "copy"
    assert set(ingest["inputs"]) == {"frame_0001.jpg", "frame_0002.PNG"}
    assert ingest["outputs"] == {"images": "images", "count": 2}


def test_ingest_links_images_when_explicitly_requested(tmp_path: Path):
    from gausser.ingest import IngestPolicy, ingest_photo_directory

    paths = _initialized_scene(tmp_path)
    source = tmp_path / "camera"
    source.mkdir()
    original = _photo(source, "frame_0001.jpg")

    result = ingest_photo_directory(paths, source, policy=IngestPolicy.LINK)

    imported = result.images[0]
    assert imported.is_symlink()
    assert imported.resolve() == original.resolve()
    assert json.loads(paths.manifest.read_text(encoding="utf-8"))["stages"]["ingest"]["policy"] == "link"


def test_ingest_refuses_a_nonempty_image_directory_without_changing_the_scene(tmp_path: Path):
    from gausser.ingest import IngestPolicy, ingest_photo_directory

    paths = _initialized_scene(tmp_path)
    existing = _photo(paths.images, "already_here.jpg")
    original_manifest = paths.manifest.read_bytes()
    source = tmp_path / "camera"
    source.mkdir()
    _photo(source, "frame_0001.jpg")

    with pytest.raises(SceneError, match="must be empty"):
        ingest_photo_directory(paths, source, policy=IngestPolicy.COPY)

    assert existing.read_bytes()
    assert paths.manifest.read_bytes() == original_manifest


def test_ingest_discards_partial_staging_when_copy_fails(tmp_path: Path, monkeypatch):
    import gausser.ingest as ingest

    paths = _initialized_scene(tmp_path)
    source = tmp_path / "camera"
    source.mkdir()
    _photo(source, "frame_0001.jpg")
    _photo(source, "frame_0002.jpg")
    copy2 = ingest.shutil.copy2
    calls = 0

    def fail_on_second_copy(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("source disconnected")
        return copy2(*args, **kwargs)

    monkeypatch.setattr(ingest.shutil, "copy2", fail_on_second_copy)

    with pytest.raises(OSError, match="source disconnected"):
        ingest.ingest_photo_directory(paths, source, policy=ingest.IngestPolicy.COPY)

    assert list(paths.images.iterdir()) == []
    assert list(paths.staging.iterdir()) == []


def test_ingest_rolls_back_published_images_when_manifest_recording_fails(tmp_path: Path, monkeypatch):
    import gausser.ingest as ingest

    paths = _initialized_scene(tmp_path)
    original_manifest = paths.manifest.read_bytes()
    source = tmp_path / "camera"
    source.mkdir()
    _photo(source, "frame_0001.jpg")

    def fail_to_record(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(ingest, "record_stage", fail_to_record)

    with pytest.raises(OSError, match="disk full"):
        ingest.ingest_photo_directory(paths, source, policy=ingest.IngestPolicy.COPY)

    assert list(paths.images.iterdir()) == []
    assert paths.manifest.read_bytes() == original_manifest
    assert list(paths.staging.iterdir()) == []


def test_ingest_requires_an_initialized_scene_and_at_least_one_supported_photo(tmp_path: Path):
    from gausser.ingest import IngestPolicy, ingest_photo_directory

    source = tmp_path / "camera"
    source.mkdir()
    (source / "._frame_0001.jpg").write_bytes(b"AppleDouble")
    uninitialized = ScenePaths.from_root(tmp_path / "not-a-scene")

    with pytest.raises(SceneError, match="not initialized"):
        ingest_photo_directory(uninitialized, source, policy=IngestPolicy.COPY)

    paths = _initialized_scene(tmp_path)
    with pytest.raises(SceneError, match="no supported images"):
        ingest_photo_directory(paths, source, policy=IngestPolicy.COPY)

    assert list(paths.images.iterdir()) == []
