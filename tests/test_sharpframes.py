import json
from pathlib import Path

from PIL import Image
import pytest

from gausser.process import CommandResult
from gausser.scene import initialize_scene
from sharpframes.cli import can_publish_into_scene, main, sharp_frames_command


def test_sharpframes_uses_current_upstream_overwrite_flag(tmp_path: Path):
    assert sharp_frames_command("sharp-frames", tmp_path / "input.mov", tmp_path / "output", ["--fps", "4"]) == [
        "sharp-frames", str(tmp_path / "input.mov"), str(tmp_path / "output"), "--force-overwrite", "--fps", "4",
    ]


def test_sharpframes_allows_initialized_empty_images_directory(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()

    assert can_publish_into_scene(images)


def test_sharpframes_records_engine_resolved_settings_in_its_stage_manifest(tmp_path: Path, monkeypatch):
    scene = tmp_path / "scene"
    scene.mkdir()
    initialize_scene(scene)
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (3, 2)).save(source / "input.jpg")

    def fake_run(command, **_kwargs):
        output = Path(command[2])
        output.mkdir(parents=True)
        Image.new("RGB", (3, 2)).save(output / "selected.jpg")
        return CommandResult(tuple(command), scene, 0, 0.0, (), False)

    monkeypatch.setattr("sharpframes.cli.run_command", fake_run)
    monkeypatch.setattr("sharpframes.cli.shutil.which", lambda _command: "/tmp/sharp-frames")
    settings = {"specification_version": 1, "preset": "custom", "values": {"fps": 2}}

    main([str(source), "--scene", str(scene), "--studio-settings", json.dumps(settings)])

    manifest = json.loads((scene / ".gausser" / "manifest.json").read_text())
    assert manifest["stages"]["sharpframes"]["studio_settings"] == settings


def test_sharpframes_rolls_back_images_when_manifest_recording_fails(tmp_path: Path, monkeypatch):
    scene = tmp_path / "scene"
    scene.mkdir()
    initialize_scene(scene)
    Image.new("RGB", (3, 2), color="red").save(scene / "images" / "old.jpg")
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (3, 2)).save(source / "input.jpg")

    def fake_run(command, **_kwargs):
        output = Path(command[2])
        output.mkdir(parents=True)
        Image.new("RGB", (3, 2), color="blue").save(output / "selected.jpg")
        return CommandResult(tuple(command), scene, 0, 0.0, (), False)

    monkeypatch.setattr("sharpframes.cli.run_command", fake_run)
    monkeypatch.setattr("sharpframes.cli.shutil.which", lambda _command: "/tmp/sharp-frames")
    monkeypatch.setattr("sharpframes.cli.record_stage", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(OSError, match="disk full"):
        main([str(source), "--scene", str(scene), "--force"])

    assert (scene / "images" / "old.jpg").is_file()
    assert not (scene / "images" / "selected.jpg").exists()


def test_sharpframes_requires_force_for_existing_images(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "photo.jpg").write_bytes(b"image")

    assert not can_publish_into_scene(images)
