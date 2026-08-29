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


def test_sharpframes_records_engine_resolved_settings_in_its_stage_manifest(
    tmp_path: Path, monkeypatch, capsys
):
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
    monkeypatch.setattr("sharpframes.cli.resolve_command", lambda _command: "/tmp/sharp-frames")
    settings = {"specification_version": 1, "preset": "custom", "values": {"fps": 2}}

    main([str(source), "--scene", str(scene), "--studio-settings", json.dumps(settings)])

    output = capsys.readouterr().out
    assert "◈ SHARP FRAMES // FRAME SELECTION SIGNAL" in output
    assert "◆ FRAME SELECTION COMPLETE" in output
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
    monkeypatch.setattr("sharpframes.cli.resolve_command", lambda _command: "/tmp/sharp-frames")
    monkeypatch.setattr(
        "gausser.publish.transition_stage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        main([str(source), "--scene", str(scene), "--force"])

    assert (scene / "images" / "old.jpg").is_file()
    assert not (scene / "images" / "selected.jpg").exists()


def test_sharpframes_force_archives_every_artifact_derived_from_the_previous_images(
    tmp_path: Path, monkeypatch
):
    scene = tmp_path / "scene"
    scene.mkdir()
    paths = initialize_scene(scene)
    Image.new("RGB", (3, 2), color="red").save(paths.images / "old.jpg")
    paths.brush_masks.mkdir()
    paths.colmap_masks.mkdir(parents=True)
    Image.new("L", (3, 2)).save(paths.brush_masks / "old.png")
    Image.new("L", (3, 2)).save(paths.colmap_masks / "old.jpg.png")
    paths.database.write_bytes(b"database")
    (paths.sparse / "0").mkdir(parents=True)
    (paths.sparse / "0" / "cameras.bin").write_bytes(b"model")
    export = paths.runs / "brush" / "old" / "exports" / "old.ply"
    export.parent.mkdir(parents=True)
    export.write_bytes(b"old export")
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (3, 2), color="blue").save(source / "input.jpg")

    def fake_run(command, **_kwargs):
        output = Path(command[2])
        output.mkdir(parents=True)
        Image.new("RGB", (3, 2), color="blue").save(output / "selected.jpg")
        return CommandResult(tuple(command), scene, 0, 0.0, (), False)

    monkeypatch.setattr("sharpframes.cli.run_command", fake_run)
    monkeypatch.setattr("sharpframes.cli.resolve_command", lambda _command: "/tmp/sharp-frames")

    main([str(source), "--scene", str(scene), "--force"])

    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    archive = scene / manifest["stages"]["sharpframes"]["invalidated_archive"]
    assert [image.name for image in paths.image_files()] == ["selected.jpg"]
    assert not paths.brush_masks.exists()
    assert not paths.colmap.exists()
    assert not paths.database.exists()
    assert not paths.sparse.exists()
    assert paths.runs.is_dir() and not any(paths.runs.iterdir())
    assert (archive / "images" / "old.jpg").is_file()
    assert (archive / "masks" / "old.png").is_file()
    assert (archive / "colmap" / "masks" / "old.jpg.png").is_file()
    assert (archive / "database.db").read_bytes() == b"database"
    assert (archive / "sparse" / "0" / "cameras.bin").read_bytes() == b"model"
    assert (archive / "runs" / "brush" / "old" / "exports" / "old.ply").read_bytes() == b"old export"


def test_sharpframes_requires_force_for_existing_images(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "photo.jpg").write_bytes(b"image")

    assert not can_publish_into_scene(images)
