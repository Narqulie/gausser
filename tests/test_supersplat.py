from pathlib import Path

from gausser.scene import ScenePaths
from gausser.supersplat import reimport_edited_export


def test_reimport_publishes_a_new_scene_owned_export_with_provenance(scene: Path, tmp_path: Path):
    source = scene / "runs" / "brush" / "source-run" / "exports" / "scene.ply"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"original")
    edited = tmp_path / "edited.sog"
    edited.write_bytes(b"edited")

    result = reimport_edited_export(ScenePaths.from_root(scene), source, edited, run_id="edited-run")

    assert result.export == scene / "runs" / "supersplat" / "edited-run" / "exports" / "edited.sog"
    assert result.export.read_bytes() == b"edited"
    assert result.provenance.is_file()
    assert '"source_artifact": "runs/brush/source-run/exports/scene.ply"' in result.provenance.read_text()


def test_reimport_removes_published_run_when_manifest_recording_fails(scene: Path, tmp_path: Path, monkeypatch):
    source = scene / "runs" / "brush" / "source-run" / "exports" / "scene.ply"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"original")
    edited = tmp_path / "edited.ply"
    edited.write_bytes(b"edited")

    def fail(*_args, **_kwargs):
        raise OSError("manifest unavailable")

    monkeypatch.setattr("gausser.supersplat.record_stage", fail)

    try:
        reimport_edited_export(ScenePaths.from_root(scene), source, edited, run_id="failed-run")
    except OSError:
        pass
    else:
        raise AssertionError("expected manifest failure")

    assert not (scene / "runs" / "supersplat" / "failed-run").exists()


def test_reimport_rejects_a_source_that_is_not_scene_owned_export(scene: Path, tmp_path: Path):
    edited = tmp_path / "edited.ply"
    edited.write_bytes(b"edited")

    try:
        reimport_edited_export(ScenePaths.from_root(scene), edited, edited)
    except ValueError as exc:
        assert "scene-owned export" in str(exc)
    else:
        raise AssertionError("expected a scene-ownership failure")
