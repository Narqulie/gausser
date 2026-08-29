from __future__ import annotations

import json
from pathlib import Path

import pytest

from gausser.publish import publish_stage
from gausser.scene import ScenePaths, initialize_scene


def _initialized_scene(tmp_path: Path) -> ScenePaths:
    root = tmp_path / "scene"
    root.mkdir()
    return initialize_scene(root)


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _record_completed_stages(paths: ScenePaths) -> None:
    document = json.loads(paths.manifest.read_text(encoding="utf-8"))
    document["stages"] = {
        "ingest": {"completed_at": "before"},
        "sammer": {"completed_at": "before"},
        "colmap": {"completed_at": "before"},
        "brush": {"completed_at": "before"},
        "msplat": {"completed_at": "before"},
        "lod": {"completed_at": "before"},
        "supersplat": {"completed_at": "before"},
    }
    paths.manifest.write_text(json.dumps(document), encoding="utf-8")


def test_mask_generation_archives_exact_dependents_and_prunes_current_stage_records(tmp_path: Path):
    paths = _initialized_scene(tmp_path)
    _write(paths.images / "frame.jpg", "image")
    _write(paths.brush_masks / "old.png", "old brush mask")
    _write(paths.colmap_masks / "old.jpg.png", "old colmap mask")
    _write(paths.database, "old database")
    _write(paths.sparse / "0" / "cameras.bin", "old model")
    _write(paths.runs / "brush" / "old" / "exports" / "old.ply", "old run")
    _record_completed_stages(paths)

    attempt = paths.staging_attempt("sammer", "new")
    staged_brush = _write(attempt / "masks" / "new.png", "new brush mask").parent
    staged_colmap = _write(attempt / "colmap" / "masks" / "new.jpg.png", "new colmap mask").parent

    archive = publish_stage(
        paths,
        "sammer",
        generation="masks",
        replacements=((staged_brush, paths.brush_masks), (staged_colmap, paths.colmap_masks)),
        data={"parameters": {"prompt": "statue"}},
    )

    assert archive is not None
    assert (paths.brush_masks / "new.png").read_text() == "new brush mask"
    assert (paths.colmap_masks / "new.jpg.png").read_text() == "new colmap mask"
    assert (archive / "masks" / "old.png").read_text() == "old brush mask"
    assert (archive / "colmap" / "masks" / "old.jpg.png").read_text() == "old colmap mask"
    assert (archive / "database.db").read_text() == "old database"
    assert (archive / "sparse" / "0" / "cameras.bin").read_text() == "old model"
    assert (archive / "runs" / "brush" / "old" / "exports" / "old.ply").read_text() == "old run"
    assert paths.images.joinpath("frame.jpg").is_file()
    assert paths.runs.is_dir() and not any(paths.runs.iterdir())

    document = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert set(document["stages"]) == {"ingest", "sammer"}
    assert document["stages"]["sammer"]["parameters"] == {"prompt": "statue"}
    assert document["stages"]["sammer"]["invalidated_archive"] == str(paths.relative_path(archive))


def test_stage_publication_restores_every_artifact_and_manifest_when_metadata_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _initialized_scene(tmp_path)
    _write(paths.brush_masks / "old.png", "old brush mask")
    _write(paths.colmap_masks / "old.jpg.png", "old colmap mask")
    _write(paths.database, "old database")
    _write(paths.sparse / "0" / "cameras.bin", "old model")
    _write(paths.runs / "brush" / "old" / "exports" / "old.ply", "old run")
    original_manifest = paths.manifest.read_bytes()

    attempt = paths.staging_attempt("sammer", "new")
    staged_brush = _write(attempt / "masks" / "new.png", "new brush mask").parent
    staged_colmap = _write(attempt / "colmap" / "masks" / "new.jpg.png", "new colmap mask").parent

    def fail_transition(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("gausser.publish.transition_stage", fail_transition)

    with pytest.raises(OSError, match="disk full"):
        publish_stage(
            paths,
            "sammer",
            generation="masks",
            replacements=((staged_brush, paths.brush_masks), (staged_colmap, paths.colmap_masks)),
            data={},
        )

    assert (paths.brush_masks / "old.png").read_text() == "old brush mask"
    assert (paths.colmap_masks / "old.jpg.png").read_text() == "old colmap mask"
    assert paths.database.read_text() == "old database"
    assert (paths.sparse / "0" / "cameras.bin").read_text() == "old model"
    assert (paths.runs / "brush" / "old" / "exports" / "old.ply").read_text() == "old run"
    assert paths.manifest.read_bytes() == original_manifest
    assert (staged_brush / "new.png").is_file()
    assert (staged_colmap / "new.jpg.png").is_file()
    assert not (paths.metadata / "invalidated").exists()


def test_stage_publication_rolls_back_when_interrupted_during_manifest_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _initialized_scene(tmp_path)
    _write(paths.brush_masks / "old.png", "old brush mask")
    _write(paths.colmap_masks / "old.jpg.png", "old colmap mask")
    original_manifest = paths.manifest.read_bytes()

    attempt = paths.staging_attempt("sammer", "interrupted")
    staged_brush = _write(attempt / "masks" / "new.png", "new brush mask").parent
    staged_colmap = _write(attempt / "colmap" / "masks" / "new.jpg.png", "new colmap mask").parent

    def interrupt_transition(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("gausser.publish.transition_stage", interrupt_transition)

    with pytest.raises(KeyboardInterrupt):
        publish_stage(
            paths,
            "sammer",
            generation="masks",
            replacements=((staged_brush, paths.brush_masks), (staged_colmap, paths.colmap_masks)),
            data={},
        )

    assert (paths.brush_masks / "old.png").read_text() == "old brush mask"
    assert (paths.colmap_masks / "old.jpg.png").read_text() == "old colmap mask"
    assert paths.manifest.read_bytes() == original_manifest
    assert (staged_brush / "new.png").read_text() == "new brush mask"
    assert (staged_colmap / "new.jpg.png").read_text() == "new colmap mask"
    assert not (paths.metadata / "invalidated").exists()


def test_reconstruction_generation_preserves_images_and_masks_but_archives_runs(tmp_path: Path):
    paths = _initialized_scene(tmp_path)
    _write(paths.images / "frame.jpg", "image")
    _write(paths.brush_masks / "frame.png", "mask")
    _write(paths.colmap_masks / "frame.jpg.png", "mask")
    _write(paths.database, "old database")
    _write(paths.sparse / "0" / "cameras.bin", "old model")
    _write(paths.runs / "msplat" / "old" / "exports" / "old.ply", "old run")

    attempt = paths.staging_attempt("colmap", "new")
    staged_database = _write(attempt / "database.db", "new database")
    staged_sparse = _write(attempt / "sparse" / "0" / "cameras.bin", "new model").parents[1]

    archive = publish_stage(
        paths,
        "colmap",
        generation="reconstruction",
        replacements=((staged_database, paths.database), (staged_sparse, paths.sparse)),
        data={"model": {"registered_images": 1}},
    )

    assert archive is not None
    assert paths.database.read_text() == "new database"
    assert (paths.sparse / "0" / "cameras.bin").read_text() == "new model"
    assert (archive / "database.db").read_text() == "old database"
    assert (archive / "sparse" / "0" / "cameras.bin").read_text() == "old model"
    assert (archive / "runs" / "msplat" / "old" / "exports" / "old.ply").is_file()
    assert (paths.images / "frame.jpg").is_file()
    assert (paths.brush_masks / "frame.png").is_file()
    assert (paths.colmap_masks / "frame.jpg.png").is_file()
