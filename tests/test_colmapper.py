from __future__ import annotations

import json
from pathlib import Path

import pytest

from colmapper import cli
from colmapper.pipeline import ColmapError
from gausser.colmap_model import ModelStats
from gausser.scene import ScenePaths


def _write_binary_model(output: Path) -> Path:
    model = output / "0"
    model.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (model / name).write_bytes(name.encode())
    return model


def test_explicit_scene_stages_then_publishes_colmap_artifacts_with_provenance(
    monkeypatch, scene, tmp_path: Path
):
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    monkeypatch.setattr(cli, "resolve_command", lambda _: "/fake/colmap")
    monkeypatch.setattr(ScenePaths, "new_run_id", lambda _self: "colmap-run")

    calls: list[tuple[str, tuple, dict]] = []

    def extract(*args, **kwargs):
        calls.append(("extract", args, kwargs))
        Path(args[1]).write_bytes(b"new database")

    def match(*args, **kwargs):
        calls.append(("match", args, kwargs))

    def reconstruct(*args, **kwargs):
        calls.append(("map", args, kwargs))
        _write_binary_model(Path(args[3]))

    monkeypatch.setattr(cli, "feature_extraction", extract)
    monkeypatch.setattr(cli, "feature_matching", match)
    monkeypatch.setattr(cli, "sparse_reconstruction", reconstruct)
    monkeypatch.setattr(cli, "_pick_best_model", lambda output: output / "0")
    monkeypatch.setattr(cli, "read_model_stats", lambda _: ModelStats("binary", 1, 7))
    monkeypatch.setattr(cli, "_colmap_version", lambda *_args: "COLMAP 3.13.0")

    cli.main(["--scene", str(scene), "--colmap", "fakecolmap", "--matcher", "sequential", "--no-gpu"])

    attempt = scene / ".gausser" / "staging" / "colmap-colmap-run"
    assert calls[0][1][1] == attempt / "database.db"
    assert calls[0][1][2] == scene / "images"
    assert calls[1][1][1] == attempt / "database.db"
    assert calls[2][1][1] == attempt / "database.db"
    assert calls[2][1][3] == attempt / "sparse"
    assert scene.joinpath("database.db").read_bytes() == b"new database"
    assert scene.joinpath("sparse", "0", "cameras.bin").is_file()

    manifest = json.loads((scene / ".gausser" / "manifest.json").read_text(encoding="utf-8"))
    record = manifest["stages"]["colmap"]
    assert record["parameters"] == {
        "existing": "error",
        "gpu": False,
        "matcher": "sequential",
        "vocab_tree": None,
    }
    assert record["tool"] == "/fake/colmap"
    assert record["tool_version"] == "COLMAP 3.13.0"
    assert record["model"] == {"format": "binary", "registered_images": 1, "points": 7}
    assert record["outputs"] == {"database": "database.db", "sparse_model": "sparse/0"}


def test_failed_colmapper_replace_preserves_canonical_reconstruction_and_runs(monkeypatch, scene):
    (scene / "database.db").write_bytes(b"old database")
    _write_binary_model(scene / "sparse")
    export = scene / "runs" / "brush" / "old" / "exports" / "old.ply"
    export.parent.mkdir(parents=True)
    export.write_bytes(b"old run")
    monkeypatch.setattr(cli, "resolve_command", lambda _: "/fake/colmap")
    monkeypatch.setattr(ScenePaths, "new_run_id", lambda _self: "colmap-run")

    def fail_extract(*_args, **_kwargs):
        raise ColmapError("synthetic failure")

    monkeypatch.setattr(cli, "feature_extraction", fail_extract)

    with pytest.raises(SystemExit) as exc_info:
        cli.main([
            "--scene", str(scene), "--colmap", "fakecolmap", "--matcher", "sequential",
            "--existing", "replace",
        ])

    assert exc_info.value.code == 1
    assert (scene / "database.db").read_bytes() == b"old database"
    assert (scene / "sparse" / "0" / "cameras.bin").is_file()
    assert export.read_bytes() == b"old run"


def test_colmapper_resume_copies_existing_artifacts_into_staging(monkeypatch, scene):
    (scene / "database.db").write_bytes(b"old database")
    _write_binary_model(scene / "sparse")
    monkeypatch.setattr(cli, "resolve_command", lambda _: "/fake/colmap")
    monkeypatch.setattr(ScenePaths, "new_run_id", lambda _self: "colmap-run")

    def inspect_staged_resume(_colmap, database, *_args, **_kwargs):
        assert database != scene / "database.db"
        assert database.read_bytes() == b"old database"
        assert database.parent.joinpath("sparse", "0", "cameras.bin").is_file()
        raise ColmapError("stop after inspection")

    monkeypatch.setattr(cli, "feature_extraction", inspect_staged_resume)

    with pytest.raises(SystemExit) as exc_info:
        cli.main([
            "--scene", str(scene), "--colmap", "fakecolmap", "--matcher", "sequential",
            "--existing", "resume",
        ])

    assert exc_info.value.code == 1
    assert (scene / "database.db").read_bytes() == b"old database"
    assert (scene / "sparse" / "0" / "cameras.bin").is_file()


def test_colmapper_rejects_mapper_output_without_a_valid_model(monkeypatch, scene):
    monkeypatch.setattr(cli, "resolve_command", lambda _: "/fake/colmap")
    monkeypatch.setattr(cli, "feature_extraction", lambda *args, **kwargs: Path(args[1]).write_bytes(b"db"))
    monkeypatch.setattr(cli, "feature_matching", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "sparse_reconstruction", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--scene", str(scene), "--colmap", "fakecolmap", "--matcher", "sequential"])

    assert exc_info.value.code == 1
