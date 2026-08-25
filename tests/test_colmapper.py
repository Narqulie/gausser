from __future__ import annotations

from pathlib import Path

import pytest

from colmapper import cli
from gausser.colmap_model import ModelStats


def test_explicit_scene_owns_colmap_artifacts(monkeypatch, scene, tmp_path: Path):
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    monkeypatch.setattr(cli.shutil, "which", lambda _: "/fake/colmap")

    calls: list[tuple[str, tuple, dict]] = []

    def record(name):
        def step(*args, **kwargs):
            calls.append((name, args, kwargs))
        return step

    monkeypatch.setattr(cli, "feature_extraction", record("extract"))
    monkeypatch.setattr(cli, "feature_matching", record("match"))
    monkeypatch.setattr(cli, "sparse_reconstruction", record("map"))
    model_path = scene / "sparse" / "0"
    monkeypatch.setattr(cli, "_pick_best_model", lambda _: model_path)
    monkeypatch.setattr(cli, "read_model_stats", lambda _: ModelStats("binary", 1, 1))

    cli.main(["--scene", str(scene), "--colmap", "fakecolmap", "--matcher", "sequential"])

    assert calls[0][1][1] == scene / "database.db"
    assert calls[0][1][2] == scene / "images"
    assert calls[1][1][1] == scene / "database.db"
    assert calls[2][1][1] == scene / "database.db"
    assert calls[2][1][3] == scene / "sparse"


def test_colmapper_rejects_mapper_output_without_a_valid_model(monkeypatch, scene):
    monkeypatch.setattr(cli.shutil, "which", lambda _: "/fake/colmap")
    monkeypatch.setattr(cli, "feature_extraction", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "feature_matching", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "sparse_reconstruction", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--scene", str(scene), "--colmap", "fakecolmap", "--matcher", "sequential"])

    assert exc_info.value.code == 1
