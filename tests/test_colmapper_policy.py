from __future__ import annotations

from pathlib import Path

import pytest

from colmapper import cli
from colmapper.cli import ColmapError, _stage_artifacts, resolve_gpu
from gausser.scene import ScenePaths


def test_macos_defaults_to_cpu(monkeypatch):
    monkeypatch.setattr("colmapper.cli.sys.platform", "darwin")
    assert not resolve_gpu(no_gpu=False)


def test_no_gpu_flag_always_disables_gpu(monkeypatch):
    monkeypatch.setattr("colmapper.cli.sys.platform", "linux")
    assert not resolve_gpu(no_gpu=True)


def test_existing_artifacts_require_an_explicit_non_destructive_staging_policy(tmp_path: Path):
    paths = ScenePaths.from_root(tmp_path)
    paths.database.write_text("old")
    paths.sparse.mkdir()
    (paths.sparse / "partial").write_text("old")

    with pytest.raises(ColmapError, match="--existing"):
        _stage_artifacts(paths, paths.staging_attempt("colmap", "error"), "error")

    staged_database, staged_sparse = _stage_artifacts(
        paths, paths.staging_attempt("colmap", "resume"), "resume"
    )
    assert staged_database.read_text() == "old"
    assert (staged_sparse / "partial").read_text() == "old"

    clean_database, clean_sparse = _stage_artifacts(
        paths, paths.staging_attempt("colmap", "replace"), "replace"
    )
    assert not clean_database.exists()
    assert not clean_sparse.exists()
    assert paths.database.read_text() == "old"
    assert (paths.sparse / "partial").read_text() == "old"


def test_vocab_tree_matching_requires_an_existing_tree(monkeypatch, scene):
    monkeypatch.setattr(cli, "resolve_command", lambda _: "/fake/colmap")

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--scene", str(scene), "--matcher", "vocab_tree"])

    assert exc_info.value.code == 2
