from __future__ import annotations

from pathlib import Path

import pytest

from colmapper import cli
from colmapper.cli import ColmapError, _prepare_artifacts, resolve_gpu


def test_macos_defaults_to_cpu(monkeypatch):
    monkeypatch.setattr("colmapper.cli.sys.platform", "darwin")
    assert not resolve_gpu(no_gpu=False)


def test_no_gpu_flag_always_disables_gpu(monkeypatch):
    monkeypatch.setattr("colmapper.cli.sys.platform", "linux")
    assert not resolve_gpu(no_gpu=True)


def test_existing_artifacts_require_an_explicit_policy(tmp_path: Path):
    database = tmp_path / "database.db"
    database.write_text("old")
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    (sparse / "partial").write_text("old")

    with pytest.raises(ColmapError, match="--existing"):
        _prepare_artifacts(database, sparse, "error")

    _prepare_artifacts(database, sparse, "resume")
    assert database.exists()
    assert sparse.exists()

    _prepare_artifacts(database, sparse, "replace")
    assert not database.exists()
    assert not sparse.exists()


def test_vocab_tree_matching_requires_an_existing_tree(monkeypatch, scene):
    monkeypatch.setattr(cli.shutil, "which", lambda _: "/fake/colmap")

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--scene", str(scene), "--matcher", "vocab_tree"])

    assert exc_info.value.code == 1
