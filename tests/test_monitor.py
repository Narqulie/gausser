from __future__ import annotations

from types import SimpleNamespace

import pytest

import monitor_and_run as monitor


def test_invalid_existing_model_falls_back_before_training(monkeypatch, scene):
    monkeypatch.setattr(monitor, "has_colmap_structure", lambda _: True)
    monkeypatch.setattr(monitor, "validate_colmap", lambda _: False)

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], desc: str, *, cwd) -> bool:
        calls.append(cmd)
        return False

    monkeypatch.setattr(monitor, "run", fake_run)

    try:
        monitor.main([str(scene)])
    except SystemExit:
        pass

    assert calls, "an invalid reconstruction must trigger the COLMAP fallback"
    assert calls[0][0] == "colmapper"


def test_msplat_trainer_runs_after_valid_colmap(monkeypatch, scene):
    monkeypatch.setattr(monitor, "has_colmap_structure", lambda _: True)
    monkeypatch.setattr(monitor, "validate_colmap", lambda _: True)
    monkeypatch.setattr(monitor, "count_images", lambda _: 1)

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], desc: str, *, cwd) -> bool:
        calls.append(cmd)
        return True

    monkeypatch.setattr(monitor, "run", fake_run)

    monitor.main([
        str(scene), "--trainer", "msplat", "--msplat-downscale-factor", "4",
    ])

    assert calls == [[
        "msplatter", str(scene.resolve()), "--iterations", "30000",
        "--downscale-factor", "4.0", "--allow-unmasked",
    ]]


def test_validate_colmap_uses_model_reader_once(monkeypatch, scene):
    monkeypatch.setattr(monitor, "_find_model_dir", lambda _: scene)
    monkeypatch.setattr(monitor, "_read_model_stats", lambda _: SimpleNamespace(images=1, points=2))
    monkeypatch.setattr(monitor, "count_images", lambda _: 1)

    assert monitor.validate_colmap(scene)


def test_monitor_image_count_excludes_appledouble_sidecars(scene):
    (scene / "images" / "._frame_0001.jpg").write_bytes(b"metadata")

    assert monitor.count_images(scene) == 1


def test_export_monitor_snapshots_exclude_appledouble_sidecars(scene):
    metadata = scene / "images" / "._frame_0001.jpg"
    metadata.write_bytes(b"metadata")

    assert metadata not in monitor.find_files(scene)
    assert metadata not in monitor.get_mtime_snapshot(scene)


def test_msplat_trainer_requires_explicit_downscale_factor(scene):
    with pytest.raises(SystemExit, match="2"):
        monitor.main([str(scene), "--trainer", "msplat"])
