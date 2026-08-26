from __future__ import annotations

from pathlib import Path

from gausser.artifacts import list_artifacts
from gausser.scene import ScenePaths


def test_list_artifacts_reports_only_scene_owned_deliverables(scene: Path, tmp_path: Path):
    brush_export = scene / "runs" / "brush" / "run-a" / "exports" / "export_15000.ply"
    brush_export.parent.mkdir(parents=True)
    brush_export.write_bytes(b"ply")
    lod_meta = scene / "runs" / "lod" / "run-b" / "lod-meta.json"
    lod_meta.parent.mkdir(parents=True)
    lod_meta.write_text("{}", encoding="utf-8")
    ignored = scene / "runs" / "brush" / "run-a" / "checkpoint.msplat"
    ignored.write_bytes(b"checkpoint")
    outside = tmp_path / "outside.splat"
    outside.write_bytes(b"splat")
    (scene / "runs" / "brush" / "run-a" / "exports" / "outside.splat").symlink_to(outside)

    artifacts = list_artifacts(ScenePaths.from_root(scene))

    assert [(artifact.relative_path, artifact.kind, artifact.format) for artifact in artifacts] == [
        ("runs/brush/run-a/exports/export_15000.ply", "brush", "ply"),
        ("runs/lod/run-b/lod-meta.json", "lod", "lod"),
    ]
    assert artifacts[0].size_bytes == 3


def test_list_artifacts_handles_untrained_initialized_scene(tmp_path: Path):
    scene = tmp_path / "scene"
    scene.mkdir()

    assert list_artifacts(ScenePaths.from_root(scene)) == []
