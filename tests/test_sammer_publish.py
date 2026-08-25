from __future__ import annotations

from pathlib import Path

from sammer.cli import _publish_masks


def test_publish_masks_replaces_both_sets_without_stale_files(tmp_path: Path):
    final_brush = tmp_path / "masks"
    final_colmap = tmp_path / "colmap" / "masks"
    final_brush.mkdir(parents=True)
    final_colmap.mkdir(parents=True)
    (final_brush / "stale.png").write_text("old")
    (final_colmap / "stale.jpg.png").write_text("old")

    staged_brush = tmp_path / "staging" / "brush"
    staged_colmap = tmp_path / "staging" / "colmap"
    staged_brush.mkdir(parents=True)
    staged_colmap.mkdir(parents=True)
    (staged_brush / "fresh.png").write_text("new")
    (staged_colmap / "fresh.jpg.png").write_text("new")

    _publish_masks(staged_brush, staged_colmap, final_brush, final_colmap)

    assert sorted(path.name for path in final_brush.iterdir()) == ["fresh.png"]
    assert sorted(path.name for path in final_colmap.iterdir()) == ["fresh.jpg.png"]
