from __future__ import annotations

import pytest


def test_msplat_rejects_text_colmap_model(scene):
    from msplatter import cli

    model = scene / "sparse" / "0"
    model.mkdir(parents=True)
    (model / "cameras.txt").write_text("1 PINHOLE 1 1 1 1 1 1\n")
    (model / "images.txt").write_text("1 1 0 0 0 0 0 0 1 frame_0001.jpg\n\n")
    (model / "points3D.txt").write_text("1 0 0 0 1 1 1 0.1 1 1\n")

    with pytest.raises(SystemExit) as exc_info:
        cli.main([str(scene), "--allow-unmasked"])

    assert exc_info.value.code == 2
