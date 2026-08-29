from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from gausser.scene_lock import acquire_scene_lock


def _lock_in_child(scene: Path) -> subprocess.Popen[str]:
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            """
from pathlib import Path
import sys
from gausser.scene_lock import acquire_scene_lock
with acquire_scene_lock(Path(sys.argv[1])):
    print('locked', flush=True)
    sys.stdin.readline()
""",
            str(scene),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert child.stdout is not None
    assert child.stdout.readline().strip() == "locked"
    return child


def _release(child: subprocess.Popen[str]) -> None:
    assert child.stdin is not None
    child.stdin.write("release\n")
    child.stdin.flush()
    child.wait(timeout=5)


def test_every_stage_cli_rejects_a_scene_owned_by_another_process(scene: Path, monkeypatch):
    from brusher import cli as brusher
    from colmapper import cli as colmapper
    from lodder import cli as lodder
    from msplatter import cli as msplatter
    from sammer import cli as sammer
    from sharpframes import cli as sharpframes

    source = scene / "runs" / "brush" / "source" / "exports" / "source.ply"
    source.parent.mkdir(parents=True)
    source.write_text("ply", encoding="utf-8")
    sharp_source = scene.parent / "source-images"
    sharp_source.mkdir()

    monkeypatch.setattr(sharpframes, "resolve_command", lambda _command: "/fake/sharp-frames")
    monkeypatch.setattr(colmapper, "resolve_command", lambda _command: "/fake/colmap")
    monkeypatch.setattr(brusher, "resolve_command", lambda _command: "/fake/brush")
    monkeypatch.setattr(lodder, "resolve_command", lambda _command: "/fake/splat-transform")

    invocations = [
        (sharpframes.main, [str(sharp_source), "--scene", str(scene), "--force"]),
        (sammer.main, ["--scene", str(scene), "--prompt", "statue"]),
        (colmapper.main, ["--scene", str(scene), "--matcher", "sequential"]),
        (brusher.main, [str(scene), "--total-train-iters", "10", "--export-every", "5"]),
        (msplatter.main, [str(scene), "--allow-unmasked"]),
        (lodder.main, [str(source), "--scene", str(scene)]),
    ]

    child = _lock_in_child(scene)
    try:
        for cli_main, arguments in invocations:
            with pytest.raises(SystemExit) as exc_info:
                cli_main(arguments)
            assert exc_info.value.code == 2
    finally:
        _release(child)


def test_sammer_rejects_a_missing_scene_without_creating_lock_metadata(tmp_path: Path):
    from sammer import cli

    missing = tmp_path / "typo-scene"

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--scene", str(missing), "--prompt", "statue"])

    assert exc_info.value.code == 2
    assert not missing.exists()


def test_direct_stage_cli_releases_its_scene_lock_after_exit(scene: Path, monkeypatch):
    from sharpframes import cli

    source = scene.parent / "source-images"
    source.mkdir()
    monkeypatch.setattr(cli, "resolve_command", lambda _command: "/fake/sharp-frames")

    with pytest.raises(SystemExit):
        cli.main([str(source), "--scene", str(scene)])

    with acquire_scene_lock(scene):
        pass
