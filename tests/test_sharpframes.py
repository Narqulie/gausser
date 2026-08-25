from __future__ import annotations

from pathlib import Path

from gausser.process import CommandResult
from sharpframes import cli


def test_sharpframes_publishes_staged_images_atomically(monkeypatch, scene: Path, tmp_path: Path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    (scene / "images" / "old.jpg").write_text("old")
    monkeypatch.setattr(cli.shutil, "which", lambda _: "/fake/sharp-frames")

    def fake_run(command, **kwargs):
        staged_images = Path(command[2])
        staged_images.mkdir(parents=True)
        (staged_images / "selected.jpg").write_text("new")
        return CommandResult(tuple(command), kwargs["cwd"], 0, 0.0, (), False)

    monkeypatch.setattr(cli, "run_command", fake_run)
    cli.main([str(source), "--scene", str(scene), "--force", "--", "--method", "best-n"])

    assert sorted(path.name for path in (scene / "images").iterdir()) == ["selected.jpg"]
    manifest = (scene / ".gausser" / "manifest.json").read_text()
    assert '"sharpframes"' in manifest
