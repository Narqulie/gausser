import os
import subprocess
import sys
from pathlib import Path

import pytest

from gausser.scene_lock import SCENE_LOCK_FD_ENV, SceneLock, SceneLockError, _lock_path, acquire_scene_lock


def test_engine_rejects_a_stage_when_another_process_owns_scene(scene: Path):
    from gausser.engine import GausserEngine

    child = _lock_in_child(scene)
    try:
        events: list[dict] = []
        GausserEngine().dispatch(
            {
                "id": "start",
                "method": "start",
                "params": {"scene": str(scene), "stage": "sammer", "options": {"prompt": "statue"}},
            },
            events.append,
        )
        assert events[0]["error"]["code"] == "scene_busy"
    finally:
        _release_child(child)


def test_scene_lock_excludes_another_engine_process(scene: Path):
    child = _lock_in_child(scene)
    try:
        with pytest.raises(SceneLockError, match="already owned"):
            with acquire_scene_lock(scene):
                pass
    finally:
        _release_child(child)


def test_scene_lock_is_owned_by_scene_metadata(scene: Path):
    with acquire_scene_lock(scene):
        assert _lock_path(scene) == scene.resolve() / ".gausser" / "scene.lock"
        assert _lock_path(scene).is_file()


def test_child_cli_adopts_a_valid_inherited_scene_lock(scene: Path):
    lock = SceneLock(scene)
    lock.acquire()
    try:
        env = {**os.environ, SCENE_LOCK_FD_ENV: str(lock.fileno())}
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
from pathlib import Path
import sys
from gausser.scene_lock import acquire_scene_lock
with acquire_scene_lock(Path(sys.argv[1])):
    print('adopted')
""",
                str(scene),
            ],
            pass_fds=(lock.fileno(),),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "adopted"
        with pytest.raises(SceneLockError, match="already owned"):
            with acquire_scene_lock(scene):
                pass
    finally:
        lock.release()


def test_inherited_scene_lock_descriptor_must_match_the_scene(scene: Path, tmp_path: Path, monkeypatch):
    _lock_path(scene).parent.mkdir(parents=True)
    _lock_path(scene).touch()
    unrelated = tmp_path / "unrelated.lock"
    with unrelated.open("w") as stream:
        monkeypatch.setenv(SCENE_LOCK_FD_ENV, str(stream.fileno()))
        with pytest.raises(SceneLockError, match="does not belong"):
            with acquire_scene_lock(scene):
                pass


def test_matching_inherited_descriptor_must_actually_own_the_lock(scene: Path, monkeypatch):
    child = _lock_in_child(scene)
    try:
        with _lock_path(scene).open("a+") as stream:
            monkeypatch.setenv(SCENE_LOCK_FD_ENV, str(stream.fileno()))
            with pytest.raises(SceneLockError, match="does not own"):
                with acquire_scene_lock(scene):
                    pass
    finally:
        _release_child(child)


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


def _release_child(child: subprocess.Popen[str]) -> None:
    if child.stdin:
        child.stdin.close()
    child.wait(timeout=5)
