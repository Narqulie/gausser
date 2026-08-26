import subprocess
import sys
from pathlib import Path

import pytest

from gausser.scene_lock import SceneLockError, acquire_scene_lock


def test_engine_rejects_a_stage_when_another_process_owns_scene(scene: Path):
    from gausser.engine import GausserEngine

    child = _lock_in_child(scene)
    try:
        events: list[dict] = []
        GausserEngine().dispatch(
            {"id": "start", "method": "start", "params": {"scene": str(scene), "stage": "sammer"}},
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
