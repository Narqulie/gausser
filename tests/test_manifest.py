from __future__ import annotations

import json
from pathlib import Path

from gausser.manifest import record_run_finished, record_run_started, transition_stage
from gausser.scene import ScenePaths


def test_run_records_preserve_resolved_settings_and_stage_data(tmp_path: Path):
    paths = ScenePaths.from_root(tmp_path)
    transition_stage(paths, "colmap", {"model": "sparse/0"}, invalidated_stages=())
    settings = {"specification_version": 1, "preset": "video-sequence", "values": {"matcher": "sequential"}}

    record_run_started(paths, "run-1", "colmap", settings)
    record_run_finished(paths, "run-1", "succeeded", {"returncode": 0})

    manifest = json.loads(paths.manifest.read_text())
    assert manifest["stages"]["colmap"]["model"] == "sparse/0"
    assert manifest["runs"]["run-1"] == {
        "stage": "colmap",
        "started_at": manifest["runs"]["run-1"]["started_at"],
        "finished_at": manifest["runs"]["run-1"]["finished_at"],
        "settings": settings,
        "state": "succeeded",
        "result": {"returncode": 0},
    }
