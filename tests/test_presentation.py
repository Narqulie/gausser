from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from gausser.scene import ScenePaths


def _artifact(scene: Path, *, content: bytes = b"ply\n") -> Path:
    export = scene / "runs" / "brush" / "run-a" / "exports" / "scene.ply"
    export.parent.mkdir(parents=True)
    export.write_bytes(content)
    return export


def _point(x: float, y: float, z: float) -> dict[str, float]:
    return {"x": x, "y": y, "z": z}


def test_new_presentation_is_a_non_mutating_artifact_bound_document(scene: Path):
    from gausser.presentation import load_presentation

    artifact = _artifact(scene)
    document = load_presentation(
        ScenePaths.from_root(scene),
        "runs/brush/run-a/exports/scene.ply",
    )

    assert document == {
        "version": 1,
        "artifact": {
            "relative_path": "runs/brush/run-a/exports/scene.ply",
            "sha256": hashlib.sha256(b"ply\n").hexdigest(),
        },
        "revision": None,
        "frame": {"origin": _point(0.0, 0.0, 0.0)},
        "default_pivot": None,
        "focus_points": [],
        "views": [],
        "home_view_id": None,
    }
    assert artifact.read_bytes() == b"ply\n"
    assert not (scene / "presentations").exists()


def test_presentation_save_is_atomic_revisioned_and_separate_from_immutable_run(scene: Path):
    from gausser.presentation import load_presentation, save_presentation

    paths = ScenePaths.from_root(scene)
    artifact = _artifact(scene)
    document = load_presentation(paths, "runs/brush/run-a/exports/scene.ply")
    document["frame"] = {"origin": _point(1.0, 2.0, 3.0)}
    document["default_pivot"] = _point(4.0, 5.0, 6.0)
    document["focus_points"] = [{
        "id": "signature",
        "name": "Artist signature",
        "position": _point(0.4, 0.3, -0.2),
        "framing_radius": 0.15,
    }]
    document["views"] = [{
        "id": "hero",
        "name": "Hero view",
        "position": _point(3.8, 1.9, 5.1),
        "target": _point(0.0, 1.2, 0.0),
        "up": _point(0.0, 1.0, 0.0),
        "vertical_fov_degrees": 50.0,
    }]
    document["home_view_id"] = "hero"

    saved = save_presentation(paths, document)

    assert isinstance(saved["revision"], str) and len(saved["revision"]) == 64
    assert saved["artifact"] == document["artifact"]
    assert load_presentation(paths, document["artifact"]["relative_path"]) == saved
    assert artifact.read_bytes() == b"ply\n"
    files = list((scene / "presentations").iterdir())
    assert len(files) == 1 and files[0].suffix == ".json"
    assert not list((scene / "presentations").glob(".*.tmp"))
    assert json.loads(files[0].read_text(encoding="utf-8")) == saved


def test_presentation_reuses_an_unchanged_large_artifact_fingerprint(
    scene: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import gausser.presentation as presentation

    paths = ScenePaths.from_root(scene)
    artifact = _artifact(scene, content=b"large artifact stand-in")
    fingerprint_calls = 0

    def count_fingerprint(path: Path) -> str:
        nonlocal fingerprint_calls
        fingerprint_calls += 1
        return hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(presentation, "fingerprint_file", count_fingerprint)
    document = presentation.load_presentation(paths, "runs/brush/run-a/exports/scene.ply")
    saved = presentation.save_presentation(paths, document)
    assert presentation.load_presentation(paths, saved["artifact"]["relative_path"]) == saved
    assert fingerprint_calls == 1
    assert artifact.read_bytes() == b"large artifact stand-in"


def test_presentation_rejects_a_stale_revision_without_overwriting(scene: Path):
    from gausser.presentation import PresentationChangedError, load_presentation, save_presentation

    paths = ScenePaths.from_root(scene)
    _artifact(scene)
    original = load_presentation(paths, "runs/brush/run-a/exports/scene.ply")
    first = copy.deepcopy(original)
    first["default_pivot"] = _point(1.0, 0.0, 0.0)
    saved = save_presentation(paths, first)

    stale = copy.deepcopy(original)
    stale["default_pivot"] = _point(2.0, 0.0, 0.0)
    with pytest.raises(PresentationChangedError, match="changed"):
        save_presentation(paths, stale)

    assert load_presentation(paths, original["artifact"]["relative_path"]) == saved


def test_presentation_rejects_an_artifact_whose_bytes_changed(scene: Path):
    from gausser.presentation import PresentationArtifactChangedError, load_presentation, save_presentation

    paths = ScenePaths.from_root(scene)
    artifact = _artifact(scene)
    saved = save_presentation(
        paths,
        load_presentation(paths, "runs/brush/run-a/exports/scene.ply"),
    )
    artifact.write_bytes(b"different")

    with pytest.raises(PresentationArtifactChangedError, match="artifact changed"):
        load_presentation(paths, saved["artifact"]["relative_path"])


def test_presentation_validates_geometry_ids_and_home_view(scene: Path):
    from gausser.presentation import PresentationValidationError, load_presentation, save_presentation

    paths = ScenePaths.from_root(scene)
    _artifact(scene)
    base = load_presentation(paths, "runs/brush/run-a/exports/scene.ply")

    invalid_documents: list[dict] = []
    non_finite = copy.deepcopy(base)
    non_finite["frame"] = {"origin": _point(math.nan, 0.0, 0.0)}
    invalid_documents.append(non_finite)

    duplicate_points = copy.deepcopy(base)
    duplicate_points["focus_points"] = [
        {"id": "same", "name": "One", "position": _point(0.0, 0.0, 0.0), "framing_radius": 1.0},
        {"id": "same", "name": "Two", "position": _point(1.0, 0.0, 0.0), "framing_radius": 1.0},
    ]
    invalid_documents.append(duplicate_points)

    missing_home = copy.deepcopy(base)
    missing_home["home_view_id"] = "missing"
    invalid_documents.append(missing_home)

    degenerate_camera = copy.deepcopy(base)
    degenerate_camera["views"] = [{
        "id": "view",
        "name": "View",
        "position": _point(1.0, 1.0, 1.0),
        "target": _point(1.0, 1.0, 1.0),
        "up": _point(0.0, 1.0, 0.0),
        "vertical_fov_degrees": 50.0,
    }]
    invalid_documents.append(degenerate_camera)

    parallel_up = copy.deepcopy(base)
    parallel_up["views"] = [{
        "id": "view",
        "name": "View",
        "position": _point(0.0, 0.0, 2.0),
        "target": _point(0.0, 0.0, 0.0),
        "up": _point(0.0, 0.0, 1.0),
        "vertical_fov_degrees": 50.0,
    }]
    invalid_documents.append(parallel_up)

    for document in invalid_documents:
        with pytest.raises(PresentationValidationError):
            save_presentation(paths, document)


def test_engine_gets_and_updates_presentation_under_the_scene_contract(scene: Path):
    from gausser.engine import GausserEngine

    _artifact(scene)
    engine = GausserEngine()
    emitted: list[dict] = []
    engine.dispatch(
        {
            "id": "get-presentation",
            "method": "presentation",
            "params": {
                "scene": str(scene),
                "artifact": "runs/brush/run-a/exports/scene.ply",
            },
        },
        emitted.append,
    )
    document = emitted.pop()["result"]["presentation"]
    document["default_pivot"] = _point(1.0, 2.0, 3.0)

    engine.dispatch(
        {
            "id": "save-presentation",
            "method": "update_presentation",
            "params": {"scene": str(scene), "presentation": document},
        },
        emitted.append,
    )

    assert emitted[0]["id"] == "save-presentation"
    assert emitted[0]["result"]["presentation"]["default_pivot"] == _point(1.0, 2.0, 3.0)
    assert emitted[0]["result"]["presentation"]["revision"]


def test_engine_reports_stale_presentation_revision_as_a_domain_error(scene: Path):
    from gausser.engine import GausserEngine

    _artifact(scene)
    engine = GausserEngine()
    emitted: list[dict] = []
    request = {
        "id": "get",
        "method": "presentation",
        "params": {"scene": str(scene), "artifact": "runs/brush/run-a/exports/scene.ply"},
    }
    engine.dispatch(request, emitted.append)
    stale = emitted.pop()["result"]["presentation"]
    engine.dispatch(
        {
            "id": "first",
            "method": "update_presentation",
            "params": {"scene": str(scene), "presentation": stale},
        },
        emitted.append,
    )
    emitted.pop()
    stale["frame"] = {"origin": _point(9.0, 0.0, 0.0)}

    engine.dispatch(
        {
            "id": "stale",
            "method": "update_presentation",
            "params": {"scene": str(scene), "presentation": stale},
        },
        emitted.append,
    )

    assert emitted == [{
        "id": "stale",
        "error": {
            "code": "presentation_changed",
            "message": "presentation changed; reload it before editing",
        },
    }]
