from __future__ import annotations

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
