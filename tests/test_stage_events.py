from __future__ import annotations

from gausser.stage_events import StageTelemetry


def test_colmap_telemetry_reports_known_phase_progress():
    telemetry = StageTelemetry("colmap", total_images=425)

    event = telemetry.from_line("Processed file [12/425]")

    assert event is not None
    assert event.kind == "progress"
    assert event.data == {
        "phase": "reconstruction",
        "current": 12,
        "total": 425,
        "unit": "images",
        "message": "Processed file [12/425]",
    }


def test_brush_telemetry_reports_observed_metrics_without_inventing_progress():
    telemetry = StageTelemetry("brush", total_iterations=30_000)

    refined = telemetry.from_line("Refine iter 1200, 38291 splats")
    evaluated = telemetry.from_line("Eval iter 1200: PSNR 27.80, ssim 0.910")

    assert refined is not None
    assert refined.kind == "metric"
    assert refined.data == {
        "phase": "training",
        "step": 1200,
        "values": {"splat_count": 38291},
    }
    assert evaluated is not None
    assert evaluated.kind == "metric"
    assert evaluated.data == {
        "phase": "training",
        "step": 1200,
        "values": {"psnr": 27.8, "ssim": 0.91},
    }


def test_sammer_telemetry_reports_masking_progress():
    telemetry = StageTelemetry("sammer")

    event = telemetry.from_line("Masking 14/425")

    assert event is not None
    assert event.data == {"phase": "masking", "current": 14, "total": 425, "unit": "images", "message": "Masking 14/425"}


def test_msplat_telemetry_reports_training_metrics():
    telemetry = StageTelemetry("msplat")

    event = telemetry.from_line("Msplat 250/30000: 2,012,333 splats")

    assert event is not None
    assert event.data == {"phase": "training", "step": 250, "total": 30000, "unit": "iterations", "values": {"splat_count": 2012333}, "message": "Msplat 250/30000: 2,012,333 splats"}
