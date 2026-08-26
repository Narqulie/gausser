from __future__ import annotations

import pytest

from gausser.stage_options import StageOptionError, catalog, resolve_options


def test_sharpframes_catalog_exposes_complete_upstream_controls():
    definition = catalog()["sharpframes"]

    assert definition["version"] == 1
    assert definition["preset"] == "upstream-defaults"
    assert [field["key"] for field in definition["fields"]] == [
        "source", "fps", "num_frames", "min_buffer", "format", "selection_method",
        "batch_size", "batch_buffer", "outlier_window_size", "outlier_sensitivity", "width",
    ]
    assert definition["fields"][1]["default"] == 10
    assert definition["fields"][1]["minimum"] == 1
    assert [group["id"] for group in definition["groups"]] == ["source", "selection", "advanced"]
    assert definition["fields"][6]["group"] == "advanced"
    assert definition["fields"][6]["advanced"] is True


def test_sharpframes_options_resolve_to_reproducible_command_arguments():
    resolved = resolve_options(
        "sharpframes",
        {
            "source": "/tmp/clips",
            "fps": 2,
            "num_frames": 240,
            "selection_method": "batched",
            "batch_size": 8,
            "width": 1920,
        },
    )

    assert resolved.preset == "custom"
    assert resolved.values == {
        "source": "/tmp/clips", "fps": 2, "num_frames": 240, "min_buffer": 3,
        "format": "jpg", "selection_method": "batched", "batch_size": 8,
        "batch_buffer": 2, "outlier_window_size": 15, "outlier_sensitivity": 60, "width": 1920,
    }
    assert resolved.arguments == (
        "/tmp/clips", "--fps", "2", "--num-frames", "240", "--min-buffer", "3",
        "--format", "jpg", "--selection-method", "batched", "--batch-size", "8",
        "--batch-buffer", "2", "--outlier-window-size", "15", "--outlier-sensitivity", "60",
        "--width", "1920",
    )


@pytest.mark.parametrize(
    "options, message",
    [
        ({"source": "/tmp/clips", "fps": 0}, "fps must be at least 1"),
        ({"source": "/tmp/clips", "format": "webp"}, "format must be one of: jpg, png"),
        ({"source": "/tmp/clips", "unexpected": True}, "unsupported sharpframes options: unexpected"),
    ],
)
def test_sharpframes_options_fail_before_launch(options, message):
    with pytest.raises(StageOptionError, match=message):
        resolve_options("sharpframes", options)
