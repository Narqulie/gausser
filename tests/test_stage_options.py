from __future__ import annotations

import pytest

from gausser.stage_options import StageOptionError, catalog, resolve_options


def test_sammer_options_resolve_standard_and_coverage_presets():
    standard = resolve_options("sammer", {"prompt": "fountain"})
    assert standard.preset == "standard-mask"
    assert standard.arguments == ("--prompt", "fountain", "--score-threshold", "0.5", "--device", "auto", "--dilation", "0", "--feather", "0")

    coverage = resolve_options("sammer", {"prompt": "fountain", "score_threshold": 0.3, "dilation": 8, "feather": 12})
    assert coverage.preset == "coverage-focus"


def test_sammer_options_reject_invalid_prompt_and_threshold():
    with pytest.raises(StageOptionError, match="prompt must be a non-empty string"):
        resolve_options("sammer", {"prompt": "  "})
    with pytest.raises(StageOptionError, match="score_threshold must be between 0 and 1"):
        resolve_options("sammer", {"prompt": "fountain", "score_threshold": 1.1})


def test_colmap_options_resolve_capture_aware_presets_and_safe_artifact_policy():
    video = resolve_options("colmap", {})
    assert video.preset == "video-sequence"
    assert video.arguments == ("--matcher", "sequential", "--existing", "error")

    photos = resolve_options("colmap", {"matcher": "exhaustive"})
    assert photos.preset == "photo-set"
    assert photos.arguments == ("--matcher", "exhaustive", "--existing", "error")

    vocab = resolve_options("colmap", {"matcher": "vocab_tree", "vocab_tree": "/models/tree.bin", "use_gpu": False})
    assert vocab.arguments == ("--matcher", "vocab_tree", "--existing", "error", "--no-gpu", "--vocab-tree", "/models/tree.bin")


def test_colmap_options_reject_incomplete_vocabulary_tree_request():
    with pytest.raises(StageOptionError, match="vocab_tree is required"):
        resolve_options("colmap", {"matcher": "vocab_tree"})


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
