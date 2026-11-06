from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from gausser import keyframes, steps
from gausser.keyframes import Ladder, Pick
from gausser.scene import Scene

WIDTH, HEIGHT = 320, 180
THRESHOLDS = (0.2, 0.1, 0.05)


def pan(speed: float, frames: int, sharps: list[float] | None = None, max_gap: int = 10_000) -> list[list[Pick]]:
    """Feed a constant horizontal pan of `speed` px per frame (frame sharpness from `sharps`)."""
    ladder = Ladder(WIDTH, HEIGHT, THRESHOLDS, max_gap)
    flow = np.zeros((HEIGHT, WIDTH, 2), dtype=np.float32)
    flow[..., 0] = speed
    for frame in range(frames):
        ladder.update(frame, None if frame == 0 else flow, sharps[frame] if sharps else 1.0)
    return ladder.finish()


def test_a_constant_pan_gives_evenly_spaced_frames_and_finer_levels_give_more():
    levels = pan(2.0, 300)  # 20% of the long edge is 64 px: about every 32 frames
    gaps = np.diff([p.frame for p in levels[0]])
    assert len(gaps) >= 5 and gaps.max() - gaps.min() <= 2 and 25 <= gaps.mean() <= 33
    assert len(levels[0]) < len(levels[1]) < len(levels[2])


def test_the_sharpest_frame_of_the_stretch_is_picked_not_the_one_that_triggered():
    sharps = [1.0] * 300
    sharps[25] = 9.0  # from 70% of the threshold (frame 23) up to the trigger (frame 32)
    assert levels_first(pan(2.0, 300, sharps)) == 25


def levels_first(levels: list[list[Pick]]) -> int:
    return levels[0][0].frame


def test_a_static_camera_gives_only_one_frame_per_max_gap():
    picks = pan(0.0, 300, max_gap=60)
    assert [len(level) for level in picks] == [5, 5, 5]


def test_choose_goes_finer_under_the_minimum_and_coarser_over_the_maximum():
    counts = [10, 20, 40, 80, 160, 320]
    assert keyframes.choose(counts, 3, 50, 500) == 3
    assert keyframes.choose(counts, 3, 100, 500) == 4
    assert keyframes.choose(counts, 3, 1000, 5000) == 5
    assert keyframes.choose(counts, 5, 10, 100) == 3


def test_every_picks_the_sharpest_frame_of_each_window():
    assert [p.frame for p in keyframes.windows([1, 5, 2, 3, 9, 4, 7], 3)] == [1, 4, 6]


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg")


def panning_video(path: Path, frames: int = 90) -> np.ndarray:
    """A lossless video of noise texture shifting 3 px per frame, so every frame number has its own content."""
    rng = np.random.default_rng(1)
    texture = cv2.GaussianBlur(rng.integers(0, 256, (120, 160 + 3 * frames), dtype=np.uint8), (5, 5), 1.5)
    source = np.stack([texture[:, 3 * n : 3 * n + 160] for n in range(frames)])
    command = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", "160x120", "-r", "30", "-i", "-", "-c:v", "ffv1", str(path)]
    subprocess.run(command, input=source.tobytes(), check=True)
    return source


@needs_ffmpeg
@pytest.mark.parametrize("span", [None, (1.0, 2.5)])
def test_extracted_files_are_exactly_the_frames_the_analysis_picked(tmp_path, span):
    video = tmp_path / "pan.mkv"
    source = panning_video(video)
    (tmp_path / "scene").mkdir()
    scene = Scene(tmp_path / "scene")
    steps.frames(scene, video, [span] if span else None, min_frames=1, max_frames=1000)

    offset = round(span[0] * 30) if span else 0
    names = sorted(path.name for path in scene.images.glob("*.jpg"))
    assert len(names) >= 4 and (scene.images / "frames.json").is_file()
    for name in names:
        number = int(name.removeprefix("frame_").removesuffix(".jpg"))
        with Image.open(scene.images / name) as image:
            pixels = np.asarray(image.convert("L"), dtype=float)
        errors = {n: abs(pixels - source[n]).mean() for n in (offset + number - 1, offset + number, offset + number + 1) if 0 <= n < len(source)}
        assert min(errors, key=errors.__getitem__) == offset + number and errors[offset + number] < 8, name


@needs_ffmpeg
def test_an_hdr_video_is_converted_to_sdr_by_the_lut(tmp_path):
    video = tmp_path / "hlg.mkv"
    # a flat gray at HLG signal 0.75 (SDR reference white) and one at 0.5, as 10-bit limited-range YUV
    frames = b"".join(np.full((64, 64), luma, dtype="<u2").tobytes() + np.full((2, 32, 32), 512 * 1, dtype="<u2").tobytes() for luma in (721, 721, 502, 502))
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", "64x64", "-r", "30",
         "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc", "-color_primaries", "bt2020", "-i", "-", "-c:v", "ffv1", str(video)],
        input=frames, check=True,
    )
    info = keyframes.probe("ffprobe", video)
    assert info.transfer == "arib-std-b67"
    out = tmp_path / "out"
    paths = keyframes.extract("ffmpeg", video, info, None, [0, 3], out, fisheyes=False)
    white, mid = (np.asarray(Image.open(p).convert("L"))[32, 32] for p in paths)
    assert white > 240 and 130 < mid < 175
