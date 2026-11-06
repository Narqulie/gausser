"""Frame selection for `gausser frames`: one analysis pass over a video, one extraction pass.

The analysis pass has ffmpeg pipe every frame as a small grey image. Each frame gets a focus score
(Laplacian variance and Tenengrad, as Sharp Frames scored them) and DIS optical flow against the
previous frame. A grid of points is tracked through the flow (after adaptive-frame-extractor): when
the 80th percentile of the points' displacement from where the grid started reaches a threshold,
or too many points have left the frame, that is a keyframe. The frame picked is the sharpest one
since the last keyframe that had moved at least `CANDIDATE` of the threshold, and the grid restarts.
Trackers with a ladder of thresholds run side by side on the same flow, so the frame count can be
chosen afterwards: `choose` takes the default level, or the next finer or coarser one when the
count is under `min_frames` or over `max_frames`.

Thresholds are fractions of the analysis image's long edge, so they do not depend on resolution.
The extraction pass decodes the video again with the same input options, so frame numbers match,
and writes only the picked frames at full resolution.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

from .scene import GausserError, run_tool

ANALYSIS_EDGE = 1280  # long edge of the images scored for focus
FLOW_EDGE = 960  # long edge of the images the flow is computed on
LEVELS = tuple(0.35 / 1.4**i for i in range(10))  # displacement thresholds, coarse to fine
SPACINGS = {"low": 5, "medium": 6, "high": 7}  # `--spacing`: default index into LEVELS
MIN_FRAMES = 150
MAX_FRAMES = 600
MAX_GAP_SECONDS = 3.0  # a keyframe at least this often, so static stretches are not lost
MIN_GAP_FRAMES = 2
MIN_IN_BOUNDS = 0.4  # keyframe when fewer of the grid's points are still inside the frame
CANDIDATE = 0.7  # a frame is a candidate from this share of the threshold
# adaptive-frame-extractor's tracker, as fractions of the flow image's long edge (at 960 px: 40, 4, 6, 25 px)
GRID_STEP, GRID_MARGIN, LOST_BORDER, MAX_STEP = 40 / 960, 4 / 960, 6 / 960, 25 / 960
HDR_TRANSFERS = {"arib-std-b67": "hlg", "smpte2084": "pq"}


@dataclass
class VideoInfo:
    width: int  # as displayed, after the rotation ffmpeg applies
    height: int
    fps: float
    frames: int  # an estimate (container header or duration × fps)
    transfer: str


@dataclass
class Pick:
    frame: int
    motion: float  # displacement score at that frame, as a share of the long edge
    sharpness: float


def probe(ffprobe: str, video: Path, span: tuple[float, float] | None = None) -> VideoInfo:
    """Size, frame rate and frame count of the video's first stream; `span` limits the count to those seconds."""
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(video)],
        capture_output=True, text=True, check=False,
    )
    streams = json.loads(result.stdout or "{}").get("streams", [])
    if result.returncode != 0 or not streams:
        raise GausserError(f"ffprobe could not read {video.name}: {result.stderr.strip() or 'no video stream'}")
    stream = streams[0]
    numerator, _, denominator = stream.get("avg_frame_rate", "0/1").partition("/")
    fps = float(numerator) / float(denominator or 1) or 30.0
    width, height = int(stream["width"]), int(stream["height"])
    rotation = next((int(data["rotation"]) for data in stream.get("side_data_list", []) if "rotation" in data), 0)
    if rotation % 180:
        width, height = height, width
    duration = float(stream.get("duration") or 0)
    frames = int(stream.get("nb_frames") or duration * fps)
    if span:
        frames = round((min(span[1], duration or span[1]) - span[0]) * fps)
    return VideoInfo(width, height, fps, frames, stream.get("color_transfer", ""))


def sharpness(grey: Any) -> float:
    """Focus score of a grey image: 0.5·log1p(Laplacian variance) + 0.5·log1p(Tenengrad), back through expm1."""
    import cv2
    import numpy as np

    smooth = cv2.GaussianBlur(grey, (5, 5), 1.0)
    laplacian = cv2.Laplacian(smooth, cv2.CV_32F)
    dx, dy = cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
    tenengrad = cv2.mean(dx * dx + dy * dy)[0]
    return float(np.expm1(0.5 * np.log1p(laplacian.var(dtype=np.float64)) + 0.5 * np.log1p(tenengrad)))


class Ladder:
    """Point trackers with one threshold each, all fed by the same flow; see the module docstring."""

    def __init__(self, width: int, height: int, thresholds: tuple[float, ...], max_gap: int) -> None:
        import numpy as np

        self.edge = max(width, height)
        step = max(1.0, self.edge * GRID_STEP)
        margin = self.edge * GRID_MARGIN
        xs, ys = np.arange(margin, width - margin, step), np.arange(margin, height - margin, step)
        self.origin = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
        self.thresholds = thresholds
        self.max_gap = max_gap
        self.limit = np.array([width - 1, height - 1], dtype=np.float32)
        count = len(thresholds)
        self.points = np.tile(self.origin, (count, 1, 1))
        self.alive = np.ones((count, len(self.origin)), dtype=bool)
        self.since = [0] * count
        self.best: list[Pick | None] = [None] * count  # sharpest candidate since the last keyframe
        self.sharpest: list[Pick | None] = [None] * count  # sharpest frame of any motion since then
        self.picks: list[list[Pick]] = [[] for _ in thresholds]

    def update(self, frame: int, flow: Any, sharp: float) -> None:
        """Move the grids by the flow from the previous frame to this one (None for the first frame)."""
        import cv2
        import numpy as np

        count = len(self.thresholds)
        if flow is None:
            progress = np.zeros(count)
            scores = np.zeros(count)
        else:
            flat = np.ascontiguousarray(self.points.reshape(1, -1, 2))
            step = cv2.remap(flow, flat, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)  # type: ignore[call-overload]
            step = step.reshape(self.points.shape)
            norm = np.hypot(step[..., 0], step[..., 1])[..., None]
            self.points += step * np.minimum(1.0, MAX_STEP * self.edge / np.maximum(norm, 1e-6))
            beyond = LOST_BORDER * self.edge
            self.alive &= ((self.points >= -beyond) & (self.points <= self.limit + beyond)).all(axis=2)
            inside = ((self.points >= 0) & (self.points <= self.limit)).all(axis=2) & self.alive
            displacement = np.hypot(*np.moveaxis(self.points - self.origin, 2, 0))
            scores = np.array([np.percentile(row[mask], 80) if mask.any() else 0.0 for row, mask in zip(displacement, inside)])
            scores /= self.edge
            ratio = inside.sum(axis=1) / len(self.origin)
            progress = np.maximum(scores / np.array(self.thresholds), (1 - ratio) / (1 - MIN_IN_BOUNDS))
        for k in range(count):
            self.since[k] += 1
            pick = Pick(frame, float(scores[k]), sharp)
            if self.sharpest[k] is None or sharp > self.sharpest[k].sharpness:  # type: ignore[union-attr]
                self.sharpest[k] = pick
            if progress[k] >= CANDIDATE and (self.best[k] is None or sharp > self.best[k].sharpness):  # type: ignore[union-attr]
                self.best[k] = pick
            if self.since[k] >= self.max_gap or (self.since[k] >= MIN_GAP_FRAMES and progress[k] >= 1):
                self._emit(k)

    def _emit(self, k: int) -> None:
        self.picks[k].append(self.best[k] or self.sharpest[k])  # type: ignore[arg-type]
        self.points[k] = self.origin
        self.alive[k] = True
        self.since[k] = 0
        self.best[k] = self.sharpest[k] = None

    def finish(self) -> list[list[Pick]]:
        """The picks of every level; the last stretch of the video adds its sharpest candidate."""
        for k in range(len(self.thresholds)):
            if self.best[k]:
                self._emit(k)
        return self.picks


def windows(sharps: list[float], every: int) -> list[Pick]:
    """The sharpest frame of each run of `every` frames."""
    return [
        Pick(start + max(range(len(run)), key=run.__getitem__), 0.0, max(run))
        for start in range(0, len(sharps), every)
        for run in [sharps[start : start + every]]
    ]


def choose(counts: list[int], default: int, min_frames: int, max_frames: int) -> int:
    """The level to use: `default`, or finer while under `min_frames`, or coarser while over `max_frames`."""
    level = default
    if counts[level] < min_frames:
        while counts[level] < min_frames and level < len(counts) - 1:
            level += 1
    else:
        while counts[level] > max_frames and level > 0:
            level -= 1
    return level


def input_options(span: tuple[float, float] | None, video: Path) -> list[str]:
    """ffmpeg's input side; the analysis and the extraction must use exactly these."""
    seek = ["-ss", f"{span[0]:g}", "-to", f"{span[1]:g}"] if span else []
    return [*seek, "-i", str(video)]


def analyse(
    ffmpeg: str, video: Path, info: VideoInfo, span: tuple[float, float] | None, every: int | None,
) -> list[list[Pick]]:
    """The picks of each level of LEVELS (one level when `every` is set: the sharpest of each `every` frames)."""
    import cv2
    import numpy as np

    scale = min(1.0, ANALYSIS_EDGE / max(info.width, info.height))
    width, height = max(2, round(info.width * scale)), max(2, round(info.height * scale))
    flow_scale = min(1.0, FLOW_EDGE / max(width, height))
    flow_size = (max(2, round(width * flow_scale)), max(2, round(height * flow_scale)))
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", *input_options(span, video), "-map", "0:v:0",
        "-vf", f"scale={width}:{height},format=gray", "-fps_mode", "passthrough", "-f", "rawvideo", "-",
    ]
    print("$ " + " ".join(command), flush=True)
    ladder = None if every else Ladder(*flow_size, LEVELS, max(MIN_GAP_FRAMES, round(MAX_GAP_SECONDS * info.fps)))
    flow_engine = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    flow_engine.setPatchSize(8)
    flow_engine.setPatchStride(8)
    flow_engine.setGradientDescentIterations(12)
    flow_engine.setVariationalRefinementIterations(0)
    sharps: list[float] = []
    previous = None
    shown = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = cast(BinaryIO, process.stdout), cast(BinaryIO, process.stderr)
    size = width * height
    while len(data := stdout.read(size)) == size:
        grey = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
        sharp = sharpness(grey)
        if ladder:
            small = cv2.resize(grey, flow_size, interpolation=cv2.INTER_AREA) if flow_scale < 1 else grey
            flow = None if previous is None else flow_engine.calc(previous, small, None)  # type: ignore[call-overload]
            ladder.update(len(sharps), flow, sharp)
            previous = small
        sharps.append(sharp)
        if time.monotonic() - shown > 5:
            shown = time.monotonic()
            print(f"analysed {len(sharps)}/{max(info.frames, len(sharps))} frames", flush=True)
    stdout.close()
    error = stderr.read().decode(errors="replace").strip()
    if process.wait() != 0 or not sharps:
        raise GausserError(f"ffmpeg could not decode {video.name}: {error[-500:] or 'no frames'}")
    print(f"analysed {len(sharps)}/{len(sharps)} frames", flush=True)
    return [windows(sharps, every)] if every else ladder.finish()  # type: ignore[union-attr]


def hdr_lut(transfer: str) -> str:
    """A .cube 3D LUT from HDR signal (HLG or PQ, BT.2020 primaries) to SDR sRGB: reference white scaled to 1, highlights clipped."""
    import numpy as np

    size = 65
    axis = np.linspace(0, 1, size)
    signal = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)[:, ::-1]  # red varies fastest

    def hlg(e: Any) -> Any:  # scene light
        return np.where(e <= 0.5, e * e / 3, (np.exp((e - 0.55991073) / 0.17883277) + 0.28466892) / 12)

    if transfer == "hlg":
        linear = hlg(signal) / hlg(np.array(0.75))  # HLG 75% is SDR reference white (BT.2408)
    else:
        p = signal ** (1 / 78.84375)
        linear = (np.maximum(p - 0.8359375, 0) / (18.8515625 - 18.6875 * p)) ** (1 / 0.1593017578125) * 10000 / 203
    linear = np.einsum("ij,nj->ni", [[1.6605, -0.5876, -0.0728], [-0.1246, 1.1329, -0.0083], [-0.0182, -0.1006, 1.1187]], linear)
    linear = np.clip(linear, 0, 1)
    srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    return f"LUT_3D_SIZE {size}\n" + "\n".join(" ".join(f"{v:.6f}" for v in row) for row in srgb) + "\n"


def _any_of(frames: list[int]) -> str:
    """An ffmpeg expression true on these frame numbers; nested in halves, since a long `+` chain overflows ffmpeg's parser."""
    if len(frames) == 1:
        return f"eq(n,{frames[0]})"
    half = len(frames) // 2
    return f"({_any_of(frames[:half])}+{_any_of(frames[half:])})"


def extract(
    ffmpeg: str, video: Path, info: VideoInfo, span: tuple[float, float] | None, frames: list[int],
    out: Path, fisheyes: bool,
) -> list[Path]:
    """Write the given frame numbers of the video as full-size JPEGs into `out`, in order; returns their paths.

    HDR video becomes SDR through a 3D LUT. A QooCam rig video (`fisheyes`) gets its two lens streams
    un-mirrored and side by side; both lenses use the same frame numbers.
    """
    out.mkdir()
    pick = f"select='{_any_of(frames)}'"
    if fisheyes:  # QooCam lens streams are stored mirrored (qoocam.py): vflip un-mirrors each one
        graph = f"[0:0]{pick},vflip[a];[0:1]{pick},vflip[b];[a][b]hstack"
    else:
        graph = f"[0:v:0]{pick}"
    if hdr := HDR_TRANSFERS.get(info.transfer):
        (out / "hdr.cube").write_text(hdr_lut(hdr))
        graph += ",lut3d=file=hdr.cube:interp=tetrahedral"
    (out / "graph.txt").write_text(graph + "[out]")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *input_options(span, video), "-/filter_complex", "graph.txt",
        "-map", "[out]", "-fps_mode", "passthrough", "-q:v", "2", "-start_number", "0", "%07d.jpg",
    ]
    run_tool(command, cwd=out)
    paths = sorted(out.glob("*.jpg"))
    if len(paths) != len(frames):
        raise GausserError(f"ffmpeg wrote {len(paths)} frames of {video.name}, expected {len(frames)}")
    return paths

