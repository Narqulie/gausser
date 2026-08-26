"""Conservative structured telemetry derived from documented upstream output."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal

from brusher.runner import TrainProgress, parse_progress as parse_brush_progress
from colmapper.runner import parse_progress as parse_colmap_progress

StageEventKind = Literal["progress", "metric"]
_SAMMER_PROGRESS = re.compile(r"\bMasking\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_MSPLAT_PROGRESS = re.compile(r"\bMsplat\s+(\d+)\s*/\s*(\d+)\s*:\s*([\d,]+)\s+splats", re.IGNORECASE)


@dataclass(frozen=True)
class StageEvent:
    """A stage-local observation; the engine adds run identity and elapsed time."""

    kind: StageEventKind
    data: dict[str, Any]



class StageTelemetry:
    """Parse only known output patterns without inferring missing upstream state."""

    def __init__(self, stage: str, *, total_images: int = 0, total_iterations: int = 30_000) -> None:
        self.stage = stage
        self.total_images = total_images
        self._brush = TrainProgress(total=total_iterations)

    @classmethod
    def for_stage(cls, stage: str, arguments: tuple[str, ...], *, total_images: int) -> "StageTelemetry":
        total_iterations = 30_000
        for index, argument in enumerate(arguments[:-1]):
            if argument == "--total-train-iters":
                try:
                    total_iterations = int(arguments[index + 1])
                except ValueError:
                    pass
        return cls(stage, total_images=total_images, total_iterations=total_iterations)

    def from_line(self, line: str) -> StageEvent | None:
        if self.stage == "colmap":
            progress = parse_colmap_progress(line, self.total_images)
            if progress is None:
                return None
            return StageEvent("progress", {
                "phase": "reconstruction",
                "current": progress.current,
                "total": progress.total,
                "unit": "images",
                "message": progress.message,
            })
        if self.stage == "sammer":
            match = _SAMMER_PROGRESS.search(line)
            if match is None:
                return None
            return StageEvent("progress", {
                "phase": "masking", "current": int(match.group(1)), "total": int(match.group(2)),
                "unit": "images", "message": line,
            })
        if self.stage == "msplat":
            match = _MSPLAT_PROGRESS.search(line)
            if match is None:
                return None
            return StageEvent("metric", {
                "phase": "training", "step": int(match.group(1)), "total": int(match.group(2)),
                "unit": "iterations", "values": {"splat_count": int(match.group(3).replace(",", ""))}, "message": line,
            })
        if self.stage == "brush":
            kind = parse_brush_progress(line, self._brush)
            if kind is None:
                return None
            state = self._brush
            values: dict[str, Any] = {}
            if kind == "refine":
                values["splat_count"] = state.splat_count
            if kind == "evaluation":
                if state.psnr is not None:
                    values["psnr"] = state.psnr
                if state.ssim is not None:
                    values["ssim"] = state.ssim
            if not values:
                return None
            return StageEvent("metric", {
                "phase": state.phase,
                "step": state.iter,
                "values": values,
            })
        return None
