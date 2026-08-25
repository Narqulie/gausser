"""Subprocess runner that streams and parses Brush log output."""

from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional

from gausser.process import CommandResult, run_command


@dataclass
class TrainProgress:
    iter: int = 0
    total: int = 30000
    splat_count: int = 0
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    phase: str = "starting"


ProgressCallback = Callable[[TrainProgress], None]
LineCallback = Callable[[str], None]

_DATASET_PAT = re.compile(r"Loaded dataset with (\d+) training,\s*(\d+) eval views")
_REFINE_PAT = re.compile(r"Refine iter (\d+),\s*(\d+) splats")
_EVAL_PAT = re.compile(r"Eval iter (\d+): PSNR ([0-9.]+(?:e[+-]?\d+)?), ssim ([0-9.]+(?:e[+-]?\d+)?)")
_DONE_PAT = re.compile(r"Done training!")
_LOADING_PAT = re.compile(r"Loading dataset|Loading initial splats|Loading colmap|Loading nerfstudio")
_LOADED_PAT = re.compile(r"Completed loading")


def run_brush(
    cmd: list[str],
    *,
    cwd: Path,
    on_progress: Optional[ProgressCallback] = None,
    on_line: Optional[LineCallback] = None,
    total_iters: int = 30000,
) -> CommandResult:
    state = TrainProgress(total=total_iters)

    def handle_line(line: str) -> None:
        if on_line:
            on_line(line)
        if on_progress:
            _parse_and_emit(line, state, on_progress)

    return run_command(
        cmd,
        cwd=cwd,
        on_line=handle_line,
        env={"RUST_LOG": "info", "NO_COLOR": "1"},
    )


def _parse_and_emit(line: str, state: TrainProgress, callback: ProgressCallback) -> None:
    msg = re.sub(r"^\[.*?\]\s*", "", line)

    if _LOADING_PAT.search(msg):
        state.phase = "loading"
        callback(state)
        return

    m = _DATASET_PAT.search(msg)
    if m:
        state.phase = "loading"
        callback(state)
        return

    if _LOADED_PAT.search(msg):
        state.phase = "training"
        callback(state)
        return

    m = _REFINE_PAT.search(msg)
    if m:
        state.iter = int(m.group(1))
        state.splat_count = int(m.group(2))
        state.phase = "training"
        callback(state)
        return

    m = _EVAL_PAT.search(msg)
    if m:
        state.iter = int(m.group(1))
        state.psnr = float(m.group(2))
        state.ssim = float(m.group(3))
        state.phase = "training"
        callback(state)
        return

    if _DONE_PAT.search(msg):
        state.iter = state.total
        state.phase = "done"
        callback(state)
        return
