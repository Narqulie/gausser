"""Subprocess runner with real-time COLMAP output parsing."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from gausser.process import CommandResult, run_command


@dataclass
class StepProgress:
    current: int
    total: int
    message: str


ProgressCallback = Callable[[StepProgress], None]
LineCallback = Callable[[str], None]

_PROGRESS_PATTERNS = [
    re.compile(r"Processed file \[(\d+)/(\d+)\]"),
    re.compile(r"\[(\d+)/(\d+)\]"),
    re.compile(r"Matching block \[(\d+)/(\d+),"),
    re.compile(r"Registering image #\d+ \((\d+)\)"),
    re.compile(r"(\d+)\s*/\s*(\d+)"),
]

_REGISTER_PATTERN = re.compile(r"Registering image #\d+ \((\d+)\)")


def run_colmap(
    cmd: list[str],
    *,
    cwd: Path,
    on_progress: Optional[ProgressCallback] = None,
    on_line: Optional[LineCallback] = None,
    total_images: int = 0,
) -> CommandResult:
    def handle_line(line: str) -> None:
        if on_line:
            on_line(line)
        if on_progress:
            _parse_and_emit(line, on_progress, total_images)

    return run_command(cmd, cwd=cwd, on_line=handle_line)


def _parse_and_emit(line: str, callback: ProgressCallback, total_images: int) -> None:
    m = _REGISTER_PATTERN.search(line)
    if m:
        current = int(m.group(1))
        callback(StepProgress(current, max(current, total_images), line))
        return
    for pattern in _PROGRESS_PATTERNS:
        m = pattern.search(line)
        if m and len(m.groups()) >= 2:
            current, total = int(m.group(1)), int(m.group(2))
            if total > 0:
                callback(StepProgress(current, total, line))
            return
