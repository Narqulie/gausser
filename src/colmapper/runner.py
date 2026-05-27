"""Subprocess runner with real-time COLMAP output parsing."""

import re
import subprocess
import threading
from dataclasses import dataclass
from typing import Callable, Optional


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
    on_progress: Optional[ProgressCallback] = None,
    on_line: Optional[LineCallback] = None,
    total_images: int = 0,
) -> int:
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, errors="replace",
    )

    def _read() -> None:
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if on_line:
                on_line(line)
            if on_progress:
                _parse_and_emit(line, on_progress, total_images)

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    process.wait()
    t.join(timeout=5)
    return process.returncode


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
