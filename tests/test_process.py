from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
import time

import pytest

from gausser.process import run_command


def test_runner_merges_output_and_keeps_a_bounded_tail(tmp_path):
    lines: list[str] = []
    result = run_command(
        [
            sys.executable,
            "-u",
            "-c",
            "import sys; print('standard'); print('diagnostic', file=sys.stderr); print('final')",
        ],
        cwd=tmp_path,
        on_line=lines.append,
        tail_lines=2,
    )

    assert result.succeeded
    assert lines == ["standard", "diagnostic", "final"]
    assert result.output_tail == ("diagnostic", "final")


def test_runner_interrupts_a_cancelled_process_group(tmp_path):
    cancelled = threading.Event()
    timer = threading.Timer(0.1, cancelled.set)
    timer.start()
    started = time.monotonic()
    try:
        result = run_command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            cancel_event=cancelled,
        )
    finally:
        timer.cancel()

    assert result.cancelled
    assert result.returncode != 0
    assert time.monotonic() - started < 3


@pytest.mark.skipif(os.name == "nt", reason="process-group descendant test is POSIX-specific")
def test_runner_terminates_descendants_after_group_leader_exits(tmp_path: Path):
    cancelled = threading.Event()
    pid_path = tmp_path / "child.pid"
    timer = threading.Timer(0.2, cancelled.set)
    timer.start()

    try:
        result = run_command(
        [
            sys.executable,
            "-u",
            "-c",
            "import signal, subprocess, sys, time; "
            "child = subprocess.Popen([sys.executable, '-c', "
            "'import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(30)']); "
            f"open({str(pid_path)!r}, 'w').write(str(child.pid)); time.sleep(30)",
        ],
        cwd=tmp_path,
        cancel_event=cancelled,
    )
    finally:
        timer.cancel()

    assert result.cancelled
    assert pid_path.exists()
    child_pid = int(pid_path.read_text())
    alive = True
    try:
        os.kill(child_pid, 0)
    except ProcessLookupError:
        alive = False
    finally:
        if alive:
            os.kill(child_pid, 9)
    assert not alive
