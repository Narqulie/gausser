"""Shared, cancellable subprocess execution for scene stages."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import Callable, Mapping


LineCallback = Callable[[str], None]


@dataclass(frozen=True)
class CommandResult:
    """The durable outcome of one externally executed command."""

    command: tuple[str, ...]
    cwd: Path
    returncode: int
    duration_seconds: float
    output_tail: tuple[str, ...]
    cancelled: bool

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and not self.cancelled


def _signal_process_tree(process: subprocess.Popen[str], sig: int) -> None:
    """Signal all descendants even after the group leader has exited."""
    if os.name == "nt":
        if sig == signal.SIGINT and process.poll() is None:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            return
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.killpg(process.pid, sig)
    except (ProcessLookupError, PermissionError):
        # The group can vanish between escalation steps. macOS may report EPERM
        # for an already-reaped process group instead of ESRCH.
        return


def _interrupt_process_group(process: subprocess.Popen[str]) -> None:
    _signal_process_tree(process, signal.SIGINT)


def run_command(
    command: list[str],
    *,
    cwd: Path,
    on_line: LineCallback | None = None,
    env: Mapping[str, str] | None = None,
    tail_lines: int = 200,
    cancel_event: threading.Event | None = None,
) -> CommandResult:
    """Run ``command`` in ``cwd``, streaming merged output and retaining its tail.

    A cancellation request sends SIGINT to the complete child process group, so a
    stage can stop its subprocess tree without the caller guessing process IDs.
    """
    if not command:
        raise ValueError("command must not be empty")
    if tail_lines < 1:
        raise ValueError("tail_lines must be positive")

    workdir = Path(cwd).expanduser().resolve()
    if not workdir.is_dir():
        raise ValueError(f"command working directory does not exist: {workdir}")

    if cancel_event and cancel_event.is_set():
        return CommandResult(tuple(command), workdir, 130, 0.0, (), True)

    child_env = os.environ.copy()
    if env:
        child_env.update(env)

    started = time.monotonic()
    popen_kwargs: dict[str, object] = {"start_new_session": True}
    if os.name == "nt":
        popen_kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}

    process = subprocess.Popen(
        command,
        cwd=workdir,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        errors="replace",
        env=child_env,
        **popen_kwargs,
    )
    assert process.stdout is not None

    tail: deque[str] = deque(maxlen=tail_lines)

    def read_output() -> None:
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            tail.append(line)
            if on_line:
                on_line(line)

    reader = threading.Thread(target=read_output, name="gausser-process-output", daemon=True)
    reader.start()

    cancelled = False
    try:
        while process.poll() is None:
            if cancel_event and cancel_event.is_set():
                cancelled = True
                _interrupt_process_group(process)
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        cancelled = True
        _interrupt_process_group(process)

    if cancelled:
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        _signal_process_tree(process, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        _signal_process_tree(process, signal.SIGKILL)
        process.wait()
    else:
        process.wait()

    reader.join(timeout=2)
    return CommandResult(
        command=tuple(command),
        cwd=workdir,
        returncode=process.returncode,
        duration_seconds=time.monotonic() - started,
        output_tail=tuple(tail),
        cancelled=cancelled,
    )
