"""Shared, cancellable subprocess execution for scene stages."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import signal
import shutil
import subprocess
import threading
import time
from typing import Callable, Mapping


LineCallback = Callable[[str], None]


def command_environment() -> dict[str, str]:
    """Return the user-facing command environment inherited by child stages.

    Pyenv inserts the selected interpreter's ``bin`` directory ahead of the
    shell's PATH while dispatching Python. Remove only that synthetic prefix so
    child stage names resolve as they do in the user's shell. This preserves
    higher-priority wrapper commands such as the isolated macOS SAM3 runtime.
    """
    environment = os.environ.copy()
    pyenv_root = environment.get("PYENV_ROOT")
    pyenv_version = environment.get("PYENV_VERSION", "").split(":", 1)[0]
    path_entries = environment.get("PATH", "").split(os.pathsep)
    if pyenv_root and pyenv_version and path_entries:
        injected = Path(pyenv_root) / "versions" / pyenv_version / "bin"
        if Path(path_entries[0]).expanduser() == injected.expanduser():
            environment["PATH"] = os.pathsep.join(path_entries[1:])
    return environment


def resolve_command(command: str) -> str | None:
    """Resolve a command using the same PATH that child stages inherit."""
    return shutil.which(command, path=command_environment().get("PATH"))


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
            process.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM))
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
    inherited_fds: tuple[int, ...] = (),
    interrupt_grace_seconds: float = 2.0,
) -> CommandResult:
    """Run ``command`` in ``cwd``, streaming merged output and retaining its tail.

    A cancellation request sends SIGINT to the complete child process group, so a
    stage can stop its subprocess tree without the caller guessing process IDs.
    """
    if not command:
        raise ValueError("command must not be empty")
    if tail_lines < 1:
        raise ValueError("tail_lines must be positive")
    if interrupt_grace_seconds < 0:
        raise ValueError("interrupt_grace_seconds cannot be negative")
    if any(not isinstance(fd, int) or fd < 0 for fd in inherited_fds):
        raise ValueError("inherited_fds must contain non-negative file descriptors")
    if inherited_fds and os.name == "nt":
        raise ValueError("inherited_fds are unsupported on Windows")

    workdir = Path(cwd).expanduser().resolve()
    if not workdir.is_dir():
        raise ValueError(f"command working directory does not exist: {workdir}")

    if cancel_event and cancel_event.is_set():
        return CommandResult(tuple(command), workdir, 130, 0.0, (), True)

    child_env = command_environment()
    if env:
        child_env.update(env)

    started = time.monotonic()
    if os.name == "nt":
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
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
    else:
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
            start_new_session=True,
            pass_fds=inherited_fds,
        )
    stdout = process.stdout
    assert stdout is not None

    tail: deque[str] = deque(maxlen=tail_lines)

    def read_output() -> None:
        for raw_line in stdout:
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
            process.wait(timeout=interrupt_grace_seconds)
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
