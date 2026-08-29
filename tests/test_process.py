from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
import time

import pytest

from gausser.process import command_environment, resolve_command, run_command
from gausser.scene_lock import SceneLock, SceneLockError, acquire_scene_lock


def test_command_environment_restores_shell_path_after_pyenv_dispatch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PYENV_ROOT", "/tmp/example-pyenv")
    monkeypatch.setenv("PYENV_VERSION", "3.12.8")
    monkeypatch.setenv(
        "PATH",
        "/tmp/example-pyenv/versions/3.12.8/bin:/Users/test/.cargo/bin:/tmp/example-pyenv/shims",
    )

    environment = command_environment()

    assert environment["PATH"] == "/Users/test/.cargo/bin:/tmp/example-pyenv/shims"


def test_command_resolution_uses_the_restored_shell_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pyenv_bin = tmp_path / "pyenv" / "versions" / "3.12.8" / "bin"
    shell_bin = tmp_path / "shell-bin"
    pyenv_bin.mkdir(parents=True)
    shell_bin.mkdir()
    for directory in (pyenv_bin, shell_bin):
        command = directory / "stage-command"
        command.write_text("#!/bin/sh\n")
        command.chmod(0o755)
    monkeypatch.setenv("PYENV_ROOT", str(tmp_path / "pyenv"))
    monkeypatch.setenv("PYENV_VERSION", "3.12.8")
    monkeypatch.setenv("PATH", f"{pyenv_bin}{os.pathsep}{shell_bin}")

    assert resolve_command("stage-command") == str(shell_bin / "stage-command")


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable PATH test")
def test_runner_resolves_stage_commands_from_the_restored_shell_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pyenv_bin = tmp_path / "pyenv" / "versions" / "3.12.8" / "bin"
    shell_bin = tmp_path / "shell-bin"
    pyenv_bin.mkdir(parents=True)
    shell_bin.mkdir()
    for directory, output in ((pyenv_bin, "injected"), (shell_bin, "shell")):
        command = directory / "stage-command"
        command.write_text(f"#!/bin/sh\necho {output}\n")
        command.chmod(0o755)
    monkeypatch.setenv("PYENV_ROOT", str(tmp_path / "pyenv"))
    monkeypatch.setenv("PYENV_VERSION", "3.12.8")
    monkeypatch.setenv("PATH", f"{pyenv_bin}{os.pathsep}{shell_bin}")

    result = run_command(["stage-command"], cwd=tmp_path)

    assert result.succeeded
    assert result.output_tail == ("shell",)


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


@pytest.mark.skipif(os.name == "nt", reason="POSIX file-descriptor inheritance test")
def test_runner_keeps_scene_lock_owned_after_parent_releases(tmp_path: Path):
    lock = SceneLock(tmp_path)
    lock.acquire()
    started = threading.Event()
    result: list[object] = []

    def run() -> None:
        result.append(run_command(
            [sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(0.3)"],
            cwd=tmp_path,
            inherited_fds=(lock.fileno(),),
            on_line=lambda line: started.set() if line == "ready" else None,
        ))

    thread = threading.Thread(target=run)
    thread.start()
    assert started.wait(timeout=2)
    lock.release()
    with pytest.raises(SceneLockError, match="already owned"):
        with acquire_scene_lock(tmp_path):
            pass
    thread.join(timeout=2)
    assert result[0].succeeded


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
