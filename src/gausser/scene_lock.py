"""Cross-process mutation lease for one scene."""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


SCENE_LOCK_FD_ENV = "GAUSSER_SCENE_LOCK_FD"


class SceneLockError(RuntimeError):
    """Raised when a scene mutation lease cannot be acquired or adopted."""


def _lock_path(scene_root: Path) -> Path:
    return scene_root.expanduser().resolve() / ".gausser" / "scene.lock"


class SceneLock:
    """An exclusive scene mutation lease, either owned or inherited."""

    def __init__(self, scene_root: Path, *, inherited_fd: int | None = None) -> None:
        self.scene_root = scene_root.expanduser().resolve()
        self.path = _lock_path(self.scene_root)
        self._inherited_fd = inherited_fd
        self._stream: TextIO | None = None
        self._borrowed = False

    def acquire(self) -> None:
        if self._stream is not None or self._borrowed:
            raise SceneLockError(f"Scene lock is already held by this process: {self.scene_root}")

        if self._inherited_fd is not None:
            self._adopt_inherited(self._inherited_fd)
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.close()
            raise SceneLockError(
                f"Scene is already owned by another Gausser process: {self.scene_root}"
            ) from exc
        except Exception:
            stream.close()
            raise
        self._stream = stream

    def _adopt_inherited(self, fd: int) -> None:
        try:
            descriptor = os.fstat(fd)
            lock_file = self.path.stat()
        except (OSError, ValueError) as exc:
            raise SceneLockError(
                f"Inherited scene lock descriptor is invalid for {self.scene_root}"
            ) from exc

        if (descriptor.st_dev, descriptor.st_ino) != (lock_file.st_dev, lock_file.st_ino):
            raise SceneLockError(
                f"Inherited scene lock descriptor does not belong to {self.scene_root}"
            )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SceneLockError(
                f"Inherited descriptor does not own the scene lock: {self.scene_root}"
            ) from exc
        self._borrowed = True

    def release(self) -> None:
        if self._borrowed:
            # The parent owns the open-file description and remains responsible
            # for unlocking it after this child has exited and finalized its run.
            self._borrowed = False
            return
        if self._stream is None:
            return
        # Do not issue LOCK_UN: a launched child may hold a duplicate of this
        # open-file description. Closing our descriptor preserves the lease until
        # the final inherited descriptor closes; with no child, close releases it.
        self._stream.close()
        self._stream = None

    def fileno(self) -> int:
        if self._borrowed and self._inherited_fd is not None:
            return self._inherited_fd
        if self._stream is None:
            raise SceneLockError("Scene lock is not held")
        return self._stream.fileno()

    def __enter__(self) -> "SceneLock":
        self.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


@contextmanager
def acquire_scene_lock(scene_root: Path) -> Iterator[SceneLock]:
    inherited_value = os.environ.get(SCENE_LOCK_FD_ENV)
    inherited_fd: int | None = None
    if inherited_value is not None:
        try:
            inherited_fd = int(inherited_value)
        except ValueError as exc:
            raise SceneLockError(
                f"Inherited scene lock descriptor is invalid for {Path(scene_root).resolve()}"
            ) from exc

    lock = SceneLock(scene_root, inherited_fd=inherited_fd)
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()
