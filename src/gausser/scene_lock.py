"""Host-local advisory ownership locks for scene-mutating engine operations."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import tempfile
from typing import Iterator, TextIO

from .scene import SceneError


class SceneLockError(SceneError):
    """Another Gausser engine currently owns a scene mutation."""


def _lock_path(root: Path) -> Path:
    canonical = str(root.expanduser().resolve()).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return Path(tempfile.gettempdir()) / f"gausser-scene-{digest}.lock"


class SceneLock:
    """An exclusive non-blocking, host-local lock for one canonical scene root."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self._stream: TextIO | None = None

    def acquire(self) -> None:
        if self._stream is not None:
            raise RuntimeError("scene lock is already acquired")
        stream = _lock_path(self.root).open("a+", encoding="utf-8")
        try:
            _lock_nonblocking(stream)
        except Exception:
            stream.close()
            raise
        self._stream = stream

    def release(self) -> None:
        if self._stream is None:
            return
        stream, self._stream = self._stream, None
        try:
            _unlock(stream)
        finally:
            stream.close()

    def __enter__(self) -> "SceneLock":
        self.acquire()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.release()


@contextmanager
def acquire_scene_lock(root: Path | str) -> Iterator[SceneLock]:
    lock = SceneLock(root)
    with lock:
        yield lock


def _lock_nonblocking(stream: TextIO) -> None:
    try:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SceneLockError(f"scene is already owned by another Gausser engine: {Path(stream.name)}") from exc


def _unlock(stream: TextIO) -> None:
    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
