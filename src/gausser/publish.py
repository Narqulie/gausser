"""Transactional publication of staged scene directories."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
from typing import Iterator
from uuid import uuid4


@contextmanager
def publish_directory(staged: Path, destination: Path) -> Iterator[None]:
    """Publish ``staged`` and roll back it and its prior destination on error.

    Metadata recording belongs inside this context. A stage is therefore never
    visible at its destination unless its companion manifest update succeeds.
    """
    staged = Path(staged)
    destination = Path(destination)
    if not staged.is_dir():
        raise ValueError(f"staged directory does not exist: {staged}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(f".{destination.name}.backup-{uuid4().hex}")
    moved_old = False
    published = False
    try:
        if destination.exists():
            destination.replace(backup)
            moved_old = True
        staged.replace(destination)
        published = True
        yield
    except Exception:
        if published and destination.exists():
            destination.replace(staged)
        if moved_old and backup.exists():
            backup.replace(destination)
        raise
    else:
        if moved_old:
            shutil.rmtree(backup, ignore_errors=True)


def replace_directory(staged: Path, destination: Path) -> None:
    """Publish a complete staged directory without a companion metadata write."""
    with publish_directory(staged, destination):
        pass
