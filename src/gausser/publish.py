"""Transactional publication of staged scene directories."""

from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4


def replace_directory(staged: Path, destination: Path) -> None:
    """Publish a complete staged directory, restoring the old one on failure."""
    staged = Path(staged)
    destination = Path(destination)
    if not staged.is_dir():
        raise ValueError(f"staged directory does not exist: {staged}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(f".{destination.name}.backup-{uuid4().hex}")
    moved_old = False
    try:
        if destination.exists():
            destination.replace(backup)
            moved_old = True
        staged.replace(destination)
    except Exception:
        if moved_old and backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    else:
        if moved_old:
            shutil.rmtree(backup, ignore_errors=True)
