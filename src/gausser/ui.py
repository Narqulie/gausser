"""Shared UI utilities for gausser tools (sammer, colmapper, brusher)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.table import Table

console = Console()

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"})


# ── Convenience imports ─────────────────────────────────────────────────────────


def count_images(path: Path) -> int:
    """Count image files in *path*."""
    return sum(1 for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)


def list_images(path: Path) -> list[Path]:
    """Return sorted list of image files in *path*."""
    return sorted(
        f for f in path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


# ── Common table / formatting helpers ────────────────────────────────────────────


def make_summary_table(title: str = "Summary") -> Table:
    table = Table(title=title, border_style="dim", show_lines=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    return table
