"""Shared UI utilities for gausser tools (sammer, colmapper, brusher)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.table import Table

console = Console()

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"})


# ── Convenience imports ─────────────────────────────────────────────────────────


def is_image_file(path: Path) -> bool:
    """Return whether a real image, excluding macOS AppleDouble sidecars."""
    return path.is_file() and not path.name.startswith("._") and path.suffix.lower() in IMAGE_EXTENSIONS


def count_images(path: Path) -> int:
    """Count real image files in *path*."""
    return sum(1 for file in path.iterdir() if is_image_file(file))


def list_images(path: Path) -> list[Path]:
    """Return sorted real image files in *path*."""
    return sorted(file for file in path.iterdir() if is_image_file(file))


# ── Common table / formatting helpers ────────────────────────────────────────────


def make_summary_table(title: str = "Summary") -> Table:
    table = Table(title=title, border_style="dim", show_lines=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    return table
