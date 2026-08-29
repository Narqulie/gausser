"""Shared terminal presentation and image helpers for Gausser's human-facing CLIs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, SpinnerColumn
from rich.table import Table
from rich.text import Text

# Synthwave '84-inspired palette. Direct color styles keep renderables portable
# when a caller injects its own Console (as launcher tests and embedders do).
NEON_CYAN = "#2DE2E6"
NEON_MAGENTA = "#FF2A6D"
NEON_VIOLET = "#9B5DE5"
NEON_BLUE = "#2B59C3"
NEON_YELLOW = "#F9F871"
NEON_GREEN = "#05FFA1"
NEON_RED = "#FF3864"
NEON_MUTED = "#8A86A8"

QUESTIONARY_STYLE: tuple[tuple[str, str], ...] = (
    ("qmark", f"fg:{NEON_CYAN} bold"),
    ("question", f"fg:{NEON_CYAN} bold"),
    ("answer", f"fg:{NEON_YELLOW} bold"),
    ("pointer", f"fg:{NEON_MAGENTA} bold"),
    ("highlighted", f"fg:{NEON_MAGENTA} bold"),
    ("selected", f"fg:{NEON_GREEN}"),
    ("separator", f"fg:{NEON_VIOLET}"),
    ("instruction", f"fg:{NEON_MUTED}"),
    ("disabled", f"fg:{NEON_MUTED} italic"),
)

console = Console(highlight=False)
error_console = Console(stderr=True, highlight=False)

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
)


def is_image_file(path: Path) -> bool:
    """Return whether a real image, excluding macOS AppleDouble sidecars."""
    return (
        path.is_file()
        and not path.name.startswith("._")
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_images(path: Path) -> int:
    """Count real image files in *path*."""
    return sum(1 for file in path.iterdir() if is_image_file(file))


def list_images(path: Path) -> list[Path]:
    """Return sorted real image files in *path*."""
    return sorted(file for file in path.iterdir() if is_image_file(file))


def _title(mark: str, title: str, accent: str) -> Text:
    text = Text()
    text.append(f"{mark} ", style=f"bold {NEON_CYAN}")
    text.append(title.upper(), style=f"bold {accent}")
    return text


def stage_header(
    stage: str,
    subtitle: str,
    rows: Sequence[tuple[str, str | Text]],
    *,
    accent: str = NEON_MAGENTA,
) -> Panel:
    """Build the common neon stage card used by every interactive CLI."""
    grid = Table.grid(padding=(0, 2), expand=False)
    grid.add_column(style=f"bold {NEON_CYAN}", no_wrap=True)
    grid.add_column()
    for label, value in rows:
        grid.add_row(label.upper(), value if isinstance(value, Text) else Text(value))

    title = Text()
    title.append("◈ ", style=f"bold {NEON_CYAN}")
    title.append(stage.upper(), style=f"bold {accent}")
    title.append(" // ", style=NEON_MUTED)
    title.append(subtitle.upper(), style=f"bold {NEON_CYAN}")
    return Panel(
        grid,
        title=title,
        title_align="left",
        border_style=accent,
        box=box.ROUNDED,
        padding=(0, 2),
    )


def make_summary_table(title: str = "Summary", *, accent: str = NEON_VIOLET) -> Table:
    """Build a compact summary card with color-independent labels."""
    table = Table(
        title=_title("◇", title, accent),
        title_justify="left",
        border_style=accent,
        header_style=f"bold {NEON_CYAN}",
        box=box.ROUNDED,
        show_lines=False,
        pad_edge=True,
    )
    table.add_column("METRIC", style=f"bold {NEON_MAGENTA}")
    table.add_column("VALUE", justify="right")
    return table


def completion_panel(
    title: str,
    rows: Sequence[tuple[str, str | Text]],
    *,
    success: bool = True,
) -> Panel:
    """Render a final result card without relying on color for its meaning."""
    accent = NEON_GREEN if success else NEON_RED
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=f"bold {NEON_CYAN}", no_wrap=True)
    grid.add_column()
    for label, value in rows:
        grid.add_row(label.upper(), value if isinstance(value, Text) else Text(value))
    return Panel(
        grid,
        title=_title("◆" if success else "✕", title, accent),
        title_align="left",
        border_style=accent,
        box=box.ROUNDED,
        padding=(0, 2),
    )


def status_text(label: str, kind: str = "info") -> Text:
    """Return a color- and glyph-coded status suitable for tables or notices."""
    glyph, color = {
        "success": ("◆", NEON_GREEN),
        "warning": ("◇", NEON_YELLOW),
        "error": ("✕", NEON_RED),
        "running": ("▸", NEON_MAGENTA),
        "muted": ("·", NEON_MUTED),
        "info": ("◈", NEON_CYAN),
    }.get(kind, ("◈", NEON_CYAN))
    text = Text()
    text.append(f"{glyph} ", style=f"bold {color}")
    text.append(label, style=color)
    return text


def neon_spinner(*, style: str = NEON_CYAN) -> SpinnerColumn:
    return SpinnerColumn(spinner_name="dots12", style=f"bold {style}")


def neon_bar(*, width: int = 32) -> BarColumn:
    return BarColumn(
        bar_width=width,
        style=NEON_MUTED,
        complete_style=NEON_MAGENTA,
        finished_style=NEON_CYAN,
        pulse_style=NEON_VIOLET,
    )
