from __future__ import annotations

import io

from rich.console import Console

from gausser.ui import (
    NEON_CYAN,
    NEON_MAGENTA,
    QUESTIONARY_STYLE,
    completion_panel,
    make_summary_table,
    stage_header,
    status_text,
)


def _render(renderable, *, color: bool) -> str:
    output = io.StringIO()
    console = Console(
        file=output,
        force_terminal=color,
        color_system="truecolor" if color else None,
        width=100,
    )
    console.print(renderable)
    return output.getvalue()


def test_stage_header_has_shared_synthwave_structure_and_plain_text_fallback() -> None:
    header = stage_header(
        "SAMMER",
        "Masking signal",
        (("Scene", "/tmp/neon-fountain"), ("Target", "statue")),
    )

    plain = _render(header, color=False)

    assert "◈ SAMMER // MASKING SIGNAL" in plain
    assert "SCENE" in plain
    assert "/tmp/neon-fountain" in plain
    assert "\x1b[" not in plain


def test_synthwave_renderables_emit_truecolor_only_for_a_terminal() -> None:
    rendered = _render(status_text("Pipeline online", "success"), color=True)

    assert "Pipeline online" in rendered
    assert "\x1b[" in rendered
    assert (
        NEON_CYAN.removeprefix("#") not in rendered
    )  # Rich emits numeric ANSI, not markup text.


def test_summary_and_completion_cards_keep_state_visible_without_color() -> None:
    summary = make_summary_table("Training summary")
    summary.add_row("Splats", "1,234,567")
    completion = completion_panel(
        "Run complete",
        (("Export", "runs/brush/neon/scene.ply"), ("Status", "Published")),
    )

    summary_text = _render(summary, color=False)
    completion_text = _render(completion, color=False)

    assert "TRAINING SUMMARY" in summary_text
    assert "1,234,567" in summary_text
    assert "◆ RUN COMPLETE" in completion_text
    assert "Published" in completion_text


def test_cards_render_markup_like_paths_literally() -> None:
    adversarial = "/tmp/[red]scene[/red]/[bold]image.jpg[/bold]"

    header = _render(
        stage_header("sammer", "masking", (("Scene", adversarial),)), color=False
    )
    completion = _render(
        completion_panel("published", (("Output", adversarial),)), color=False
    )

    assert adversarial in header
    assert adversarial in completion


def test_sammer_progress_renders_markup_like_filenames_literally(monkeypatch) -> None:
    from sammer import cli

    output = io.StringIO()
    monkeypatch.setattr(
        cli,
        "console",
        Console(file=output, force_terminal=False, color_system=None, width=120),
    )
    progress = cli._make_progress()
    progress.start()
    task = progress.add_task(
        "[red]frame.jpg[/red]",
        total=1,
        status="pending",
        hit_rate="0%",
        max_conf="—",
    )
    progress.update(task, completed=1)
    progress.stop()

    assert "[red]frame.jpg[/red]" in output.getvalue()


def test_rich_stage_clis_use_the_shared_synthwave_header(tmp_path) -> None:
    from brusher.cli import _header as brush_header
    from colmapper.cli import _header as colmap_header
    from sammer.cli import _header as sammer_header

    headers = (
        sammer_header(tmp_path / "images", 12, "statue", "mps", 0.5, None),
        colmap_header(tmp_path, tmp_path / "images", 12, "sequential", True),
        brush_header(tmp_path, 30_000, tmp_path / "runs" / "scene.ply"),
    )

    rendered = "".join(_render(header, color=False) for header in headers)
    assert "◈ SAMMER // SAM3 MASKING SIGNAL" in rendered
    assert "◈ COLMAPPER // SPARSE RECONSTRUCTION SIGNAL" in rendered
    assert "◈ BRUSHER // GAUSSIAN TRAINING SIGNAL" in rendered


def test_questionary_palette_uses_the_same_neon_accents() -> None:
    styles = dict(QUESTIONARY_STYLE)

    assert NEON_CYAN in styles["qmark"]
    assert NEON_MAGENTA in styles["pointer"]
    assert styles["disabled"].endswith("italic")
