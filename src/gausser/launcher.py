"""Lightweight interactive command palette for the primary Gausser stage CLIs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Protocol, Sequence, cast

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .artifacts import Artifact, list_artifacts
from .controller import SceneController, Stage, StageRequest, StageUnavailable
from .interactive import (
    PromptBackend,
    PromptChoice,
    QuestionaryPrompter,
    StageConfigurator,
)
from .preflight import Readiness, inspect_scene, inspect_tools
from .process import CommandResult, LineCallback
from .scene import SceneError, ScenePaths
from .ui import (
    NEON_CYAN,
    NEON_MAGENTA,
    NEON_MUTED,
    NEON_VIOLET,
    console as shared_console,
    stage_header,
    status_text as neon_status_text,
)


_STAGE_ORDER: tuple[Stage, ...] = (
    "sharpframes",
    "sammer",
    "colmap",
    "brush",
    "msplat",
    "lod",
)
_STAGE_TITLES: Mapping[Stage, str] = {
    "sharpframes": "Sharp Frames",
    "sammer": "SAM3 Masking",
    "colmap": "COLMAP Reconstruction",
    "brush": "Brush Training",
    "msplat": "Msplat Training",
    "lod": "Streamed LOD",
}
_RUNNABLE_STATES = frozenset({"ready", "completed"})


class ControllerBackend(Protocol):
    def readiness(self) -> dict[str, str]: ...

    def command_for(self, request: StageRequest) -> list[str]: ...

    def run(
        self,
        request: StageRequest,
        *,
        on_line: LineCallback | None = None,
    ) -> CommandResult: ...


@dataclass(frozen=True)
class ChainRecipe:
    id: str
    title: str
    description: str
    stages: tuple[Stage, ...]


@dataclass(frozen=True)
class StageOutcome:
    stage: Stage
    state: str
    result: CommandResult | None = None


_CHAIN_RECIPES: tuple[ChainRecipe, ...] = (
    ChainRecipe(
        "masked-brush",
        "Masked Brush",
        "SAM3 masks → COLMAP → Brush",
        ("sammer", "colmap", "brush"),
    ),
    ChainRecipe(
        "unmasked-msplat",
        "Unmasked Msplat",
        "COLMAP → experimental unmasked Msplat",
        ("colmap", "msplat"),
    ),
    ChainRecipe(
        "brush-lod",
        "Brush to streamed LOD",
        "Brush → explicit streamed LOD export",
        ("brush", "lod"),
    ),
)


Inspector = Callable[[ScenePaths, Sequence[Artifact]], list[Readiness]]
ArtifactLister = Callable[[ScenePaths], list[Artifact]]
ToolInspector = Callable[[], list[Readiness]]


class Launcher:
    """Scene overview and interactive delegation to the real stage CLIs."""

    def __init__(
        self,
        scene: ScenePaths,
        *,
        prompter: PromptBackend | None = None,
        console: Console | None = None,
        controller: ControllerBackend | None = None,
        inspector: Inspector = inspect_scene,
        artifact_lister: ArtifactLister = list_artifacts,
        tool_inspector: ToolInspector = inspect_tools,
    ) -> None:
        self.scene = scene
        self.prompter = prompter or QuestionaryPrompter()
        self.console = console or shared_console
        self.controller = controller or SceneController(scene)
        self.inspector = inspector
        self.artifact_lister = artifact_lister
        self.tool_inspector = tool_inspector
        self._tool_statuses: dict[str, Readiness] = {}

    def run(self) -> int:
        """Render, select, execute, and refresh until the user exits."""
        with self.console.status(
            f"[bold {NEON_CYAN}]◈ Scanning stage signals…[/bold {NEON_CYAN}]",
            spinner="dots12",
            spinner_style=NEON_MAGENTA,
        ):
            self._tool_statuses = {item.stage: item for item in self.tool_inspector()}
        while True:
            artifacts = self.artifact_lister(self.scene)
            scene_statuses = {
                item.stage: item for item in self.inspector(self.scene, artifacts)
            }
            statuses = self._merge_readiness(scene_statuses)
            self._render_dashboard(statuses, artifacts)
            selection = self.prompter.select(
                "Choose a step", self._menu_choices(statuses)
            )
            if selection is None or selection == "exit":
                self.console.print(neon_status_text("Leaving scene unchanged", "muted"))
                return 0
            if selection.startswith("chain:"):
                recipe = next(
                    item
                    for item in _CHAIN_RECIPES
                    if item.id == selection.removeprefix("chain:")
                )
                self._run_chain(recipe)
                continue
            stage = cast(Stage, selection)
            self._configure_and_run(stage, scene_statuses.get(stage))

    def _merge_readiness(
        self, scene_statuses: Mapping[str, Readiness]
    ) -> dict[str, Readiness]:
        statuses = dict(scene_statuses)
        for stage in _STAGE_ORDER:
            scene_status = statuses.get(stage)
            tool_status = self._tool_statuses.get(stage)
            if (
                scene_status is not None
                and scene_status.state in _RUNNABLE_STATES
                and tool_status is not None
                and tool_status.state == "unavailable"
            ):
                statuses[stage] = Readiness(stage, "unavailable", tool_status.detail)
        return statuses

    def _render_dashboard(
        self, statuses: Mapping[str, Readiness], artifacts: Sequence[Artifact]
    ) -> None:
        published_runs: dict[str, set[str]] = {
            "brush": set(),
            "msplat": set(),
            "lod": set(),
        }
        for artifact in artifacts:
            if artifact.kind in {"brush", "msplat"} and artifact.format in {
                "ply",
                "sog",
                "spz",
                "splat",
            }:
                published_runs[artifact.kind].add(artifact.run_id)
            elif artifact.kind == "lod" and artifact.format == "lod":
                published_runs["lod"].add(artifact.run_id)

        self.console.print(
            stage_header(
                "GAUSSER",
                "Synthwave ’84 command deck",
                (
                    ("Scene", self.scene.root.name or str(self.scene.root)),
                    ("Root", str(self.scene.root)),
                    ("Artifacts", str(len(artifacts))),
                ),
                accent=NEON_MAGENTA,
            )
        )

        table = Table(
            show_header=True,
            header_style=f"bold {NEON_CYAN}",
            border_style=NEON_VIOLET,
            box=box.SIMPLE_HEAVY,
            pad_edge=False,
        )
        table.add_column("STEP", min_width=22, style=f"bold {NEON_MAGENTA}")
        table.add_column("SIGNAL", min_width=14)
        table.add_column("SCENE DETAIL", style=NEON_MUTED)
        for stage in _STAGE_ORDER:
            status = statuses.get(
                stage, Readiness(stage, "unavailable", "No readiness information")
            )
            detail = status.detail
            count = len(published_runs.get(stage, set()))
            if count:
                detail = f"{detail} · {count} published run{'s' if count != 1 else ''}"
            table.add_row(
                _STAGE_TITLES[stage],
                self._status_text(status.state),
                Text(detail, style=NEON_MUTED),
            )
        self.console.print(table)
        self.console.print()

    @staticmethod
    def _status_text(state: str) -> Text:
        if state == "completed":
            return neon_status_text("completed", "success")
        if state == "ready":
            return neon_status_text("ready", "info")
        return neon_status_text("unavailable", "muted")

    def _menu_choices(self, statuses: Mapping[str, Readiness]) -> list[PromptChoice]:
        choices: list[PromptChoice] = []
        for stage in _STAGE_ORDER:
            status = statuses.get(
                stage, Readiness(stage, "unavailable", "No readiness information")
            )
            symbol = (
                "✓"
                if status.state == "completed"
                else "●"
                if status.state == "ready"
                else "○"
            )
            disabled = None if status.state in _RUNNABLE_STATES else status.detail
            title = f"{symbol}  {_STAGE_TITLES[stage]}"
            if disabled is None:
                title += f"  —  {status.detail}"
            choices.append(PromptChoice(title, stage, disabled))
        for recipe in _CHAIN_RECIPES:
            first = statuses.get(recipe.stages[0])
            disabled = self._chain_tool_blocker(recipe)
            if disabled is None and (
                first is None or first.state not in _RUNNABLE_STATES
            ):
                disabled = (
                    first.detail if first is not None else "First stage is unavailable"
                )
            choices.append(
                PromptChoice(
                    f"↳  {recipe.title}  —  {recipe.description}",
                    f"chain:{recipe.id}",
                    disabled,
                )
            )
        choices.append(PromptChoice("Exit", "exit"))
        return choices

    def _chain_tool_blocker(self, recipe: ChainRecipe) -> str | None:
        for stage in recipe.stages:
            status = self._tool_statuses.get(stage)
            if status is not None and status.state == "unavailable":
                return f"{_STAGE_TITLES[stage]}: {status.detail}"
        return None

    def _configure_and_run(
        self,
        stage: Stage,
        scene_status: Readiness | None = None,
    ) -> StageOutcome:
        tool_status = self._tool_statuses.get(stage)
        if tool_status is not None and tool_status.state == "unavailable":
            self.console.print(
                neon_status_text(
                    f"{_STAGE_TITLES[stage]} is unavailable: {tool_status.detail}",
                    "warning",
                )
            )
            return StageOutcome(stage, "unavailable")

        state = self.controller.readiness().get(stage, "unavailable")
        if state not in _RUNNABLE_STATES:
            self.console.print(
                neon_status_text(
                    f"{_STAGE_TITLES[stage]} is currently {state}", "warning"
                )
            )
            return StageOutcome(stage, "unavailable")

        status = scene_status or Readiness(stage, state, state)
        configurator = StageConfigurator(
            self.scene,
            prompter=self.prompter,
            console=self.console,
            artifact_lister=self.artifact_lister,
        )
        resolved = configurator.configure(
            stage,
            status=status,
            command_builder=lambda options: self.controller.command_for(
                StageRequest(stage, options.arguments, options.as_settings())
            ),
        )
        if resolved is None:
            return StageOutcome(stage, "configuration_cancelled")
        return self._execute(stage, resolved.arguments, resolved.as_settings())

    def _execute(
        self,
        stage: Stage,
        arguments: tuple[str, ...],
        settings: Mapping[str, Any],
    ) -> StageOutcome:
        self.console.rule(
            Text(
                f"RUNNING // {_STAGE_TITLES[stage].upper()}",
                style=f"bold {NEON_MAGENTA}",
            ),
            style=NEON_VIOLET,
        )
        try:
            result = self.controller.run(
                StageRequest(stage, arguments, settings),
                on_line=lambda line: self.console.print(Text(line)),
            )
        except (OSError, SceneError, StageUnavailable) as exc:
            self.console.print(
                neon_status_text(
                    f"Could not start {_STAGE_TITLES[stage]}: {exc}", "error"
                )
            )
            return StageOutcome(stage, "failed")
        if result.cancelled:
            state = "cancelled"
            self.console.print(
                neon_status_text(
                    f"Cancelled after {result.duration_seconds:.1f}s", "warning"
                )
            )
        elif result.succeeded:
            state = "succeeded"
            self.console.print(
                neon_status_text(
                    f"Succeeded in {result.duration_seconds:.1f}s", "success"
                )
            )
        else:
            state = "failed"
            self.console.print(
                neon_status_text(
                    f"Failed with exit code {result.returncode} after {result.duration_seconds:.1f}s",
                    "error",
                )
            )
        return StageOutcome(stage, state, result)

    def _run_chain(self, recipe: ChainRecipe) -> None:
        sequence = " → ".join(_STAGE_TITLES[stage] for stage in recipe.stages)
        self.console.rule(
            Text(
                f"GUIDED SEQUENCE // {recipe.title.upper()}",
                style=f"bold {NEON_MAGENTA}",
            ),
            style=NEON_VIOLET,
        )
        self.console.print(recipe.description)
        self.console.print(f"[bold]{sequence}[/bold]")
        tool_blocker = self._chain_tool_blocker(recipe)
        if tool_blocker is not None:
            self.console.print(
                neon_status_text(f"Sequence unavailable: {tool_blocker}", "warning")
            )
            return
        if (
            self.prompter.confirm("Configure and run this sequence?", default=False)
            is not True
        ):
            return

        outcomes: list[StageOutcome] = []
        for stage in recipe.stages:
            artifacts = self.artifact_lister(self.scene)
            scene_statuses = {
                item.stage: item for item in self.inspector(self.scene, artifacts)
            }
            state = self.controller.readiness().get(stage, "unavailable")
            if state not in _RUNNABLE_STATES:
                outcomes.append(StageOutcome(stage, "unavailable"))
                break
            outcome = self._configure_and_run(stage, scene_statuses.get(stage))
            outcomes.append(outcome)
            if outcome.state != "succeeded":
                break
        self._render_chain_summary(recipe, outcomes)

    def _render_chain_summary(
        self, recipe: ChainRecipe, outcomes: Sequence[StageOutcome]
    ) -> None:
        states = {outcome.stage: outcome.state for outcome in outcomes}
        completed = len(outcomes) == len(recipe.stages) and all(
            outcome.state == "succeeded" for outcome in outcomes
        )
        heading = "Completed" if completed else "Stopped"
        style = "green" if completed else "yellow"
        self.console.print(f"[{style}]{heading}:[/{style}] {recipe.title}")
        for stage in recipe.stages:
            state = states.get(stage, "not started")
            self.console.print(f"  {_STAGE_TITLES[stage]}: {state} ({stage})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gausser",
        description="Interactive scene overview and launcher for the independent Gausser stage CLIs.",
    )
    parser.add_argument(
        "scene",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="Scene directory (default: current directory)",
    )
    return parser


def main(argv: list[str] | None = None, *, require_tty: bool = True) -> int:
    args = build_parser().parse_args(argv)
    paths = ScenePaths.from_root(args.scene)
    if not paths.root.is_dir():
        shared_console.print(
            neon_status_text(f"Scene directory does not exist: {paths.root}", "error")
        )
        return 2
    if require_tty and (not sys.stdin.isatty() or not sys.stdout.isatty()):
        shared_console.print(
            neon_status_text(
                "gausser is an interactive launcher and requires a terminal", "error"
            )
        )
        shared_console.print(
            "Use sharpframes, sammer, colmapper, brusher, msplatter, or lodder directly for automation."
        )
        return 2
    return Launcher(paths).run()
