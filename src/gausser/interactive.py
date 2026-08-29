"""Shared normal-screen prompting for Gausser's human CLI entry points."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import shlex
import sys
from typing import Any, Protocol, TextIO, cast

import questionary
from questionary import Choice, Style
from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text

from .artifacts import Artifact, list_artifacts
from .controller import Stage
from .preflight import Readiness, inspect_direct_tool, inspect_scene
from .scene import ScenePaths
from .stage_options import (
    ResolvedStageOptions,
    StageOptionError,
    catalog,
    resolve_options,
)
from .ui import (
    NEON_MAGENTA,
    QUESTIONARY_STYLE,
    console as shared_console,
    stage_header,
    status_text as neon_status_text,
)


BARE_COMMAND_EPILOG = "Run without arguments in a terminal for guided setup."


class InteractiveUsageError(ValueError):
    """Bare interactive invocation cannot proceed safely."""


@dataclass(frozen=True)
class PromptChoice:
    """Framework-independent terminal prompt choice."""

    title: str
    value: str
    disabled: str | None = None


class PromptBackend(Protocol):
    """Small test seam around the normal-screen interactive prompt library."""

    def select(self, message: str, choices: Sequence[PromptChoice]) -> str | None: ...

    def text(
        self, message: str, *, default: str = "", path: bool = False
    ) -> str | None: ...

    def confirm(self, message: str, *, default: bool = False) -> bool | None: ...


class QuestionaryPrompter:
    """Arrow/Enter prompts that leave the terminal in its normal screen mode."""

    _STYLE = Style(list(QUESTIONARY_STYLE))

    def select(self, message: str, choices: Sequence[PromptChoice]) -> str | None:
        answer = questionary.select(
            message,
            choices=[
                Choice(item.title, value=item.value, disabled=item.disabled)
                for item in choices
            ],
            style=self._STYLE,
            use_shortcuts=True,
        ).ask()
        return cast(str | None, answer)

    def text(
        self, message: str, *, default: str = "", path: bool = False
    ) -> str | None:
        prompt = (
            questionary.path(message, default=default, style=self._STYLE)
            if path
            else questionary.text(message, default=default, style=self._STYLE)
        )
        return cast(str | None, prompt.ask())

    def confirm(self, message: str, *, default: bool = False) -> bool | None:
        return cast(
            bool | None,
            questionary.confirm(message, default=default, style=self._STYLE).ask(),
        )


ArtifactLister = Callable[[ScenePaths], list[Artifact]]
CommandBuilder = Callable[[ResolvedStageOptions], Sequence[str]]
SceneInspector = Callable[[ScenePaths, Sequence[Artifact]], list[Readiness]]
ToolInspector = Callable[[Stage], Readiness]

_CANCELLED = object()


class StageConfigurator:
    """Resolve one stage interactively without executing or mutating it."""

    def __init__(
        self,
        scene: ScenePaths,
        *,
        prompter: PromptBackend | None = None,
        console: Console | None = None,
        artifact_lister: ArtifactLister = list_artifacts,
    ) -> None:
        self.scene = scene
        self.prompter = prompter or QuestionaryPrompter()
        self.console = console or shared_console
        self.artifact_lister = artifact_lister

    def configure(
        self,
        stage: Stage,
        *,
        status: Readiness,
        command_builder: CommandBuilder,
    ) -> ResolvedStageOptions | None:
        """Prompt for one typed stage request and return it after confirmation."""
        locked_values: dict[str, object] = {}
        if status.required_action == "force":
            if stage not in {"sharpframes", "sammer"}:
                raise RuntimeError(f"unsupported replacement policy for stage {stage}")
            if (
                self.prompter.confirm(
                    f"{status.detail}. Replace transactionally?", default=False
                )
                is not True
            ):
                return None
            locked_values["force"] = True

        definition = catalog()[stage]
        self._render_stage_intro(definition, status)
        options = self._select_initial_options(
            stage, definition, locked_values=locked_values
        )
        if options is None:
            return None

        options = self._prompt_required_values(stage, definition, options)
        if options is None:
            return None

        if (
            stage == "colmap"
            and status.required_action == "existing_policy"
            and options.get("existing") == "error"
        ):
            policy = self.prompter.select(
                "COLMAP artifacts already exist",
                (
                    PromptChoice("Resume the staged reconstruction", "resume"),
                    PromptChoice("Replace it transactionally", "replace"),
                    PromptChoice("Back", "cancel"),
                ),
            )
            if policy in (None, "cancel"):
                return None
            options["existing"] = policy

        while True:
            try:
                resolved = resolve_options(stage, options, path_base=self.scene.root)
            except StageOptionError as exc:
                self.console.print(
                    neon_status_text(f"Settings are not valid: {exc}", "error")
                )
                action = self.prompter.select(
                    "Correct the settings",
                    (
                        PromptChoice("Edit standard settings", "edit-standard"),
                        PromptChoice("Edit advanced settings", "edit-advanced"),
                        PromptChoice("Back", "cancel"),
                    ),
                )
            else:
                self._render_resolved_command(
                    resolved.preset, command_builder(resolved), definition
                )
                action = self.prompter.select(
                    "Next action",
                    (
                        PromptChoice("Run this command", "run"),
                        PromptChoice("Edit standard settings", "edit-standard"),
                        PromptChoice("Edit advanced settings", "edit-advanced"),
                        PromptChoice("Back", "cancel"),
                    ),
                )
                if action == "run":
                    if (
                        self.prompter.confirm("Start this stage?", default=False)
                        is not True
                    ):
                        return None
                    return resolved
            if action in (None, "cancel"):
                return None
            options = self._edit_fields(
                stage,
                definition,
                options,
                advanced=action == "edit-advanced",
                locked_values=locked_values,
            )
            if options is None:
                return None
            options = self._prompt_required_values(stage, definition, options)
            if options is None:
                return None

    def _render_stage_intro(
        self, definition: Mapping[str, Any], status: Readiness
    ) -> None:
        self.console.print(
            stage_header(
                str(definition["title"]),
                str(definition["description"]),
                (("Scene", str(self.scene.root)), ("Readiness", status.detail)),
                accent=NEON_MAGENTA,
            )
        )
        for warning in cast(
            Sequence[Mapping[str, object]], definition.get("warnings", [])
        ):
            self.console.print(neon_status_text(str(warning["message"]), "warning"))

    def _select_initial_options(
        self,
        stage: Stage,
        definition: Mapping[str, Any],
        *,
        locked_values: Mapping[str, object] | None = None,
    ) -> dict[str, object] | None:
        locked = locked_values or {}
        presets = cast(Sequence[Mapping[str, Any]], definition["presets"])
        choices = [
            PromptChoice(
                f"{'★ ' if preset.get('recommended') else '  '}{preset['title']} — {preset['description']}",
                str(preset["id"]),
            )
            for preset in presets
        ]
        choices.append(PromptChoice("Custom settings", "custom"))
        selected = self.prompter.select("Choose a starting preset", choices)
        if selected is None:
            return None
        base = next(
            (preset for preset in presets if preset["id"] == selected), presets[0]
        )
        values = dict(cast(Mapping[str, object], base["values"]))
        values.update(locked)
        if selected != "custom":
            return values
        edited = self._edit_fields(
            stage,
            definition,
            values,
            advanced=False,
            locked_values=locked,
        )
        if edited is None:
            return None
        if self.prompter.confirm("Configure advanced settings?", default=False):
            return self._edit_fields(
                stage,
                definition,
                edited,
                advanced=True,
                locked_values=locked,
            )
        return edited

    def _prompt_required_values(
        self,
        stage: Stage,
        definition: Mapping[str, Any],
        options: dict[str, object],
    ) -> dict[str, object] | None:
        values = dict(options)
        fields = cast(Sequence[Mapping[str, Any]], definition["fields"])
        for field in fields:
            key = str(field["key"])
            if not field.get("required") or self._has_required_value(values.get(key)):
                continue
            prompted = self._prompt_field(stage, field, values.get(key), values)
            if prompted is _CANCELLED:
                return None
            if field.get("type") == "boolean" and prompted is False:
                self.console.print(
                    neon_status_text(
                        f"{field['label']} was not accepted; stage cancelled",
                        "warning",
                    )
                )
                return None
            values[key] = prompted
        return values

    @staticmethod
    def _has_required_value(value: object) -> bool:
        if isinstance(value, str):
            return bool(value.strip())
        return value is not None and value is not False

    def _edit_fields(
        self,
        stage: Stage,
        definition: Mapping[str, Any],
        options: dict[str, object],
        *,
        advanced: bool,
        locked_values: Mapping[str, object] | None = None,
    ) -> dict[str, object] | None:
        locked = locked_values or {}
        values = dict(options)
        values.update(locked)
        fields = cast(Sequence[Mapping[str, Any]], definition["fields"])
        for field in fields:
            if bool(field.get("advanced")) != advanced or not self._field_is_visible(
                field, values
            ):
                continue
            key = str(field["key"])
            if key in locked:
                continue
            prompted = self._prompt_field(
                stage, field, values.get(key, field.get("default")), values
            )
            if prompted is _CANCELLED:
                return None
            values[key] = prompted
        return values

    @staticmethod
    def _field_is_visible(
        field: Mapping[str, Any], values: Mapping[str, object]
    ) -> bool:
        condition = field.get("visible_when")
        if not isinstance(condition, Mapping):
            return True
        return values.get(str(condition.get("key"))) == condition.get("equals")

    def _prompt_field(
        self,
        stage: Stage,
        field: Mapping[str, Any],
        current: object,
        values: Mapping[str, object],
    ) -> object:
        del values
        key = str(field["key"])
        label = str(field["label"])
        kind = str(field["type"])
        if stage == "lod" and key == "source":
            candidates = [
                artifact
                for artifact in self.artifact_lister(self.scene)
                if artifact.kind != "lod" and artifact.format in {"ply", "sog"}
            ]
            if not candidates:
                self.console.print(
                    neon_status_text(
                        "No scene-owned PLY or SOG export is available", "warning"
                    )
                )
                return _CANCELLED
            selected = self.prompter.select(
                "Choose a scene-owned source export",
                tuple(
                    PromptChoice(
                        f"{artifact.relative_path}  ·  {artifact.size_bytes:,} bytes",
                        artifact.relative_path,
                    )
                    for artifact in candidates
                ),
            )
            return _CANCELLED if selected is None else selected
        if kind == "enum":
            selected = self.prompter.select(
                label,
                tuple(
                    PromptChoice(str(choice), str(choice))
                    for choice in cast(Sequence[object], field.get("choices", ()))
                ),
            )
            return _CANCELLED if selected is None else selected
        if kind == "boolean":
            answer = self.prompter.confirm(label, default=bool(current))
            return _CANCELLED if answer is None else answer

        default = "" if current is None else str(current)
        while True:
            text_answer = self.prompter.text(
                label, default=default, path=kind == "path"
            )
            if text_answer is None:
                return _CANCELLED
            if not text_answer.strip() and not field.get("required"):
                return None
            try:
                value = self._coerce_text_value(kind, text_answer)
            except ValueError as exc:
                self.console.print(neon_status_text(str(exc), "error"))
                continue
            minimum = field.get("minimum")
            maximum = field.get("maximum")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if isinstance(minimum, (int, float)) and value < minimum:
                    self.console.print(
                        neon_status_text(f"{label} must be at least {minimum}", "error")
                    )
                    continue
                if isinstance(maximum, (int, float)) and value > maximum:
                    self.console.print(
                        neon_status_text(f"{label} must be at most {maximum}", "error")
                    )
                    continue
            return value

    @staticmethod
    def _coerce_text_value(kind: str, value: str) -> object:
        if kind == "integer":
            try:
                return int(value)
            except ValueError as exc:
                raise ValueError("Enter a whole number.") from exc
        if kind == "decimal":
            try:
                return float(value)
            except ValueError as exc:
                raise ValueError("Enter a number.") from exc
        return value

    def _render_resolved_command(
        self,
        preset: str,
        command: Sequence[str],
        definition: Mapping[str, Any],
    ) -> None:
        preset_line = Text("Preset: ", style="bold")
        preset_line.append(preset)
        self.console.print(preset_line)
        self.console.print(Text(str(definition["description"]), style="dim"))
        self.console.print(
            Syntax(
                shlex.join(command), "bash", word_wrap=True, background_color="default"
            )
        )


def prepare_stage_arguments(
    stage: Stage,
    argv: list[str] | None,
    *,
    program: str,
    scene_root: str | Path | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    prompter: PromptBackend | None = None,
    console: Console | None = None,
    artifact_lister: ArtifactLister = list_artifacts,
    inspector: SceneInspector = inspect_scene,
    tool_inspector: ToolInspector = inspect_direct_tool,
) -> list[str] | None:
    """Return explicit args unchanged or configure a bare TTY invocation."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        return arguments

    input_stream = stdin or sys.stdin
    output_stream = stdout or sys.stdout
    if not input_stream.isatty() or not output_stream.isatty():
        raise InteractiveUsageError(
            f"bare `{program}` is interactive and requires a terminal; "
            "pass explicit arguments for non-interactive use"
        )

    paths = ScenePaths.from_root(scene_root or Path.cwd())
    if not paths.root.is_dir():
        raise InteractiveUsageError(f"scene directory does not exist: {paths.root}")

    tool_status = tool_inspector(stage)
    if tool_status.state == "unavailable":
        raise InteractiveUsageError(f"{program} is unavailable: {tool_status.detail}")

    artifacts = artifact_lister(paths)
    statuses = {item.stage: item for item in inspector(paths, artifacts)}
    status = statuses.get(stage)
    if status is None:
        raise InteractiveUsageError(f"no scene readiness information for {program}")
    if status.state not in {"ready", "completed"}:
        raise InteractiveUsageError(f"{program} is unavailable: {status.detail}")

    active_console = console or shared_console
    configurator = StageConfigurator(
        paths,
        prompter=prompter,
        console=active_console,
        artifact_lister=artifact_lister,
    )
    try:
        resolved = configurator.configure(
            stage,
            status=status,
            command_builder=lambda options: [program, *options.arguments],
        )
    except (EOFError, KeyboardInterrupt):
        resolved = None
    if resolved is None:
        active_console.print(neon_status_text("Leaving scene unchanged", "muted"))
        return None
    return list(resolved.arguments)
