from __future__ import annotations

from collections import deque
from io import StringIO
import importlib
from pathlib import Path
from typing import Sequence

import pytest
from rich.console import Console

from gausser.interactive import (
    InteractiveUsageError,
    PromptBackend,
    PromptChoice,
    StageConfigurator,
    prepare_stage_arguments,
)
from gausser.preflight import Readiness
from gausser.scene import ScenePaths


class TerminalBuffer(StringIO):
    def __init__(self, *, tty: bool) -> None:
        super().__init__()
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


class ScriptedPrompter(PromptBackend):
    def __init__(
        self,
        *,
        selections: Sequence[str | None] = (),
        texts: Sequence[str | None] = (),
        confirmations: Sequence[bool | None] = (),
    ) -> None:
        self.selections = deque(selections)
        self.texts = deque(texts)
        self.confirmations = deque(confirmations)
        self.select_calls: list[tuple[str, tuple[PromptChoice, ...]]] = []

    def select(self, message: str, choices: Sequence[PromptChoice]) -> str | None:
        self.select_calls.append((message, tuple(choices)))
        return self.selections.popleft()

    def text(
        self, message: str, *, default: str = "", path: bool = False
    ) -> str | None:
        del message, default, path
        return self.texts.popleft()

    def confirm(self, message: str, *, default: bool = False) -> bool | None:
        del message, default
        return self.confirmations.popleft()


def _console() -> tuple[Console, StringIO]:
    output = StringIO()
    return Console(
        file=output, force_terminal=False, color_system=None, width=120
    ), output


def test_explicit_arguments_bypass_interactive_inspection() -> None:
    arguments = ["--prompt", "fountain"]

    resolved = prepare_stage_arguments(
        "sammer",
        arguments,
        program="sammer",
        inspector=lambda *_args: pytest.fail("scene inspection must not run"),
        tool_inspector=lambda _stage: pytest.fail("tool inspection must not run"),
    )

    assert resolved == arguments
    assert resolved is not arguments


def test_bare_stage_requires_a_real_terminal_before_scene_inspection() -> None:
    with pytest.raises(InteractiveUsageError, match="requires a terminal"):
        prepare_stage_arguments(
            "sammer",
            [],
            program="sammer",
            stdin=TerminalBuffer(tty=False),
            stdout=TerminalBuffer(tty=True),
            inspector=lambda *_args: pytest.fail("scene inspection must not run"),
            tool_inspector=lambda _stage: pytest.fail("tool inspection must not run"),
        )


def test_bare_stage_rejects_missing_scene_without_metadata_side_effect(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "missing"

    with pytest.raises(InteractiveUsageError, match="does not exist"):
        prepare_stage_arguments(
            "sammer",
            [],
            program="sammer",
            scene_root=scene,
            stdin=TerminalBuffer(tty=True),
            stdout=TerminalBuffer(tty=True),
        )

    assert not scene.exists()


def test_bare_stage_reports_its_own_unavailable_tool_before_prompting(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "scene"
    scene.mkdir()

    with pytest.raises(InteractiveUsageError, match="brusher is unavailable"):
        prepare_stage_arguments(
            "brush",
            [],
            program="brusher",
            scene_root=scene,
            stdin=TerminalBuffer(tty=True),
            stdout=TerminalBuffer(tty=True),
            prompter=ScriptedPrompter(),
            tool_inspector=lambda stage: Readiness(
                stage, "unavailable", "brush is not on PATH"
            ),
            inspector=lambda *_args: pytest.fail("scene inspection must not run"),
        )


def test_bare_sammer_prompts_required_value_and_uses_typed_defaults(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "scene"
    (scene / "images").mkdir(parents=True)
    console, output = _console()
    prompter = ScriptedPrompter(
        selections=("standard-mask", "run"),
        texts=("fountain",),
        confirmations=(True,),
    )

    arguments = prepare_stage_arguments(
        "sammer",
        [],
        program="sammer",
        scene_root=scene,
        stdin=TerminalBuffer(tty=True),
        stdout=TerminalBuffer(tty=True),
        prompter=prompter,
        console=console,
        artifact_lister=lambda _scene: [],
        inspector=lambda _scene, _artifacts: [
            Readiness("sammer", "ready", "Images are ready")
        ],
        tool_inspector=lambda stage: Readiness(stage, "ready", "runtime ready"),
    )

    assert arguments is not None
    assert arguments[:2] == ["--prompt", "fountain"]
    assert arguments[arguments.index("--score-threshold") + 1] == "0.5"
    assert arguments[arguments.index("--device") + 1] == "auto"
    assert "sammer --prompt fountain" in output.getvalue()


def test_stage_configurator_locks_confirmed_replacement_into_arguments(
    tmp_path: Path,
) -> None:
    scene = ScenePaths.from_root(tmp_path)
    console, _ = _console()
    prompter = ScriptedPrompter(
        selections=("standard-mask", "run"),
        texts=("statue",),
        confirmations=(True, True),
    )
    configurator = StageConfigurator(scene, prompter=prompter, console=console)

    resolved = configurator.configure(
        "sammer",
        status=Readiness(
            "sammer",
            "completed",
            "A complete mask generation already exists",
            required_action="force",
        ),
        command_builder=lambda options: ["sammer", *options.arguments],
    )

    assert resolved is not None
    assert resolved.arguments.count("--force") == 1


@pytest.mark.parametrize("state", ["completed", "ready"])
def test_existing_colmap_requires_resume_or_replace_policy(
    tmp_path: Path, state: str
) -> None:
    scene = ScenePaths.from_root(tmp_path)
    console, _ = _console()
    prompter = ScriptedPrompter(
        selections=("recommended", "resume", "run"),
        confirmations=(True,),
    )
    configurator = StageConfigurator(scene, prompter=prompter, console=console)

    resolved = configurator.configure(
        "colmap",
        status=Readiness(
            "colmap",
            state,
            "existing model" if state == "completed" else "incomplete artifacts",
            required_action="existing_policy",
        ),
        command_builder=lambda options: ["colmapper", *options.arguments],
    )

    assert resolved is not None
    existing_index = resolved.arguments.index("--existing")
    assert resolved.arguments[existing_index + 1] == "resume"


def test_bare_cancel_returns_without_creating_scene_metadata(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    scene.mkdir()
    console, _ = _console()

    arguments = prepare_stage_arguments(
        "sharpframes",
        [],
        program="sharpframes",
        scene_root=scene,
        stdin=TerminalBuffer(tty=True),
        stdout=TerminalBuffer(tty=True),
        prompter=ScriptedPrompter(selections=(None,)),
        console=console,
        artifact_lister=lambda _scene: [],
        inspector=lambda _scene, _artifacts: [
            Readiness("sharpframes", "ready", "Select a source")
        ],
        tool_inspector=lambda stage: Readiness(stage, "ready", "tool ready"),
    )

    assert arguments is None
    assert not (scene / ".gausser").exists()


@pytest.mark.parametrize(
    ("module_name", "stage", "program"),
    (
        ("sharpframes.cli", "sharpframes", "sharpframes"),
        ("sammer.cli", "sammer", "sammer"),
        ("colmapper.cli", "colmap", "colmapper"),
        ("brusher.cli", "brush", "brusher"),
        ("msplatter.cli", "msplat", "msplatter"),
        ("lodder.cli", "lod", "lodder"),
    ),
)
def test_every_stage_main_routes_zero_arguments_through_shared_preparation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    module_name: str,
    stage: str,
    program: str,
) -> None:
    module = importlib.import_module(module_name)
    calls: list[tuple[str, list[str] | None, str]] = []

    def fake_prepare(
        actual_stage: str,
        argv: list[str] | None,
        *,
        program: str,
    ) -> list[str]:
        calls.append((actual_stage, argv, program))
        return ["--help"]

    monkeypatch.setattr(module, "prepare_stage_arguments", fake_prepare)

    with pytest.raises(SystemExit) as exc_info:
        module.main([])

    assert exc_info.value.code == 0
    assert calls == [(stage, [], program)]
    assert "Run without arguments in a terminal for guided setup." in capsys.readouterr().out
