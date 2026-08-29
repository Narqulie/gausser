from __future__ import annotations

from collections import deque
from io import StringIO
from pathlib import Path
from typing import Sequence

from rich.console import Console

from gausser.artifacts import Artifact
from gausser.controller import SceneController, StageRequest
from gausser.launcher import Launcher, PromptChoice, PromptBackend, main
from gausser.preflight import Readiness
from gausser.process import CommandResult
from gausser.scene import ScenePaths


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


def test_launcher_runs_a_resolved_stage_and_refreshes_the_scene(scene: Path):
    output = StringIO()
    calls: list[list[str]] = []
    inspections = 0

    def runner(command: list[str], **kwargs) -> CommandResult:
        calls.append(command)
        kwargs["on_line"]("[red]literal child output[/red]")
        return CommandResult(tuple(command), kwargs["cwd"], 0, 1.25, (), False)

    def inspector(
        _paths: ScenePaths, _artifacts: Sequence[Artifact]
    ) -> list[Readiness]:
        nonlocal inspections
        inspections += 1
        return [
            Readiness("images", "ready", "1 image"),
            Readiness("sharpframes", "unavailable", "Images already exist"),
            Readiness("sammer", "ready", "SAM3 required"),
            Readiness("colmap", "ready", "No sparse model"),
            Readiness("brush", "unavailable", "COLMAP required"),
            Readiness("msplat", "unavailable", "COLMAP required"),
            Readiness("lod", "unavailable", "No PLY or SOG export"),
        ]

    prompter = ScriptedPrompter(
        selections=("sammer", "standard-mask", "run", "exit"),
        texts=("statue",),
        confirmations=(True,),
    )
    paths = ScenePaths.from_root(scene)
    launcher = Launcher(
        paths,
        prompter=prompter,
        console=Console(
            file=output, force_terminal=False, color_system=None, width=120
        ),
        controller=SceneController(paths, runner=runner),
        inspector=inspector,
        tool_inspector=ready_tool_inspector,
    )

    assert launcher.run() == 0
    assert calls == [
        [
            "sammer",
            "--scene",
            str(scene),
            "--prompt",
            "statue",
            "--score-threshold",
            "0.5",
            "--device",
            "auto",
            "--dilation",
            "0",
            "--feather",
            "0",
        ]
    ]
    assert inspections >= 2
    assert "◈ GAUSSER // SYNTHWAVE ’84 COMMAND DECK" in output.getvalue()
    assert "[red]literal child output[/red]" in output.getvalue()
    assert "Succeeded" in output.getvalue()


def test_launcher_dashboard_renders_markup_like_scene_data_literally(tmp_path: Path):
    scene = tmp_path / "[red]scene"
    scene.mkdir()
    output = StringIO()
    launcher = Launcher(
        ScenePaths.from_root(scene),
        prompter=ScriptedPrompter(),
        console=Console(
            file=output, force_terminal=False, color_system=None, width=120
        ),
        tool_inspector=ready_tool_inspector,
    )

    launcher._render_dashboard(
        {"sharpframes": Readiness("sharpframes", "ready", "[bold]literal[/bold]")},
        (),
    )

    rendered = output.getvalue()
    assert "[red]scene" in rendered
    assert "[bold]literal[/bold]" in rendered


def test_launcher_menu_lists_every_stage_and_disables_unavailable_steps(scene: Path):
    prompter = ScriptedPrompter(selections=("exit",))
    launcher = Launcher(
        ScenePaths.from_root(scene),
        prompter=prompter,
        tool_inspector=ready_tool_inspector,
    )

    assert launcher.run() == 0

    menu = prompter.select_calls[0][1]
    choices = {choice.value: choice for choice in menu}
    assert {"sharpframes", "sammer", "colmap", "brush", "msplat", "lod"} <= set(choices)
    assert choices["brush"].disabled is not None
    assert choices["sammer"].disabled is None
    assert {"chain:masked-brush", "chain:unmasked-msplat", "chain:brush-lod"} <= set(
        choices
    )


def test_launcher_lod_source_prompt_only_lists_scene_owned_ply_and_sog(scene: Path):
    paths = ScenePaths.from_root(scene)
    prompter = ScriptedPrompter(
        selections=(
            "lod",
            "web-streaming",
            "runs/brush/run-1/exports/model.ply",
            "run",
            "exit",
        ),
        confirmations=(True,),
    )
    controller = FakeController(paths, results=[successful_result(paths, "lodder")])
    artifacts = [
        Artifact("runs/brush/run-1/exports/model.ply", "brush", "run-1", "ply", 100),
        Artifact("runs/brush/run-1/exports/model.spz", "brush", "run-1", "spz", 80),
        Artifact("runs/lod/run-2/source.sog", "lod", "run-2", "sog", 60),
    ]
    launcher = Launcher(
        paths,
        prompter=prompter,
        controller=controller,
        inspector=ready_inspector,
        artifact_lister=lambda _paths: artifacts,
        tool_inspector=ready_tool_inspector,
    )

    assert launcher.run() == 0
    assert controller.requests[0].arguments[0] == "runs/brush/run-1/exports/model.ply"
    source_choices = next(
        choices
        for message, choices in prompter.select_calls
        if "source export" in message.lower()
    )
    assert {choice.value for choice in source_choices} == {
        "runs/brush/run-1/exports/model.ply",
    }


def test_masked_brush_chain_runs_each_real_stage_in_order(scene: Path):
    paths = ScenePaths.from_root(scene)
    controller = FakeController(
        paths,
        results=[
            successful_result(paths, "sammer"),
            successful_result(paths, "colmapper"),
            successful_result(paths, "brusher"),
        ],
    )
    prompter = ScriptedPrompter(
        selections=(
            "chain:masked-brush",
            "standard-mask",
            "run",
            "video-sequence",
            "run",
            "apple-silicon-safe",
            "run",
            "exit",
        ),
        texts=("statue",),
        confirmations=(True, True, True, True),
    )
    launcher = Launcher(
        paths,
        prompter=prompter,
        controller=controller,
        inspector=ready_inspector,
        tool_inspector=ready_tool_inspector,
    )

    assert launcher.run() == 0
    assert [request.stage for request in controller.requests] == [
        "sammer",
        "colmap",
        "brush",
    ]
    assert controller.readiness_calls >= 6


def test_msplat_declined_acknowledgement_cancels_without_launch(scene: Path):
    paths = ScenePaths.from_root(scene)
    controller = FakeController(paths, results=[])
    prompter = ScriptedPrompter(
        selections=("msplat", "local-memory-safe", "exit"),
        confirmations=(False,),
    )
    launcher = Launcher(
        paths,
        prompter=prompter,
        controller=controller,
        inspector=ready_inspector,
        tool_inspector=ready_tool_inspector,
    )

    assert launcher.run() == 0
    assert controller.requests == []


def test_unmasked_msplat_chain_runs_colmap_then_acknowledged_msplat(scene: Path):
    paths = ScenePaths.from_root(scene)
    controller = FakeController(
        paths,
        results=[
            successful_result(paths, "colmapper"),
            successful_result(paths, "msplatter"),
        ],
    )
    prompter = ScriptedPrompter(
        selections=(
            "chain:unmasked-msplat",
            "video-sequence",
            "run",
            "local-memory-safe",
            "run",
            "exit",
        ),
        confirmations=(True, True, True, True),
    )
    launcher = Launcher(
        paths,
        prompter=prompter,
        controller=controller,
        inspector=ready_inspector,
        tool_inspector=ready_tool_inspector,
    )

    assert launcher.run() == 0
    assert [request.stage for request in controller.requests] == ["colmap", "msplat"]
    assert controller.requests[1].arguments[-1] == "--allow-unmasked"


def test_brush_lod_chain_refreshes_artifacts_before_lod_source_selection(scene: Path):
    paths = ScenePaths.from_root(scene)
    controller = FakeController(
        paths,
        results=[
            successful_result(paths, "brusher"),
            successful_result(paths, "lodder"),
        ],
    )
    published = Artifact(
        "runs/brush/new-run/exports/model.ply", "brush", "new-run", "ply", 100
    )
    artifact_calls = 0

    def artifacts(_paths: ScenePaths) -> list[Artifact]:
        nonlocal artifact_calls
        artifact_calls += 1
        return [] if artifact_calls == 1 else [published]

    prompter = ScriptedPrompter(
        selections=(
            "chain:brush-lod",
            "apple-silicon-safe",
            "run",
            "web-streaming",
            published.relative_path,
            "run",
            "exit",
        ),
        confirmations=(True, True, True),
    )
    launcher = Launcher(
        paths,
        prompter=prompter,
        controller=controller,
        inspector=ready_inspector,
        artifact_lister=artifacts,
        tool_inspector=ready_tool_inspector,
    )

    assert launcher.run() == 0
    assert [request.stage for request in controller.requests] == ["brush", "lod"]
    assert controller.requests[1].arguments[0] == published.relative_path
    assert artifact_calls >= 3


def test_launcher_disables_a_scene_ready_stage_when_its_tool_is_unavailable(
    scene: Path,
):
    paths = ScenePaths.from_root(scene)
    controller = FakeController(paths, results=[])
    prompter = ScriptedPrompter(selections=("exit",))

    def tools() -> list[Readiness]:
        return [
            Readiness(
                stage,
                "unavailable" if stage == "sammer" else "ready",
                "SAM3 runtime missing",
            )
            for stage in ("sharpframes", "sammer", "colmap", "brush", "msplat", "lod")
        ]

    launcher = Launcher(
        paths,
        prompter=prompter,
        controller=controller,
        inspector=ready_inspector,
        tool_inspector=tools,
    )

    assert launcher.run() == 0
    choices = {choice.value: choice for choice in prompter.select_calls[0][1]}
    assert choices["sammer"].disabled == "SAM3 runtime missing"
    assert choices["colmap"].disabled is None


def test_launcher_requires_explicit_confirmation_to_replace_complete_masks(scene: Path):
    paths = ScenePaths.from_root(scene)
    controller = FakeController(paths, results=[successful_result(paths, "sammer")])
    prompter = ScriptedPrompter(
        selections=("sammer", "standard-mask", "run", "exit"),
        texts=("statue",),
        confirmations=(True, True),
    )

    def inspector(
        _paths: ScenePaths, _artifacts: Sequence[Artifact]
    ) -> list[Readiness]:
        return [
            Readiness(
                stage,
                "completed" if stage == "sammer" else "ready",
                "Masks complete",
                required_action="force" if stage == "sammer" else "none",
            )
            for stage in (
                "images",
                "sharpframes",
                "sammer",
                "colmap",
                "brush",
                "msplat",
                "lod",
            )
        ]

    launcher = Launcher(
        paths,
        prompter=prompter,
        controller=controller,
        inspector=inspector,
        tool_inspector=ready_tool_inspector,
    )

    assert launcher.run() == 0
    assert controller.requests[0].arguments[-1] == "--force"


def test_launcher_requires_explicit_confirmation_to_replace_images(scene: Path):
    paths = ScenePaths.from_root(scene)
    source = scene.parent / "source.mp4"
    source.write_bytes(b"video")
    controller = FakeController(
        paths, results=[successful_result(paths, "sharpframes")]
    )
    prompter = ScriptedPrompter(
        selections=("sharpframes", "upstream-defaults", "run", "exit"),
        texts=(str(source),),
        confirmations=(True, True),
    )

    def inspector(
        _paths: ScenePaths, _artifacts: Sequence[Artifact]
    ) -> list[Readiness]:
        return [
            Readiness(
                stage,
                "completed" if stage == "sharpframes" else "ready",
                "Images exist",
                required_action="force" if stage == "sharpframes" else "none",
            )
            for stage in (
                "images",
                "sharpframes",
                "sammer",
                "colmap",
                "brush",
                "msplat",
                "lod",
            )
        ]

    launcher = Launcher(
        paths,
        prompter=prompter,
        controller=controller,
        inspector=inspector,
        tool_inspector=ready_tool_inspector,
    )

    assert launcher.run() == 0
    assert "--force" in controller.requests[0].arguments


def test_launcher_disables_a_chain_when_any_required_tool_is_missing(scene: Path):
    paths = ScenePaths.from_root(scene)
    controller = FakeController(paths, results=[])
    prompter = ScriptedPrompter(selections=("exit",))

    def tools() -> list[Readiness]:
        return [
            Readiness(
                stage,
                "unavailable" if stage == "brush" else "ready",
                "Brush executable missing",
            )
            for stage in ("sharpframes", "sammer", "colmap", "brush", "msplat", "lod")
        ]

    launcher = Launcher(
        paths,
        prompter=prompter,
        controller=controller,
        inspector=ready_inspector,
        tool_inspector=tools,
    )

    assert launcher.run() == 0
    choices = {choice.value: choice for choice in prompter.select_calls[0][1]}
    assert (
        choices["chain:masked-brush"].disabled
        == "Brush Training: Brush executable missing"
    )
    assert choices["chain:unmasked-msplat"].disabled is None


def test_masked_brush_chain_stops_immediately_after_a_failed_stage(scene: Path):
    paths = ScenePaths.from_root(scene)
    failed = CommandResult(("sammer",), paths.root, 1, 0.5, ("failure",), False)
    controller = FakeController(paths, results=[failed])
    prompter = ScriptedPrompter(
        selections=("chain:masked-brush", "standard-mask", "run", "exit"),
        texts=("statue",),
        confirmations=(True, True),
    )
    output = StringIO()
    launcher = Launcher(
        paths,
        prompter=prompter,
        console=Console(
            file=output, force_terminal=False, color_system=None, width=120
        ),
        controller=controller,
        inspector=ready_inspector,
        tool_inspector=ready_tool_inspector,
    )

    assert launcher.run() == 0
    assert [request.stage for request in controller.requests] == ["sammer"]
    assert "Stopped" in output.getvalue()
    assert "colmap" in output.getvalue()


def test_launcher_rejects_a_missing_scene_without_creating_metadata(tmp_path: Path):
    missing = tmp_path / "missing"

    assert main([str(missing)], require_tty=False) == 2
    assert not missing.exists()


class FakeController:
    def __init__(self, paths: ScenePaths, *, results: Sequence[CommandResult]) -> None:
        self.scene = paths
        self.results = deque(results)
        self.requests: list[StageRequest] = []
        self.readiness_calls = 0

    def readiness(self) -> dict[str, str]:
        self.readiness_calls += 1
        return {
            stage: "ready"
            for stage in ("sharpframes", "sammer", "colmap", "brush", "msplat", "lod")
        }

    def command_for(self, request: StageRequest) -> list[str]:
        executable = {
            "sharpframes": "sharpframes",
            "sammer": "sammer",
            "colmap": "colmapper",
            "brush": "brusher",
            "msplat": "msplatter",
            "lod": "lodder",
        }[request.stage]
        return [executable, *request.arguments]

    def run(self, request: StageRequest, **kwargs) -> CommandResult:
        self.requests.append(request)
        on_line = kwargs.get("on_line")
        if on_line is not None:
            on_line("fake output")
        return self.results.popleft()


def ready_inspector(
    _paths: ScenePaths, _artifacts: Sequence[Artifact]
) -> list[Readiness]:
    return [
        Readiness(stage, "ready", "Ready")
        for stage in (
            "images",
            "sharpframes",
            "sammer",
            "colmap",
            "brush",
            "msplat",
            "lod",
        )
    ]


def ready_tool_inspector() -> list[Readiness]:
    return [
        Readiness(stage, "ready", "Available")
        for stage in ("sharpframes", "sammer", "colmap", "brush", "msplat", "lod")
    ]


def successful_result(paths: ScenePaths, executable: str) -> CommandResult:
    return CommandResult((executable,), paths.root, 0, 0.5, (), False)
