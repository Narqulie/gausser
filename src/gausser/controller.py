"""Stage orchestration boundary shared by Gausser's headless and Textual clients."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
from typing import Any, Callable, Literal, Mapping

from .preflight import inspect_scene
from .process import CommandResult, LineCallback, run_command
from .scene import ScenePaths


Stage = Literal["sharpframes", "sammer", "colmap", "brush", "msplat", "lod"]
Trainer = Literal["brush", "msplat"]
_PIPELINE_STAGES: tuple[Stage, Stage] = ("sammer", "colmap")


class StageUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class StageRequest:
    stage: Stage
    arguments: tuple[str, ...] = ()
    settings: Mapping[str, Any] | None = None


class SceneController:
    """Runs stage CLIs; presentation code only consumes its events/results."""

    def __init__(
        self,
        scene: ScenePaths,
        *,
        runner: Callable[..., CommandResult] = run_command,
    ) -> None:
        self.scene = scene
        self._runner = runner

    def readiness(self) -> dict[str, str]:
        return {item.stage: item.state for item in inspect_scene(self.scene)}

    def command_for(self, request: StageRequest) -> list[str]:
        if request.stage == "sharpframes":
            command = ["sharpframes", *request.arguments, "--scene", str(self.scene.root)]
            if request.settings is not None:
                command.extend(["--studio-settings", json.dumps(request.settings, sort_keys=True)])
            return command
        if request.stage == "sammer":
            return ["sammer", "--scene", str(self.scene.root), *request.arguments]
        if request.stage == "colmap":
            return ["colmapper", "--scene", str(self.scene.root), *request.arguments]
        if request.stage == "brush":
            return ["brusher", str(self.scene.root), *request.arguments]
        if request.stage == "msplat":
            return ["msplatter", str(self.scene.root), "--allow-unmasked", *request.arguments]
        if request.stage == "lod":
            return ["lodder", "--scene", str(self.scene.root), *request.arguments]
        raise ValueError(f"unknown stage: {request.stage}")

    def run(
        self,
        request: StageRequest,
        *,
        on_line: LineCallback | None = None,
        cancel_event: threading.Event | None = None,
        inherited_fds: tuple[int, ...] = (),
    ) -> CommandResult:
        state = self.readiness().get(request.stage)
        if state != "ready":
            raise StageUnavailable(f"{request.stage} is {state}; it cannot be started")
        return self._runner(
            self.command_for(request),
            cwd=self.scene.root,
            on_line=on_line,
            cancel_event=cancel_event,
            inherited_fds=inherited_fds,
        )

    def run_pipeline(
        self,
        requests: dict[Stage, StageRequest],
        *,
        trainer: Trainer = "brush",
        on_line: LineCallback | None = None,
        cancel_event: threading.Event | None = None,
        inherited_fds: tuple[int, ...] = (),
    ) -> list[CommandResult]:
        """Run SAM3, COLMAP, then the explicitly selected trainer."""
        results: list[CommandResult] = []
        for stage in (*_PIPELINE_STAGES, trainer):
            result = self.run(requests[stage], on_line=on_line, cancel_event=cancel_event, inherited_fds=inherited_fds)
            results.append(result)
            if not result.succeeded:
                break
        return results
