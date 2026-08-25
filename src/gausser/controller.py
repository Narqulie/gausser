"""Stage orchestration boundary shared by Gausser's headless and Textual clients."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Callable, Literal

from .preflight import inspect_scene
from .process import CommandResult, LineCallback, run_command
from .scene import ScenePaths


Stage = Literal["sammer", "colmap", "brush"]
_STAGE_ORDER: tuple[Stage, ...] = ("sammer", "colmap", "brush")


class StageUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class StageRequest:
    stage: Stage
    arguments: tuple[str, ...] = ()


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
        if request.stage == "sammer":
            return ["sammer", "--scene", str(self.scene.root), *request.arguments]
        if request.stage == "colmap":
            return ["colmapper", "--scene", str(self.scene.root), *request.arguments]
        if request.stage == "brush":
            return ["brusher", str(self.scene.root), *request.arguments]
        raise ValueError(f"unknown stage: {request.stage}")

    def run(
        self,
        request: StageRequest,
        *,
        on_line: LineCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> CommandResult:
        state = self.readiness().get(request.stage)
        if state != "ready":
            raise StageUnavailable(f"{request.stage} is {state}; it cannot be started")
        return self._runner(
            self.command_for(request),
            cwd=self.scene.root,
            on_line=on_line,
            cancel_event=cancel_event,
        )

    def run_pipeline(
        self,
        requests: dict[Stage, StageRequest],
        *,
        on_line: LineCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[CommandResult]:
        results: list[CommandResult] = []
        for stage in _STAGE_ORDER:
            result = self.run(requests[stage], on_line=on_line, cancel_event=cancel_event)
            results.append(result)
            if not result.succeeded:
                break
        return results
