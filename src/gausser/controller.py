"""Stage orchestration boundary used by the optional Gausser Studio engine."""

from __future__ import annotations

from dataclasses import dataclass
import json
import threading
from typing import Any, Callable, Literal, Mapping

from .preflight import inspect_scene
from .process import CommandResult, LineCallback, run_command
from .scene import SceneError, ScenePaths
from .scene_lock import SCENE_LOCK_FD_ENV
from .stage_events import StageEvent, StageTelemetry


Stage = Literal["sharpframes", "sammer", "colmap", "brush", "msplat", "lod"]
StageEventCallback = Callable[[StageEvent], None]
_MSPLAT_INTERRUPT_GRACE_SECONDS = 300.0


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
            return ["msplatter", str(self.scene.root), *request.arguments]
        if request.stage == "lod":
            return ["lodder", "--scene", str(self.scene.root), *request.arguments]
        raise ValueError(f"unknown stage: {request.stage}")

    def run(
        self,
        request: StageRequest,
        *,
        on_line: LineCallback | None = None,
        on_event: StageEventCallback | None = None,
        cancel_event: threading.Event | None = None,
        scene_lock_fd: int | None = None,
    ) -> CommandResult:
        state = self.readiness().get(request.stage)
        if state not in {"ready", "completed"}:
            raise StageUnavailable(f"{request.stage} is {state}; it cannot be started")
        try:
            total_images = len(self.scene.image_files())
        except SceneError:
            total_images = 0
        telemetry = StageTelemetry.for_stage(
            request.stage,
            request.arguments,
            total_images=total_images,
        )

        def handle_line(line: str) -> None:
            if on_line is not None:
                on_line(line)
            event = telemetry.from_line(line)
            if event is not None and on_event is not None:
                on_event(event)

        inherited_fds = (scene_lock_fd,) if scene_lock_fd is not None else ()
        env = {SCENE_LOCK_FD_ENV: str(scene_lock_fd)} if scene_lock_fd is not None else None
        return self._runner(
            self.command_for(request),
            cwd=self.scene.root,
            on_line=handle_line,
            cancel_event=cancel_event,
            env=env,
            inherited_fds=inherited_fds,
            interrupt_grace_seconds=(
                _MSPLAT_INTERRUPT_GRACE_SECONDS if request.stage == "msplat" else 2.0
            ),
        )
