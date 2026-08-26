"""Local JSON-lines engine for native Gausser clients.

The engine is deliberately a thin service boundary around :class:`SceneController`.
It owns active-run identity and event serialization while the existing controller
continues to own stage commands and the shared process runner owns child lifecycles.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
import sys
import threading
from typing import Any, Callable, TextIO
from uuid import uuid4

from .artifacts import list_artifacts
from .controller import SceneController, Stage, StageRequest
from .ingest import ingest_photo_directory
from .preflight import inspect_scene, inspect_tools
from .process import CommandResult
from .scene import SceneError, ScenePaths, initialize_scene
from .scene_lock import SceneLock, SceneLockError
from .supersplat import reimport_edited_export


EventSink = Callable[[dict[str, Any]], None]
ControllerFactory = Callable[[ScenePaths], SceneController]
_STAGE_NAMES: frozenset[str] = frozenset({"sharpframes", "sammer", "colmap", "brush", "msplat", "lod"})


class RequestError(ValueError):
    """An invalid client request that is safe to report without stopping the engine."""

    code = "invalid_request"


class SceneBusyError(RequestError):
    """A valid start request that conflicts with an active scene run."""

    code = "scene_busy"


@dataclass
class ActiveRun:
    id: str
    scene: ScenePaths
    stage: Stage
    arguments: tuple[str, ...]
    cancel_event: threading.Event
    lock: SceneLock
    thread: threading.Thread | None = None

    def descriptor(self) -> dict[str, str]:
        return {"id": self.id, "scene": str(self.scene.root), "stage": self.stage}


class GausserEngine:
    """Dispatch JSON-compatible requests and stream structured run events."""

    def __init__(self, *, controller_factory: ControllerFactory = SceneController) -> None:
        self._controller_factory = controller_factory
        self._active_by_id: dict[str, ActiveRun] = {}
        self._active_by_scene: dict[Path, ActiveRun] = {}
        self._lock = threading.Lock()

    def dispatch(self, request: object, emit: EventSink) -> None:
        """Handle one request without allowing malformed input to stop the engine."""
        request_id: object = None
        method: str | None = None
        try:
            request_id, method, params = self._validate_envelope(request)
            if method == "preflight":
                scene = self._scene_from_params(params)
                emit({"id": request_id, "result": self._preflight(scene)})
            elif method == "status":
                self._require_empty_params(params)
                emit({"id": request_id, "result": {"active_runs": self._active_descriptors()}})
            elif method == "initialize":
                scene = self._scene_from_params(params)
                with self._scene_mutation(scene):
                    initialized = initialize_scene(scene.root)
                emit({
                    "id": request_id,
                    "result": {"scene": str(initialized.root), "manifest": str(initialized.manifest)},
                })
            elif method == "artifacts":
                scene = self._scene_from_params(params)
                emit({"id": request_id, "result": {"artifacts": [asdict(item) for item in list_artifacts(scene)]}})
            elif method == "ingest_photos":
                scene = self._scene_from_params(params)
                with self._scene_mutation(scene):
                    source = self._required_string(params, "source")
                    policy = self._required_string(params, "policy")
                    result = ingest_photo_directory(scene, source, policy=policy)
                emit({
                    "id": request_id,
                    "result": {
                        "source": str(result.source),
                        "policy": result.policy.value,
                        "image_count": len(result.images),
                    },
                })
            elif method == "reimport_supersplat":
                scene = self._scene_from_params(params)
                with self._scene_mutation(scene):
                    source = self._required_string(params, "source")
                    edited = self._required_string(params, "edited")
                    result = reimport_edited_export(scene, source, edited)
                emit({
                    "id": request_id,
                    "result": {
                        "run_id": result.run_id,
                        "export": str(scene.relative_path(result.export)),
                        "provenance": str(scene.relative_path(result.provenance)),
                    },
                })
            elif method == "start":
                active = self._reserve_run(params)
                # Acknowledge before launching the thread so clients always observe
                # a run id before any event for that run.
                emit({"id": request_id, "result": {"run_id": active.id}})
                try:
                    self._launch(active, emit)
                except Exception:
                    with self._lock:
                        self._active_by_id.pop(active.id, None)
                        self._active_by_scene.pop(active.scene.root, None)
                    active.lock.release()
                    raise
            elif method == "cancel":
                run_id = self._required_string(params, "run_id")
                emit({"id": request_id, "result": self._cancel(run_id)})
            else:
                raise RequestError(f"unsupported method: {method}")
        except SceneLockError as exc:
            emit({"id": request_id, "error": {"code": "scene_busy", "message": str(exc)}})
        except SceneError as exc:
            code = "scene_initialization_failed" if method == "initialize" else "scene_error"
            emit({"id": request_id, "error": {"code": code, "message": str(exc)}})
        except RequestError as exc:
            emit({"id": request_id, "error": {"code": exc.code, "message": str(exc)}})
        except Exception as exc:  # Defensive boundary: a client request never kills the service.
            emit({"id": request_id, "error": {"code": "internal_error", "message": str(exc)}})

    def shutdown(self) -> None:
        """Cancel and join every active stage before the engine exits."""
        with self._lock:
            active = tuple(self._active_by_id.values())
        for run in active:
            run.cancel_event.set()
        for run in active:
            if run.thread:
                run.thread.join()

    @staticmethod
    def _validate_envelope(request: object) -> tuple[str, str, dict[str, object]]:
        if not isinstance(request, dict):
            raise RequestError("request must be an object")
        request_id = request.get("id")
        if not isinstance(request_id, str) or not request_id:
            raise RequestError("id must be a non-empty string")
        method = request.get("method")
        if not isinstance(method, str) or not method:
            raise RequestError("method must be a non-empty string")
        params = request.get("params", {})
        if not isinstance(params, dict):
            raise RequestError("params must be an object")
        return request_id, method, params

    @staticmethod
    def _required_string(params: dict[str, object], name: str) -> str:
        value = params.get(name)
        if not isinstance(value, str) or not value:
            raise RequestError(f"{name} must be a non-empty string")
        return value

    @staticmethod
    def _require_empty_params(params: dict[str, object]) -> None:
        if params:
            raise RequestError("status does not accept parameters")

    def _scene_from_params(self, params: dict[str, object]) -> ScenePaths:
        scene = Path(self._required_string(params, "scene")).expanduser().resolve()
        if not scene.is_dir():
            raise RequestError(f"scene does not exist: {scene}")
        return ScenePaths.from_root(scene)

    def _require_no_local_run(self, scene: ScenePaths) -> None:
        with self._lock:
            if scene.root in self._active_by_scene:
                raise SceneBusyError(f"a run is already active for {scene.root}")

    @contextmanager
    def _scene_mutation(self, scene: ScenePaths):
        """Exclude both local and other-engine mutations for a short operation."""
        self._require_no_local_run(scene)
        lock = SceneLock(scene.root)
        lock.acquire()
        try:
            self._require_no_local_run(scene)
            yield
        finally:
            lock.release()

    def _reserve_run(self, params: dict[str, object]) -> ActiveRun:
        scene = self._scene_from_params(params)
        stage_name = self._required_string(params, "stage")
        if stage_name not in _STAGE_NAMES:
            raise RequestError(f"unsupported stage: {stage_name}")
        raw_arguments = params.get("arguments", [])
        if not isinstance(raw_arguments, list) or any(not isinstance(item, str) or "\0" in item for item in raw_arguments):
            raise RequestError("arguments must be an array of strings without NUL bytes")

        self._require_no_local_run(scene)
        scene_lock = SceneLock(scene.root)
        scene_lock.acquire()
        try:
            active = ActiveRun(
                id=f"run-{uuid4().hex}",
                scene=scene,
                stage=stage_name,  # type: ignore[arg-type]
                arguments=tuple(raw_arguments),
                cancel_event=threading.Event(),
                lock=scene_lock,
            )
            with self._lock:
                if scene.root in self._active_by_scene:
                    raise SceneBusyError(f"a run is already active for {scene.root}")
                self._active_by_id[active.id] = active
                self._active_by_scene[scene.root] = active
            return active
        except Exception:
            scene_lock.release()
            raise

    def _launch(self, active: ActiveRun, emit: EventSink) -> None:
        thread = threading.Thread(
            target=self._execute,
            args=(active, emit),
            name=f"gausser-engine-{active.id}",
            daemon=False,
        )
        active.thread = thread
        thread.start()

    def _execute(self, active: ActiveRun, emit: EventSink) -> None:
        emit({"event": "started", "run": active.descriptor()})
        try:
            controller = self._controller_factory(active.scene)
            result = controller.run(
                StageRequest(active.stage, active.arguments),
                on_line=lambda line: emit({"event": "log", "run_id": active.id, "line": line}),
                cancel_event=active.cancel_event,
                inherited_fds=(active.lock.fileno(),),
            )
            state = "succeeded" if result.succeeded else "cancelled" if result.cancelled else "failed"
            event: dict[str, Any] = {
                "event": "finished",
                "run": {**active.descriptor(), "state": state, "result": self._result_data(result)},
            }
        except Exception as exc:
            event = {
                "event": "finished",
                "run": {**active.descriptor(), "state": "failed", "error": {"code": "stage_failed", "message": str(exc)}},
            }
        finally:
            with self._lock:
                self._active_by_id.pop(active.id, None)
                self._active_by_scene.pop(active.scene.root, None)
            active.lock.release()
        emit(event)

    @staticmethod
    def _result_data(result: CommandResult) -> dict[str, Any]:
        return {
            "command": list(result.command),
            "cwd": str(result.cwd),
            "returncode": result.returncode,
            "duration_seconds": result.duration_seconds,
            "output_tail": list(result.output_tail),
            "cancelled": result.cancelled,
        }

    def _cancel(self, run_id: str) -> dict[str, object]:
        with self._lock:
            active = self._active_by_id.get(run_id)
        if active is None:
            raise RequestError(f"unknown active run: {run_id}")
        active.cancel_event.set()
        return {"run_id": run_id, "cancellation_requested": True}

    def _active_descriptors(self) -> list[dict[str, str]]:
        with self._lock:
            return [active.descriptor() for active in self._active_by_id.values()]

    @staticmethod
    def _preflight(scene: ScenePaths) -> dict[str, object]:
        return {
            "scene": str(scene.root),
            "scene_readiness": [item.__dict__ for item in inspect_scene(scene)],
            "tool_readiness": [item.__dict__ for item in inspect_tools()],
        }


def run_stdio(input_stream: TextIO, output_stream: TextIO) -> None:
    """Serve one JSON-lines client until EOF, cancelling work during shutdown."""
    engine = GausserEngine()
    write_lock = threading.Lock()

    def emit(message: dict[str, Any]) -> None:
        with write_lock:
            output_stream.write(json.dumps(message, separators=(",", ":")) + "\n")
            output_stream.flush()

    try:
        for raw_line in input_stream:
            try:
                request = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                emit({"id": None, "error": {"code": "invalid_json", "message": str(exc)}})
                continue
            engine.dispatch(request, emit)
    finally:
        engine.shutdown()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gausser-engine", description="Serve Gausser's local JSON-lines engine.")
    parser.parse_args(argv)
    run_stdio(sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
