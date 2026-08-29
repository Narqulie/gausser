"""Local JSON-lines engine for native Gausser clients.

The engine is deliberately a thin service boundary around :class:`SceneController`.
It owns active-run identity and event serialization while the existing controller
continues to own stage commands and the shared process runner owns child lifecycles.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import argparse
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, TextIO, cast
from uuid import uuid4

from .artifacts import list_artifacts
from .colmap_model import read_reconstruction_preview
from .controller import SceneController, Stage, StageRequest
from .dataset import DatasetChangedError, list_dataset_images
from .ingest import ingest_photo_directory, mutate_photo_dataset
from .manifest import record_run_finished, record_run_started
from .preflight import (
    Readiness,
    inspect_launch_tool,
    inspect_scene,
    inspect_tools,
    validate_stage_readiness,
)
from .presentation import (
    PresentationArtifactChangedError,
    PresentationChangedError,
    PresentationValidationError,
    load_presentation,
    save_presentation,
)
from .process import CommandResult
from .scene import SceneError, ScenePaths, initialize_scene
from .scene_lock import SceneLock, SceneLockError
from .stage_options import StageOptionError, catalog, resolve_options
from .supersplat import reimport_edited_export


EventSink = Callable[[dict[str, Any]], None]
ControllerFactory = Callable[[ScenePaths], SceneController]
ToolInspector = Callable[[Stage], Readiness]
_STAGE_NAMES: frozenset[str] = frozenset({"sharpframes", "sammer", "colmap", "brush", "msplat", "lod"})
_STAGE_TITLES = {
    "sharpframes": "Sharp Frames",
    "sammer": "Sammer",
    "colmap": "COLMAP",
    "brush": "Brusher",
    "msplat": "Msplatter",
    "lod": "LOD Builder",
}


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
    settings: dict[str, Any] | None
    cancel_event: threading.Event
    lock: SceneLock
    started_monotonic: float = field(default_factory=time.monotonic)
    last_observed_monotonic: float | None = None
    last_phase: str | None = None
    thread: threading.Thread | None = None

    def descriptor(self) -> dict[str, Any]:
        elapsed = max(0.0, time.monotonic() - self.started_monotonic)
        return {
            "id": self.id,
            "scene": str(self.scene.root),
            "stage": self.stage,
            "elapsed_seconds": elapsed,
            "last_observed_elapsed_seconds": (
                max(0.0, self.last_observed_monotonic - self.started_monotonic)
                if self.last_observed_monotonic is not None else None
            ),
            "phase": self.last_phase,
        }


class GausserEngine:
    """Dispatch JSON-compatible requests and stream structured run events."""

    def __init__(
        self,
        *,
        controller_factory: ControllerFactory = SceneController,
        tool_inspector: ToolInspector | None = None,
    ) -> None:
        self._controller_factory = controller_factory
        self._tool_inspector = tool_inspector or inspect_launch_tool
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
            elif method == "stage_catalog":
                self._scene_from_params(params)
                emit({"id": request_id, "result": {"stages": catalog()}})
            elif method == "resolve_stage":
                scene, stage, resolved = self._resolve_stage_options(params)
                controller = self._controller_factory(scene)
                request = StageRequest(stage, resolved.arguments, resolved.as_settings())
                emit({"id": request_id, "result": {
                    "stage": stage,
                    "specification_version": resolved.specification_version,
                    "preset": resolved.preset,
                    "values": resolved.values,
                    "arguments": list(resolved.arguments),
                    "command": controller.command_for(request),
                }})
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
            elif method == "dataset_images":
                scene = self._scene_from_params(params)
                revision = params.get("revision")
                if revision is not None and (not isinstance(revision, str) or not revision):
                    raise RequestError("revision must be a non-empty string when provided")
                page = list_dataset_images(
                    scene,
                    offset=self._integer(params, "offset", default=0, minimum=0),
                    limit=self._integer(params, "limit", default=200, minimum=1, maximum=500),
                    revision=revision,
                )
                emit({"id": request_id, "result": {
                    "images": [{key: value for key, value in asdict(image).items() if value is not None} for image in page.images],
                    "image_count": page.total_count,
                    "next_offset": page.next_offset,
                    "revision": page.revision,
                }})
            elif method == "reconstruction":
                scene = self._scene_from_params(params)
                preview = read_reconstruction_preview(scene.sparse / "0")
                if preview is None:
                    emit({"id": request_id, "result": {
                        "cameras": [], "points": [], "camera_count": 0, "point_count": 0, "focus": None,
                    }})
                else:
                    emit({"id": request_id, "result": {
                        # Optical directions are used only by the engine to derive
                        # the view pivot; the native canvas needs camera centres.
                        "cameras": [
                            {"name": camera.name, "x": camera.x, "y": camera.y, "z": camera.z}
                            for camera in preview.cameras
                        ],
                        "points": [asdict(point) for point in preview.points],
                        "camera_count": preview.camera_count,
                        "point_count": preview.point_count,
                        "focus": (
                            {"x": preview.focus[0], "y": preview.focus[1], "z": preview.focus[2]}
                            if preview.focus is not None else None
                        ),
                    }})
            elif method == "presentation":
                scene = self._scene_from_params(params)
                artifact = self._required_string(params, "artifact")
                emit({
                    "id": request_id,
                    "result": {"presentation": load_presentation(scene, artifact)},
                })
            elif method == "update_presentation":
                scene = self._scene_from_params(params)
                presentation = params.get("presentation")
                if not isinstance(presentation, dict):
                    raise RequestError("presentation must be an object")
                with self._scene_mutation(scene):
                    saved_presentation = save_presentation(scene, presentation)
                emit({
                    "id": request_id,
                    "result": {"presentation": saved_presentation},
                })
            elif method == "ingest_photos":
                scene = self._scene_from_params(params)
                with self._scene_mutation(scene):
                    source = self._required_string(params, "source")
                    policy = self._required_string(params, "policy")
                    ingest_result = ingest_photo_directory(scene, source, policy=policy)
                emit({
                    "id": request_id,
                    "result": {
                        "source": str(ingest_result.source),
                        "policy": ingest_result.policy.value,
                        "image_count": len(ingest_result.images),
                    },
                })
            elif method == "mutate_dataset":
                scene = self._scene_from_params(params)
                additions = self._string_array(params, "additions")
                removals = self._string_array(params, "removals")
                policy = self._required_string(params, "policy")
                expected_revision = self._required_string(params, "revision")
                with self._scene_mutation(scene):
                    current_revision = list_dataset_images(scene, offset=0, limit=1).revision
                    if expected_revision != current_revision:
                        raise DatasetChangedError("dataset changed; reload it before editing")
                    mutation_result = mutate_photo_dataset(scene, additions=additions, removals=removals, policy=policy)
                emit({
                    "id": request_id,
                    "result": {
                        "added": list(mutation_result.added),
                        "removed": list(mutation_result.removed),
                        "archive": str(scene.relative_path(mutation_result.archive)),
                    },
                })
            elif method == "reimport_supersplat":
                scene = self._scene_from_params(params)
                with self._scene_mutation(scene):
                    source = self._required_string(params, "source")
                    edited = self._required_string(params, "edited")
                    reimport_result = reimport_edited_export(scene, source, edited)
                emit({
                    "id": request_id,
                    "result": {
                        "run_id": reimport_result.run_id,
                        "export": str(scene.relative_path(reimport_result.export)),
                        "provenance": str(scene.relative_path(reimport_result.provenance)),
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
                    try:
                        record_run_finished(active.scene, active.id, "failed")
                    finally:
                        active.lock.release()
                    raise
            elif method == "cancel":
                run_id = self._required_string(params, "run_id")
                emit({"id": request_id, "result": self._cancel(run_id)})
            else:
                raise RequestError(f"unsupported method: {method}")
        except DatasetChangedError as exc:
            emit({"id": request_id, "error": {"code": "dataset_changed", "message": str(exc)}})
        except PresentationChangedError as exc:
            emit({"id": request_id, "error": {"code": "presentation_changed", "message": str(exc)}})
        except PresentationArtifactChangedError as exc:
            emit({"id": request_id, "error": {"code": "presentation_artifact_changed", "message": str(exc)}})
        except PresentationValidationError as exc:
            emit({"id": request_id, "error": {"code": "invalid_presentation", "message": str(exc)}})
        except SceneLockError as exc:
            emit({"id": request_id, "error": {"code": "scene_busy", "message": str(exc)}})
        except SceneError as exc:
            code = "scene_initialization_failed" if method == "initialize" else "scene_error"
            emit({"id": request_id, "error": {"code": code, "message": str(exc)}})
        except (RequestError, StageOptionError) as exc:
            code = exc.code if isinstance(exc, RequestError) else "invalid_options"
            emit({"id": request_id, "error": {"code": code, "message": str(exc)}})
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
    def _string_array(params: dict[str, object], name: str) -> list[str]:
        value = params.get(name, [])
        if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
            raise RequestError(f"{name} must be an array of non-empty strings")
        return value

    @staticmethod
    def _integer(
        params: dict[str, object],
        name: str,
        *,
        default: int,
        minimum: int,
        maximum: int | None = None,
    ) -> int:
        value = params.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
            upper = f" and at most {maximum}" if maximum is not None else ""
            raise RequestError(f"{name} must be an integer at least {minimum}{upper}")
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
        if "arguments" in params:
            raise RequestError("start accepts typed options only; raw arguments are not supported")
        raw_options = params.get("options")
        if not isinstance(raw_options, dict):
            raise RequestError("options must be an object")
        resolved = resolve_options(stage_name, raw_options, path_base=scene.root)
        stage = cast(Stage, stage_name)
        raw_arguments = list(resolved.arguments)
        settings = resolved.as_settings()

        self._require_no_local_run(scene)
        scene_lock = SceneLock(scene.root)
        scene_lock.acquire()
        try:
            tool_status = self._tool_inspector(stage)
            if tool_status.state == "unavailable":
                raise RequestError(
                    f"{_STAGE_TITLES[stage]} is unavailable: {tool_status.detail}"
                )
            try:
                validate_stage_readiness(scene, stage, resolved.values)
            except ValueError as exc:
                raise RequestError(str(exc)) from exc
            active = ActiveRun(
                id=f"run-{uuid4().hex}",
                scene=scene,
                stage=stage,
                arguments=tuple(raw_arguments),
                settings=settings,
                cancel_event=threading.Event(),
                lock=scene_lock,
            )
            record_run_started(scene, active.id, active.stage, active.settings)
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
        emit({"event": "started", "run": active.descriptor(), "elapsed_seconds": 0.0})

        def emit_line(line: str) -> None:
            emit({
                "event": "log",
                "run_id": active.id,
                "line": line,
                "elapsed_seconds": self._elapsed(active),
            })

        def emit_stage_event(stage_event) -> None:
            observed = time.monotonic()
            active.last_observed_monotonic = observed
            phase = stage_event.data.get("phase")
            if isinstance(phase, str):
                active.last_phase = phase
            emit({
                "event": stage_event.kind,
                "run_id": active.id,
                "elapsed_seconds": max(0.0, observed - active.started_monotonic),
                "observed_elapsed_seconds": max(0.0, observed - active.started_monotonic),
                **stage_event.data,
            })

        state = "failed"
        result_data: dict[str, Any] | None = None
        try:
            controller = self._controller_factory(active.scene)
            result = controller.run(
                StageRequest(active.stage, active.arguments, active.settings),
                on_line=emit_line,
                on_event=emit_stage_event,
                cancel_event=active.cancel_event,
                scene_lock_fd=active.lock.fileno(),
            )
            state = "succeeded" if result.succeeded else "cancelled" if result.cancelled else "failed"
            result_data = self._result_data(result)
            event: dict[str, Any] = {
                "event": "finished",
                "run": {**active.descriptor(), "state": state, "result": result_data},
            }
        except Exception as exc:
            event = {
                "event": "finished",
                "run": {**active.descriptor(), "state": "failed", "error": {"code": "stage_failed", "message": str(exc)}},
            }
        finally:
            try:
                record_run_finished(active.scene, active.id, state, result_data)
            except Exception as exc:
                event = {
                    "event": "finished",
                    "run": {**active.descriptor(), "state": "failed", "error": {"code": "manifest_failed", "message": str(exc)}},
                }
            with self._lock:
                self._active_by_id.pop(active.id, None)
                self._active_by_scene.pop(active.scene.root, None)
            active.lock.release()
        emit(event)

    def _resolve_stage_options(self, params: dict[str, object]):
        scene = self._scene_from_params(params)
        stage_name = self._required_string(params, "stage")
        if stage_name not in _STAGE_NAMES:
            raise RequestError(f"unsupported stage: {stage_name}")
        raw_options = params.get("options")
        if not isinstance(raw_options, dict):
            raise RequestError("options must be an object")
        return scene, stage_name, resolve_options(
            stage_name, raw_options, path_base=scene.root
        )

    @staticmethod
    def _elapsed(active: ActiveRun) -> float:
        return max(0.0, time.monotonic() - active.started_monotonic)

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

    def _active_descriptors(self) -> list[dict[str, Any]]:
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
