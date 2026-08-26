"""Optional Textual control surface for a Gausser scene."""

from __future__ import annotations

import argparse
from pathlib import Path
import threading

from .controller import SceneController, StageRequest, StageUnavailable
from .preflight import inspect_scene, inspect_tools
from .scene import ScenePaths


def create_app(scene: ScenePaths):
    """Construct the app without entering terminal mode; used by pilot tests."""
    try:
        from textual import work
        from textual.app import App, ComposeResult
        from textual.containers import Container, Horizontal
        from textual.widgets import Button, Footer, Header, Input, RichLog, Static
    except ImportError as exc:
        raise SystemExit("Install the TUI extra first: pip install -e '.[tui]'") from exc

    class GausserApp(App[None]):
        TITLE = "Gausser"
        CSS = """
        #overview, #tools { height: auto; margin: 1 2; }
        #stage-controls { height: auto; margin: 1 2; }
        #log { height: 1fr; margin: 1 2; border: round $primary; }
        #inputs { height: auto; margin: 0 2; }
        #prompt, #lod-source { width: 1fr; }
        """
        BINDINGS = [("q", "quit", "Quit"), ("escape", "cancel_stage", "Cancel stage"), ("r", "refresh", "Refresh")]

        def __init__(self, scene: ScenePaths) -> None:
            super().__init__()
            self.scene = scene
            self.controller = SceneController(scene)
            self.cancel_event: threading.Event | None = None
            self.pending_request: StageRequest | str | None = None

        def compose(self) -> ComposeResult:
            yield Header()
            with Container():
                yield Static(f"Scene: {self.scene.root}", id="scene")
                yield Static("", id="overview")
                yield Static("", id="tools")
                with Horizontal(id="inputs"):
                    yield Input(placeholder="SAM3 prompt", id="prompt")
                    yield Input(placeholder="PLY export for LOD", id="lod-source")
                with Horizontal(id="stage-controls"):
                    yield Button("SAM3", id="sammer", variant="primary")
                    yield Button("COLMAP", id="colmap")
                    yield Button("Brush", id="brush")
                    yield Button("Msplat", id="msplat", variant="warning")
                    yield Button("LOD", id="lod")
                    yield Button("Pipeline", id="pipeline", variant="success")
                    yield Button("Cancel", id="cancel", variant="error")
                yield RichLog(id="log", wrap=True, highlight=False)
            yield Footer()

        def on_mount(self) -> None:
            self.refresh_status()

        def refresh_status(self) -> None:
            scene_status = inspect_scene(self.scene)
            tool_status = inspect_tools()
            self.query_one("#overview", Static).update(
                "\n".join(f"{item.stage:8} {item.state:11} {item.detail}" for item in scene_status)
            )
            self.query_one("#tools", Static).update(
                "Tools\n" + "\n".join(f"{item.stage:8} {item.state:11} {item.detail}" for item in tool_status)
            )
            states = {item.stage: item.state for item in scene_status}
            tools = {item.stage: item.state for item in tool_status}
            for stage in ("sammer", "colmap", "brush", "msplat", "lod"):
                self.query_one(f"#{stage}", Button).disabled = (
                    states.get(stage) != "ready" or tools.get(stage) != "ready"
                )
            self.query_one("#pipeline", Button).disabled = states["sammer"] != "ready" or any(
                tools[stage] != "ready" for stage in ("sammer", "colmap", "brush")
            )

        def action_refresh(self) -> None:
            self.refresh_status()

        def action_cancel_stage(self) -> None:
            if self.cancel_event:
                self.cancel_event.set()
                self.query_one("#log", RichLog).write("Cancellation requested; waiting for process cleanup.")
            elif self.pending_request:
                self.pending_request = None
                self.query_one("#log", RichLog).write("Pending stage launch cancelled.")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "cancel":
                self.action_cancel_stage()
                return
            stage = event.button.id
            if stage == "pipeline":
                prompt = self.query_one("#prompt", Input).value.strip()
                if not prompt:
                    self.query_one("#log", RichLog).write("A SAM3 prompt is required for the pipeline.")
                    return
                if self.pending_request != "pipeline":
                    self.pending_request = "pipeline"
                    self.query_one("#log", RichLog).write("Ready to run SAM3 → COLMAP → Brush. Press Run pipeline again to confirm.")
                    return
                self.pending_request = None
                self.start_pipeline(prompt)
                return
            if stage not in {"sammer", "colmap", "brush", "msplat", "lod"}:
                return
            arguments: tuple[str, ...] = ()
            if stage == "sammer":
                prompt = self.query_one("#prompt", Input).value.strip()
                if not prompt:
                    self.query_one("#log", RichLog).write("SAM3 needs a text prompt.")
                    return
                arguments = ("--prompt", prompt)
            if stage == "lod":
                source = self.query_one("#lod-source", Input).value.strip()
                if not source:
                    self.query_one("#log", RichLog).write("LOD creation requires an explicit scene-owned PLY export path.")
                    return
                arguments = (source,)
            request = StageRequest(stage, arguments)
            if request != self.pending_request:
                self.pending_request = request
                message = (
                    "Msplat does not consume SAM masks and will train unmasked. "
                    "Press Run Msplat (unmasked) again to acknowledge and start."
                    if stage == "msplat"
                    else "LOD creation derives deployable data from the selected PLY export. "
                    "Press Create LOD again to confirm."
                    if stage == "lod"
                    else f"Ready to run {stage}. Press its button again to confirm. Esc cancels a running stage."
                )
                self.query_one("#log", RichLog).write(message)
                return
            self.pending_request = None
            self.start_stage(request)

        @work(thread=True, group="stage", exclusive=True, exit_on_error=False)
        def start_stage(self, request: StageRequest) -> None:
            self.cancel_event = threading.Event()
            log = self.query_one("#log", RichLog)
            self.call_from_thread(log.write, f"$ {' '.join(self.controller.command_for(request))}")
            try:
                result = self.controller.run(
                    request,
                    on_line=lambda line: self.call_from_thread(log.write, line),
                    cancel_event=self.cancel_event,
                )
            except StageUnavailable as exc:
                self.call_from_thread(log.write, str(exc))
            else:
                status = "completed" if result.succeeded else "cancelled" if result.cancelled else f"failed ({result.returncode})"
                self.call_from_thread(log.write, f"Stage {request.stage} {status}.")
            finally:
                self.cancel_event = None
                self.call_from_thread(self.refresh_status)

        @work(thread=True, group="stage", exclusive=True, exit_on_error=False)
        def start_pipeline(self, prompt: str) -> None:
            self.cancel_event = threading.Event()
            log = self.query_one("#log", RichLog)
            requests = {
                "sammer": StageRequest("sammer", ("--prompt", prompt)),
                "colmap": StageRequest("colmap"),
                "brush": StageRequest("brush"),
            }
            self.call_from_thread(log.write, "Starting SAM3 → COLMAP → Brush.")
            try:
                results = self.controller.run_pipeline(
                    requests, on_line=lambda line: self.call_from_thread(log.write, line), cancel_event=self.cancel_event
                )
                self.call_from_thread(log.write, f"Pipeline stopped after {len(results)} stage(s).")
            except StageUnavailable as exc:
                self.call_from_thread(log.write, str(exc))
            finally:
                self.cancel_event = None
                self.call_from_thread(self.refresh_status)

    return GausserApp(scene)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gausser", description="Open the Gausser Textual scene controller.")
    parser.add_argument("scene", nargs="?", default=Path.cwd(), type=Path, help="Scene root (default: current directory)")
    args = parser.parse_args(argv)
    create_app(ScenePaths.from_root(args.scene)).run()
