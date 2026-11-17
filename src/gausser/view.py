"""Open a splat in a local SuperSplat build, served from 127.0.0.1."""

from __future__ import annotations

import os
import threading
import webbrowser
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from .scene import GausserError, Scene

SUPERSPLAT_HINT = (
    "git clone https://github.com/playcanvas/supersplat ~/Developer/supersplat && "
    "cd ~/Developer/supersplat && npm ci && npm run build"
)


def supersplat_dist() -> Path:
    return Path(os.environ.get("GAUSSER_SUPERSPLAT", "~/Developer/supersplat/dist")).expanduser()


@contextmanager
def serve(splat: Path) -> Iterator[str]:
    """Serve SuperSplat plus one splat on 127.0.0.1 in a thread; yield the URL that opens it."""
    dist = supersplat_dist()
    if not (dist / "index.html").is_file():
        raise GausserError(f"No SuperSplat build at {dist} (set GAUSSER_SUPERSPLAT). Install: {SUPERSPLAT_HINT}")
    splat_path = "/_gausser/" + quote(splat.name)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(dist), **kwargs)

        def translate_path(self, path: str) -> str:
            # The one splat is served next to SuperSplat so the editor can fetch it same-origin.
            return str(splat) if urlsplit(path).path == splat_path else super().translate_path(path)

        def handle(self) -> None:
            try:
                super().handle()
            except ConnectionError:
                pass  # the browser abandons requests it no longer needs (e.g. the splat on reload)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}/?load={quote(splat_path)}"
        finally:
            server.shutdown()


def view(scene: Scene, source: Path | None) -> None:
    splat = scene.splat(source)
    with serve(splat) as url:
        print(f"Viewing {splat}\n{url}\nCtrl+C to stop", flush=True)
        webbrowser.open(url)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
