"""Open a splat in a local SuperSplat build, served from 127.0.0.1."""

from __future__ import annotations

import os
import webbrowser
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


def view(scene: Scene, source: Path | None) -> None:
    dist = supersplat_dist()
    if not (dist / "index.html").is_file():
        raise GausserError(f"No SuperSplat build at {dist} (set GAUSSER_SUPERSPLAT). Install: {SUPERSPLAT_HINT}")
    if source is None:
        exports = scene.exports()
        if not exports:
            raise GausserError("No trained .ply export found; run `gausser train` first")
        source = exports[-1]
    splat = source.expanduser().resolve()
    if not splat.is_file():
        raise GausserError(f"No such file: {splat}")
    splat_path = "/_gausser/" + quote(splat.name)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(dist), **kwargs)

        def translate_path(self, path: str) -> str:
            # The one splat is served next to SuperSplat so the editor can fetch it same-origin.
            return str(splat) if urlsplit(path).path == splat_path else super().translate_path(path)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        url = f"http://127.0.0.1:{server.server_address[1]}/?load={quote(splat_path)}"
        print(f"Viewing {splat}\n{url}\nCtrl+C to stop", flush=True)
        webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
