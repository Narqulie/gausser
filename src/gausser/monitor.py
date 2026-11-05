"""`gausser monitor`: a read-only web page of every scene, live while a command runs and a viewer afterwards.

One monitor (port 8737) serves all scenes in the index (`scene.index_path()`). It reads the
scenes' files and never takes a scene lock. The page itself is a React app built
from web/ into src/gausser/web/ (scripts/install-macos.sh builds it).
"""

from __future__ import annotations

import io
import json
import math
import mimetypes
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

from . import pano
from .scene import GausserError, Scene, index, index_path, is_image, mtime, registered_images

WEB = Path(__file__).with_name("web")
WEB_HINT = "cd web && npm ci && npm run build (in the gausser repository), or rerun scripts/install-macos.sh"
PORT = 8737  # one monitor serves every scene, so its address stays the same
IDLE_SECONDS = 600  # a spawned monitor quits after this long without a page polling it and nothing running
MAX_POINTS = 200_000
# Parameter counts of COLMAP camera models by model id, and the index of fy (else fy = fx).
CAMERA_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12, 11: 16}
FY_INDEX = {1: 1, 4: 1, 5: 1, 6: 1, 10: 1, 11: 1}


# Terminal control sequences: CSI (colours, cursor moves), OSC/APC/DCS strings (titles, kitty graphics), others.
ANSI = re.compile(rb"\x1b\[[0-?]*[ -/]*[@-~]|\x1b[\]_P^][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")


def clean(raw: bytes) -> str:
    """Tool output as a terminal would finally show it: no control sequences, a progress line only as last drawn."""
    text = ANSI.sub(b"", raw).decode(errors="replace")
    return "\n".join(line.rstrip("\r").rsplit("\r", 1)[-1] for line in text.split("\n"))


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def holder(scene: Scene) -> dict[str, Any] | None:
    """The command holding the scene lock, if it is still running."""
    info = _read_json(scene.root / ".gausser" / "lock")
    if isinstance(info, dict) and isinstance(info.get("pid"), int) and _alive(info["pid"]):
        return info
    return None


def legacy_step(scene: Scene) -> dict[str, Any] | None:
    """A step of a gausser from before history and logs existed, seen by its staging folder still being written."""
    if isinstance(_read_json(scene.root / ".gausser" / "lock"), dict):
        return None  # the lock names its holder, and it has ended
    newest, when = None, 0.0
    for staging in scene.tmp.glob("*-*"):
        written = max([mtime(staging), *(mtime(p) for p in staging.iterdir())]) if staging.is_dir() else 0.0
        if written > when:
            newest, when = staging, written
    if newest is None or time.time() - when > 300:
        return None
    kind, _, started = newest.name.partition("-")
    step = {"brush": "train"}.get(kind, kind)
    try:
        start = time.mktime(time.strptime(started[:15], "%Y%m%d-%H%M%S"))
    except ValueError:
        start = mtime(newest)
    return {"step": step, "start": start, "log": None}


def history(scene: Scene) -> list[dict[str, Any]]:
    """Steps from .gausser/history.jsonl, oldest first; a step's end record completes its start record."""
    steps: dict[tuple[str, float], dict[str, Any]] = {}
    try:
        lines = scene.history.read_text().splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            record = json.loads(line)
            steps.setdefault((record["step"], record["start"]), {}).update(record)
        except (ValueError, KeyError, TypeError):
            continue
    return list(steps.values())


def _rel(scene: Scene, path: Path) -> str:
    return str(path.relative_to(scene.root))


def _newest(paths: list[Path]) -> Path | None:
    return max(paths, key=mtime, default=None)


def live(scene: Scene, step: dict[str, Any]) -> dict[str, Any]:
    """What a running step has made so far in its staging folder (.gausser/tmp/<step>-<time>/)."""
    staging = _newest([p for p in scene.tmp.glob(f"{'brush' if step['step'] == 'train' else step['step']}-*") if p.is_dir()])
    if staging is None or mtime(staging) < step["start"] - 5:
        return {}
    result: dict[str, Any] = {"staging": _rel(scene, staging)}
    if step["step"] == "train":
        settled = time.time() - 3  # a .ply still being written would not load
        exports = sorted(
            (p for p in staging.glob("exports/*.ply") if mtime(p) < settled), key=mtime,
        )
        result["exports"] = [_export(scene, p) for p in exports]
    elif step["step"] == "solve":
        models = [p for p in staging.glob("snapshots/*") if (p / "images.bin").is_file()]
        models += [p for p in staging.glob("sparse/*") if (p / "points3D.bin").is_file()]
        newest = _newest(models)
        if newest is not None:
            result["model"] = {"path": _rel(scene, newest), "registered": registered_images(newest), "mtime": mtime(newest)}
    elif step["step"] == "mask":
        result["count"] = sum(1 for _ in staging.glob("masks/*.png"))
    elif step["step"] == "frames":
        result["count"] = sum(1 for p in staging.glob("images/*") if is_image(p))
    return result


def _export(scene: Scene, path: Path) -> dict[str, Any]:
    stem = path.stem.rsplit("_", 1)[-1]
    run = path.parent.parent
    marker = next((m for m in ("masked", "unmasked", "excluded") if (run / m).exists()), None)
    return {
        "path": _rel(scene, path), "run": _rel(scene, run), "size": path.stat().st_size, "mtime": mtime(path),
        "iteration": int(stem) if stem.isdigit() else None, "marker": marker,
    }


def state(scene: Scene) -> dict[str, Any]:
    images = scene.image_files()
    model = scene.model()
    running = holder(scene)
    steps = history(scene)
    current = next((s for s in reversed(steps) if "end" not in s), None) if running else None
    if running is None and (current := legacy_step(scene)):
        running = {"pid": None, "argv": [current["step"], "(started by an older gausser: no log)"], "start": current["start"]}
    first = None
    if images:
        try:
            from PIL import Image

            with Image.open(images[0]) as image:
                first = image.size
        except OSError:
            pass
    return {
        "name": scene.root.name,
        "root": str(scene.root),
        "now": time.time(),
        "holder": running,
        "history": steps,
        "current": {**current, "live": live(scene, current)} if current else None,
        "images": {
            "count": len(images), "size": first, "loose": len(scene.loose_photos()),
            "panos": sum(1 for p in scene.panos.iterdir() if is_image(p)) if scene.is_pano() else 0,
            "views": pano.VIEWS if scene.is_pano() else 0,
        },
        "masks": {
            "count": sum(1 for _ in scene.masks.glob("*.png")) if scene.masks.is_dir() else 0,
            "exclude": sum(1 for _ in scene.masks_exclude.glob("*.png")) if scene.masks_exclude.is_dir() else 0,
            "done": bool(images) and scene.masks_done(images),
        },
        "model": model and {
            "path": _rel(scene, model), "registered": registered_images(model), "mtime": mtime(model / "images.bin"),
            "current": scene.solve_done(images), "no_masks": (scene.sparse / "no-masks").is_file(),
        },
        "exports": [_export(scene, p) for p in scene.exports()],
        "trained": scene.train_done(),
        "lods": [{"path": _rel(scene, p), "mtime": mtime(p / "lod-meta.json")} for p in scene.lod_runs()],
        "lod_current": scene.lod_done(),
    }


def read_model(model: Path, max_points: int = MAX_POINTS) -> dict[str, Any]:
    """Cameras, a subsample of the points and statistics of a binary COLMAP model."""
    import numpy as np

    data = (model / "cameras.bin").read_bytes()
    intrinsics: dict[int, tuple[float, float, int, int]] = {}
    (count,), at = struct.unpack_from("<Q", data), 8
    for _ in range(count):
        camera_id, kind, width, height = struct.unpack_from("<iiQQ", data, at)
        at += 24
        params = struct.unpack_from(f"<{CAMERA_PARAMS[kind]}d", data, at)
        at += 8 * CAMERA_PARAMS[kind]
        intrinsics[camera_id] = (params[0], params[FY_INDEX.get(kind, 0)], width, height)

    data = (model / "images.bin").read_bytes()
    cameras = []
    rights, downs = [], []
    (count,), at = struct.unpack_from("<Q", data), 8
    for _ in range(count):
        _id, qw, qx, qy, qz, tx, ty, tz, camera_id = struct.unpack_from("<idddddddi", data, at)
        end = data.index(b"\0", at + 64)
        name = data[at + 64:end].decode(errors="replace")
        (points,) = struct.unpack_from("<Q", data, end + 1)
        at = end + 9
        observed = np.frombuffer(data, dtype="<i8", count=points * 3, offset=at)[2::3] if points else np.empty(0)
        at += points * 24
        r = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ])
        fx, fy, width, height = intrinsics.get(camera_id, (1.0, 1.0, 1, 1))
        rights.append(r[0])
        downs.append(r[1])
        cameras.append({
            "name": name, "center": np.round(-r.T @ np.array([tx, ty, tz]), 5).tolist(),
            "rotation": np.round(r.T, 5).ravel().tolist(),  # camera to world, OpenCV axes (x right, y down, z forward)
            "fx": fx, "fy": fy, "width": width, "height": height, "points": int((observed >= 0).sum()),
        })

    # Up: normal to the cameras' level right axes, on the side image-up points to (see orbit.camera_up).
    up = [0.0, -1.0, 0.0]
    if len(rights) >= 3:
        normal = np.linalg.svd(np.array(rights), full_matrices=False)[2][-1]
        up = (normal if normal @ -np.mean(downs, axis=0) > 0 else -normal).tolist()

    data = (model / "points3D.bin").read_bytes()
    (count,), at = struct.unpack_from("<Q", data), 8
    xyz = np.empty((count, 3))
    rgb = np.empty((count, 3), dtype=np.uint8)
    errors = np.empty(count)
    tracks = np.empty(count, dtype=np.int64)
    for i in range(count):
        x, y, z, r_, g, b, error, track = struct.unpack_from("<dddBBBdQ", data, at + 8)
        xyz[i] = x, y, z
        rgb[i] = r_, g, b
        errors[i], tracks[i] = error, track
        at += 51 + track * 8
    keep = np.random.default_rng(0).choice(count, max_points, replace=False) if count > max_points else slice(None)
    lengths, counts = np.unique(np.minimum(tracks, 30), return_counts=True)
    return {
        "up": up,
        "cameras": cameras,
        "points": {"xyz": np.round(xyz[keep], 4).ravel().tolist(), "rgb": rgb[keep].ravel().tolist()},
        "stats": {
            "points": count,
            "mean_error": float(errors.mean()) if count else 0.0,
            "mean_track": float(tracks.mean()) if count else 0.0,
            "track_lengths": [[int(n), int(c)] for n, c in zip(lengths, counts)],  # 30 = 30 or more
        },
    }


def image_stats(path: Path, masks: list[Path]) -> dict[str, Any]:
    """Sharpness (variance of the Laplacian at 512 px) and the white share of each mask."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        image.draft("L", (512, 512))
        gray = image.convert("L")
        gray.thumbnail((512, 512))
        g = np.asarray(gray, dtype=np.float32)
    laplacian = g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4 * g[1:-1, 1:-1]
    stats: dict[str, Any] = {"sharpness": round(float(laplacian.var()), 1)}
    for mask in masks:
        if mask.is_file():
            with Image.open(mask) as m:
                m.draft("L", (128, 128))
                stats[mask.parent.name] = round(float((np.asarray(m.convert("L").resize((128, 128))) > 127).mean()), 4)
    return stats


def thumbnail(path: Path, size: int) -> bytes:
    from PIL import Image

    with Image.open(path) as source:
        source.draft("RGB", (size, size))
        image = source.convert("RGB")
    image.thumbnail((size, size))
    out = io.BytesIO()
    image.save(out, "JPEG", quality=80)
    return out.getvalue()


def psnr(render: Path, photo: Path) -> float:
    """PSNR of a render against its photo, scaled to the render's size, over the whole image."""
    import numpy as np
    from PIL import Image

    with Image.open(render) as r:
        a = np.asarray(r.convert("RGB"), dtype=np.float32) / 255
    with Image.open(photo) as p:
        p.draft("RGB", (a.shape[1], a.shape[0]))
        b = np.asarray(p.convert("RGB").resize((a.shape[1], a.shape[0])), dtype=np.float32) / 255
    mse = float(((a - b) ** 2).mean())
    return round(10 * math.log10(1 / mse), 2) if mse > 0 else 99.0


def known_scenes() -> dict[str, float]:
    """Scene folders in the index, with the time of their last record."""
    scenes: dict[str, float] = {}
    try:
        lines = index_path().read_text().splitlines()
    except OSError:
        return scenes
    for line in lines:
        try:
            record = json.loads(line)
            scenes[record["scene"]] = max(scenes.get(record["scene"], 0.0), record.get("end") or record.get("start") or record.get("opened", 0.0))
        except (ValueError, KeyError, TypeError):
            continue
    return scenes


def recent_runs(limit: int = 300) -> list[dict[str, Any]]:
    """The newest finished or running steps of all scenes, newest first."""
    steps: dict[tuple[str, str, float], dict[str, Any]] = {}
    try:
        lines = index_path().read_text().splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            record = json.loads(line)
            if "step" in record:
                steps.setdefault((record["scene"], record["step"], record["start"]), {}).update(record)
        except (ValueError, KeyError, TypeError):
            continue
    return sorted(steps.values(), key=lambda r: r["start"], reverse=True)[:limit]


def summary(root: str, last: float) -> dict[str, Any]:
    """A scene for the list of all scenes: its state with only the newest record of each step."""
    path = Path(root)
    if not path.is_dir():
        return {"root": root, "name": path.name, "last": last, "missing": True}
    try:
        info = state(Scene(path))
    except GausserError as exc:
        return {"root": root, "name": path.name, "last": last, "error": str(exc)}
    newest: dict[str, dict[str, Any]] = {}
    for record in info["history"]:
        newest[record["step"]] = record
    return {**info, "history": list(newest.values()), "last": last}


def looks_like_scene(scene: Scene) -> bool:
    return any(p.exists() for p in (scene.images, scene.panos, scene.root / ".gausser")) or bool(scene.loose_photos())


class Monitor:
    """Serves every scene in the index; each request names its scene folder."""

    def __init__(self) -> None:
        self.last_request = time.time()
        self.cache: dict[Any, Any] = {}
        self.pool = ThreadPoolExecutor(8)

    def cached(self, key: Any, make: Any) -> Any:
        if key not in self.cache:
            self.cache[key] = make()
        return self.cache[key]

    def scene(self, root: str | None) -> Scene:
        """Only scenes in the index are served."""
        if not root or root not in known_scenes():
            raise FileNotFoundError(root)
        return Scene(root)

    def model(self, scene: Scene, rel: str) -> bytes:
        path = self.path(scene, rel)
        if not (path / "images.bin").is_file():
            raise FileNotFoundError(rel)
        key = ("model", path, mtime(path / "images.bin"), mtime(path / "points3D.bin"))
        return self.cached(key, lambda: json.dumps(read_model(path)).encode())  # type: ignore[no-any-return]

    def images(self, scene: Scene) -> list[dict[str, Any]]:
        files = scene.image_files()
        panos = {p.stem: _rel(scene, p) for p in scene.panos.iterdir() if is_image(p)} if scene.is_pano() else {}

        def one(image: Path) -> dict[str, Any]:
            masks = [scene.masks / f"{image.stem}.png", scene.masks_exclude / f"{image.stem}.png"]
            key = ("image", image, mtime(image), *(mtime(m) for m in masks))
            stats = self.cached(key, lambda: image_stats(image, masks))
            view = pano.view_index(image.name) if scene.is_pano() else None
            frame = image.stem.split("_", 1)[1] if view is not None else image.stem
            return {
                "name": image.name, "frame": frame, "view": view, "pano": panos.get(frame), "mtime": mtime(image),
                "mask": masks[0].is_file() and _rel(scene, masks[0]),
                "exclude": masks[1].is_file() and _rel(scene, masks[1]), **stats,
            }

        return list(self.pool.map(one, files))

    def evals(self, scene: Scene, rel: str) -> list[dict[str, Any]]:
        """Brush's renders of the held-out images (`train --eval-split N`) in a run or staging folder, by step."""
        run = self.path(scene, rel)
        photos = {image.stem: image for image in scene.image_files()}
        result = []
        for folder in run.glob("exports/eval_*"):
            step = folder.name.removeprefix("eval_")
            renders = sorted(folder.glob("images/*.png"))
            if not step.isdigit() or not renders:
                continue

            def one(render: Path) -> dict[str, Any]:
                photo = photos.get(render.stem)
                score = self.cached(("psnr", render, mtime(render)), lambda: psnr(render, photo)) if photo else None
                return {"name": render.stem, "render": _rel(scene, render), "photo": photo and _rel(scene, photo), "psnr": score}

            result.append({"step": int(step), "mtime": mtime(folder), "images": list(self.pool.map(one, renders))})
        return sorted(result, key=lambda e: e["step"])

    def path(self, scene: Scene, rel: str) -> Path:
        """A path inside the scene; nothing outside it is served."""
        path = (scene.root / rel).resolve()
        if not path.is_relative_to(scene.root):
            raise PermissionError(rel)
        return path

    def log(self, scene: Scene, name: str, offset: int | None) -> dict[str, Any]:
        """Whole lines of a step log from ``offset`` (default: the last 4 MB), cleaned, plus the unfinished last line."""
        path = scene.logs / Path(name).name
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            start = max(0, size - 4_000_000) if offset is None else min(offset, size)
            stream.seek(start)
            raw = stream.read(size - start)
        if offset is None and start > 0:  # begin at a line
            cut = raw.find(b"\n") + 1
            raw, start = raw[cut:], start + cut
        end = raw.rfind(b"\n") + 1
        return {
            "text": clean(raw[:end]), "partial": clean(raw[end:]), "offset": start + end, "truncated": start > 0 and offset is None,
        }


def make_handler(app: Monitor) -> type[SimpleHTTPRequestHandler]:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(WEB), **kwargs)

        def do_GET(self) -> None:  # noqa: N802
            url = urlsplit(self.path)
            if not url.path.startswith("/api/"):
                super().do_GET()
                return
            app.last_request = time.time()
            query = {key: values[-1] for key, values in parse_qs(url.query).items()}
            try:
                self.api(url.path[5:], query)
            except (FileNotFoundError, NotADirectoryError):
                self.send_error(404)
            except PermissionError:
                self.send_error(403)
            except ConnectionError:
                pass

        def api(self, route: str, query: dict[str, str]) -> None:
            if route == "scenes":
                scenes = sorted(known_scenes().items(), key=lambda item: item[1], reverse=True)
                body = {"scenes": list(app.pool.map(lambda item: summary(*item), scenes)), "runs": recent_runs(), "now": time.time()}
                self.json(body)
                return
            scene = app.scene(query.get("scene"))
            if route == "state":
                self.json(state(scene))
            elif route == "images":
                self.json(app.images(scene))
            elif route == "frames":  # Sharp Frames' record of the frames it chose
                folder = scene.panos if scene.is_pano() else scene.images
                self.json(_read_json(folder / "selected_metadata.json"))
            elif route == "model":
                self.send(app.model(scene, query.get("path", "sparse/0")), "application/json")
            elif route == "evals":
                self.json(app.evals(scene, query["run"]))
            elif route == "log":
                offset = int(query["offset"]) if "offset" in query else None
                self.json(app.log(scene, query["name"], offset))
            elif route == "thumb":
                path, size = app.path(scene, query["path"]), min(int(query.get("size", 256)), 1024)
                data = app.cached(("thumb", path, mtime(path), size), lambda: thumbnail(path, size))
                self.send(data, "image/jpeg", cache=True)
            elif route == "file":
                self.file(app.path(scene, query["path"]))
            else:
                self.send_error(404)

        def json(self, value: Any) -> None:
            self.send(json.dumps(value).encode(), "application/json")

        def send(self, body: bytes, kind: str, cache: bool = False) -> None:
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def file(self, path: Path) -> None:
            with path.open("rb") as stream:
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(os.fstat(stream.fileno()).st_size))
                self.end_headers()
                shutil.copyfileobj(stream, self.wfile, 1 << 20)

        def handle(self) -> None:
            try:
                super().handle()
            except ConnectionError:
                pass  # the page abandons requests it no longer needs

        def log_message(self, format: str, *args: Any) -> None:
            pass

    return Handler


def _info_path() -> Path:
    return index_path().with_name("monitor.json")


def running_monitor() -> str | None:
    """The URL of the monitor that is running, if any."""
    info = _read_json(_info_path())
    if isinstance(info, dict) and isinstance(info.get("pid"), int) and _alive(info["pid"]):
        return str(info["url"])
    return None


def scene_url(base: str, scene: Scene | None) -> str:
    return base + (f"?scene={quote(str(scene.root))}" if scene else "")


def register(scene: Scene | None) -> Scene | None:
    """Add the scene to the index so the monitor serves it; a folder that is no scene opens the list of scenes."""
    if scene is None or not looks_like_scene(scene):
        return None
    index(scene, {"opened": time.time()})
    return scene


def monitor(scene: Scene | None, *, port: int = PORT, open_browser: bool = True, quit_when_idle: bool = False) -> None:
    """Show the scene (or, with None, every scene) in the monitor, starting the one monitor if it is not running."""
    scene = register(scene)
    url = running_monitor()
    if url:
        print(f"Monitor: {scene_url(url, scene)}", flush=True)
        if open_browser:
            webbrowser.open(scene_url(url, scene))
        return
    if not (WEB / "index.html").is_file():
        raise GausserError(f"The monitor page is not built ({WEB} is missing). Build it: {WEB_HINT}")
    app = Monitor()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    except OSError:  # the port is taken by something else
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    info = _info_path()
    with server:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        info.parent.mkdir(parents=True, exist_ok=True)
        info.write_text(json.dumps({"pid": os.getpid(), "url": url}))
        print(f"Monitor: {scene_url(url, scene)}\nAll scenes: {url}\nCtrl+C to stop", flush=True)
        if open_browser:
            webbrowser.open(scene_url(url, scene))

        def watch_idle() -> None:
            while True:
                time.sleep(30)
                busy = any(holder(Scene(root)) for root in known_scenes() if Path(root).is_dir())
                if time.time() - app.last_request > IDLE_SECONDS and not busy:
                    server.shutdown()
                    return

        if quit_when_idle:
            threading.Thread(target=watch_idle, daemon=True).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if (_read_json(info) or {}).get("pid") == os.getpid():
                info.unlink(missing_ok=True)


def spawn(scene: Scene) -> None:
    """Show the scene in the running monitor, or start one in the background that quits once its pages are closed."""
    if running_monitor():
        monitor(scene)
        return
    if not (WEB / "index.html").is_file():
        print(f"gausser: no monitor, the page is not built. Build it: {WEB_HINT}", file=sys.stderr)
        return
    log = index_path().with_name("monitor.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as stream:
        subprocess.Popen(
            [sys.executable, "-m", "gausser.cli", "monitor", "-s", str(scene.root), "--quit-when-idle"],
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
        )
    for _ in range(50):
        url = running_monitor()
        if url:
            print(f"Monitor: {scene_url(url, scene)}", flush=True)
            return
        time.sleep(0.1)
    print(f"gausser: the monitor did not start; see {log}", file=sys.stderr)
