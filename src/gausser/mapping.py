"""`gausser map`: an orthophoto and a surface model (DSM) of a georeferenced scene, from its trained splat.

The splat is rendered looking straight down with an orthographic camera: each Gaussian projects onto
the ground plane as a 2D Gaussian and is alpha-blended top first, as a splat renderer does, but on a
fixed ground grid. The same blend gives each cell its surface height, which is the DSM: the height of
the splat at which the cell becomes half covered (the median depth, so faint splats above do not lift it). Both are
GeoTIFFs in the scene's CRS (sparse/geo.json), next to map.html, which lays the orthophoto over a web
map with the cameras' GPS positions.

Local axes are gausser's: x east, y down, z north, in metres from the origin; image row 0 is north.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import ply
from .colmap_model import quaternion_matrix
from .scene import GausserError, Scene, replace

SH_C0 = 0.28209479177387814
# Real spherical-harmonics basis of degrees 1-3 for a view straight down, d = (0, 1, 0) in local axes
# (3DGS's convention: the direction from the camera to the splat). Only terms with y alone survive.
SH_DOWN = (
    -0.4886025119029199, 0.0, 0.0,  # degree 1: -C1·y, C1·z, -C1·x
    0.0, 0.0, -0.31539156525252005, 0.0, -0.5462742152960396,  # degree 2
    0.5900435899266435, 0.0, 0.4570457994644658, 0.0, 0.0, 0.0, 0.0,  # degree 3
)
MAX_RADIUS = 512  # px; see render()
MAX_PAIRS = 10_000_000  # splat-pixel pairs blended at once: about 4 GB at the peak


def splat_colour(splat: dict[str, Any]) -> Any:
    """RGB in 0-1 of each splat seen from straight above (spherical harmonics up to the degree present)."""
    import numpy as np

    rgb = 0.5 + SH_C0 * np.stack([splat[f"f_dc_{c}"] for c in range(3)], 1)
    rest = sorted((name for name in splat if name.startswith("f_rest_")), key=lambda n: int(n.split("_")[-1]))
    per_channel = len(rest) // 3  # 3DGS stores f_rest channel by channel
    for c in range(3):
        for k in range(min(per_channel, len(SH_DOWN))):
            if SH_DOWN[k]:
                rgb[:, c] += SH_DOWN[k] * splat[rest[c * per_channel + k]]
    return np.clip(rgb, 0, 1)


def ground_covariance(splat: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Each Gaussian's covariance projected onto the ground: the x-x, x-z and z-z terms of R·S²·Rᵀ."""
    import numpy as np

    q = np.stack([splat[f"rot_{i}"] for i in range(4)], 1)  # w, x, y, z
    r = np.moveaxis(quaternion_matrix(*(q / np.linalg.norm(q, axis=1, keepdims=True)).T), 2, 0)
    s2 = np.exp(2 * np.stack([splat[f"scale_{i}"] for i in range(3)], 1))
    rx, rz = r[:, 0, :], r[:, 2, :]  # the x and z rows of R
    return (rx * rx * s2).sum(1), (rx * rz * s2).sum(1), (rz * rz * s2).sum(1)


def render(splat: dict[str, Any], west: float, north: float, width: int, height: int, resolution: float) -> tuple[Any, Any, Any]:
    """Orthographic top-down render: RGB (0-1, unpremultiplied), coverage (0-1) and surface y per cell (NaN below half coverage)."""
    import numpy as np

    opacity = 1 / (1 + np.exp(-splat["opacity"]))
    keep = opacity > 1 / 255
    x, y, z = splat["x"][keep], splat["y"][keep], splat["z"][keep]
    colour, opacity = splat_colour(splat)[keep], opacity[keep]
    sxx, sxz, szz = (term[keep] for term in ground_covariance(splat))
    # Pixel coordinates: column east, row south. A tiny blur keeps sub-pixel splats from aliasing (3DGS adds 0.3 px²).
    col, row = (x - west) / resolution, (north - z) / resolution
    a, b, c = sxx / resolution**2 + 0.3, -sxz / resolution**2, szz / resolution**2 + 0.3
    det = a * c - b * b
    # Out to where its alpha falls below 1/255 along its longest axis (3DGS uses a looser 3σ).
    sigma = np.sqrt(0.5 * (a + c) + np.sqrt(np.maximum(0.25 * (a - c) ** 2 + b * b, 0)))
    radius = np.ceil(sigma * np.sqrt(2 * np.log(np.maximum(255 * opacity, 1)))).astype(np.int64)
    # Wider splats are haze and floaters; cutting them to a box would draw its edges, so they are left out.
    radius = np.where(radius <= MAX_RADIUS, radius, -1)
    c0, c1 = np.maximum(np.floor(col - radius), 0).astype(np.int64), np.minimum(np.ceil(col + radius), width).astype(np.int64)
    r0, r1 = np.maximum(np.floor(row - radius), 0).astype(np.int64), np.minimum(np.ceil(row + radius), height).astype(np.int64)
    w, h = c1 - c0, r1 - r0
    visible = (w > 0) & (h > 0)
    order = np.argsort(y[visible], kind="stable")  # top first: y points down
    index = np.flatnonzero(visible)[order]
    conic = np.stack([c / det, -b / det, a / det], 1)  # inverse covariance (a b; b c)

    accum = np.zeros((height * width, 3))
    surface = np.full(height * width, np.nan)
    transmit = np.ones(height * width)
    sizes = (w * h)[index]
    starts = np.concatenate([[0], np.cumsum(sizes)])
    chunk_start = 0
    while chunk_start < len(index):
        chunk_end = int(np.searchsorted(starts, starts[chunk_start] + MAX_PAIRS, side="right")) - 1
        chunk_end = max(chunk_end, chunk_start + 1)
        ids = index[chunk_start:chunk_end]
        counts = sizes[chunk_start:chunk_end]
        owner = np.repeat(np.arange(len(ids)), counts)
        local = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        sid = ids[owner]
        cc = c0[sid] + local % w[sid]
        rr = r0[sid] + local // w[sid]
        dx, dy = cc + 0.5 - col[sid], rr + 0.5 - row[sid]
        power = -0.5 * (conic[sid, 0] * dx * dx + 2 * conic[sid, 1] * dx * dy + conic[sid, 2] * dy * dy)
        alpha = np.minimum(0.99, opacity[sid] * np.exp(power))
        use = alpha >= 1 / 255
        pixel, alpha, sid = (rr * width + cc)[use], alpha[use], sid[use]
        # Blend front to back per pixel: sort by pixel, keeping the depth order within each pixel.
        by_pixel = np.argsort(pixel, kind="stable")
        pixel, alpha, sid = pixel[by_pixel], alpha[by_pixel], sid[by_pixel]
        log_keep = np.log1p(-alpha)
        total = np.cumsum(log_keep)
        first = np.flatnonzero(np.r_[True, pixel[1:] != pixel[:-1]])
        group_base = np.repeat(np.r_[0.0, total][first], np.diff(np.r_[first, len(pixel)]))
        before = np.exp(total - log_keep - group_base)  # transmittance in front of each splat within this chunk
        ahead = transmit[pixel] * before  # transmittance in front of each splat
        weight = alpha * ahead
        half = (ahead > 0.5) & (ahead * (1 - alpha) <= 0.5)  # the splat at which the pixel becomes half covered
        surface[pixel[half]] = y[sid[half]]
        for channel in range(3):
            accum[:, channel] += np.bincount(pixel, weight * colour[sid, channel], minlength=len(transmit))
        ends = np.r_[first[1:], len(pixel)] - 1
        transmit[pixel[first]] *= np.exp(total[ends] - group_base[first])
        chunk_start = chunk_end
    coverage = 1 - transmit
    seen = np.maximum(coverage, 1e-6)
    rgb = (accum / seen[:, None]).reshape(height, width, 3)
    return rgb, coverage.reshape(height, width), surface.reshape(height, width)


def extent(splat: dict[str, Any], margin: float = 0.0) -> tuple[float, float, float, float]:
    """West, east, south, north of the opaque splats (0.5-99.5 percentiles, so lone floaters do not count)."""
    import numpy as np

    solid = 1 / (1 + np.exp(-splat["opacity"])) > 0.5
    x, z = splat["x"][solid], splat["z"][solid]
    if len(x) < 100:
        raise GausserError("The splat has too few opaque Gaussians to map")
    west, east = np.percentile(x, [0.5, 99.5])
    south, north = np.percentile(z, [0.5, 99.5])
    return west - margin, east + margin, south - margin, north + margin


@dataclass
class MeshPart:
    """One primitive of a textured mesh: model-coordinate vertices, glTF UVs (origin top left), triangles, texture."""

    vertices: Any
    uv: Any
    faces: Any
    texture: Path


def read_mesh(path: Path) -> list[MeshPart]:
    """The textured primitives of an OpenMVS `.glb` (TextureMesh --export-type glb).

    Vertex data are in model coordinates; the root node's matrix (OpenMVS adds a Y-up turn) is not applied.
    Buffers live in the BIN chunk or as base64 data URIs; textures are PNGs next to the file.
    """
    import base64
    import struct

    import numpy as np

    data = path.read_bytes()
    if data[:4] != b"glTF":
        raise GausserError(f"{path.name} is not a binary glTF file")
    length, at, gltf, binary = struct.unpack_from("<I", data, 8)[0], 12, None, b""
    while at < length:
        size, kind = struct.unpack_from("<I4s", data, at)
        chunk = data[at + 8:at + 8 + size]
        if kind == b"JSON":
            try:
                gltf = json.loads(chunk)
            except ValueError as exc:
                raise GausserError(f"{path.name} has a broken glTF JSON chunk: {exc}") from exc
        elif kind == b"BIN\0":
            binary = chunk
        at += 8 + size
    if gltf is None:
        raise GausserError(f"{path.name} has no glTF JSON chunk")
    buffers = [
        base64.b64decode(b["uri"].split(",", 1)[1]) if "uri" in b and b["uri"].startswith("data:")
        else (path.parent / b["uri"]).read_bytes() if "uri" in b else binary
        for b in gltf["buffers"]
    ]
    types = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
    widths = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}

    def accessor(index: int) -> Any:
        acc = gltf["accessors"][index]
        view = gltf["bufferViews"][acc["bufferView"]]
        dtype, width = np.dtype(types[acc["componentType"]]), widths[acc["type"]]
        offset = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        values = np.frombuffer(buffers[view["buffer"]], dtype=dtype, count=acc["count"] * width, offset=offset)
        return values.reshape(acc["count"], width) if width > 1 else values

    parts = []
    for primitive in (p for mesh in gltf["meshes"] for p in mesh["primitives"]):
        material = gltf["materials"][primitive["material"]]
        texture = gltf["textures"][material["pbrMetallicRoughness"]["baseColorTexture"]["index"]]
        image = gltf["images"][texture["source"]]
        parts.append(MeshPart(
            accessor(primitive["attributes"]["POSITION"]).astype(np.float64),
            accessor(primitive["attributes"]["TEXCOORD_0"]),
            accessor(primitive["indices"]).reshape(-1, 3).astype(np.int64),
            path.parent / image["uri"],
        ))
    return parts


def mesh_extent(parts: list[MeshPart]) -> tuple[float, float, float, float]:
    """West, east, south, north of the mesh vertices (0.5-99.5 percentiles: a stray fringe does not stretch the map)."""
    import numpy as np

    xyz = np.concatenate([part.vertices for part in parts])
    west, east = np.percentile(xyz[:, 0], [0.5, 99.5])
    south, north = np.percentile(xyz[:, 2], [0.5, 99.5])
    return float(west), float(east), float(south), float(north)


def render_mesh(parts: list[MeshPart], west: float, north: float, width: int, height: int, resolution: float,
                samples: int = 2) -> tuple[Any, Any, Any]:
    """Orthographic top-down render of a textured mesh: RGB (0-1), coverage (0/1) and the top surface's y per cell.

    Each triangle is rasterised on a grid `samples` times finer than the map (z-buffer: the highest surface wins,
    y points down), its texture sampled bilinearly, and the samples are averaged into map cells.
    """
    import numpy as np
    from PIL import Image

    step = resolution / samples
    w, h = width * samples, height * samples
    best = np.full(w * h, np.inf, dtype=np.float32)  # y of the top surface so far
    where = np.zeros((w * h, 2), dtype=np.float32)  # its texture coordinates
    which = np.full(w * h, -1, dtype=np.int16)  # and texture
    for number, part in enumerate(parts):
        v = part.vertices[part.faces]  # (n, 3 corners, xyz)
        col, row = (v[..., 0] - west) / step, (north - v[..., 2]) / step
        uv = part.uv[part.faces].astype(np.float64)
        area = (col[:, 1] - col[:, 0]) * (row[:, 2] - row[:, 0]) - (col[:, 2] - col[:, 0]) * (row[:, 1] - row[:, 0])
        c0 = np.maximum(np.floor(col.min(1) - 0.5), 0).astype(np.int64)
        c1 = np.minimum(np.ceil(col.max(1) - 0.5), w - 1).astype(np.int64)
        r0 = np.maximum(np.floor(row.min(1) - 0.5), 0).astype(np.int64)
        r1 = np.minimum(np.ceil(row.max(1) - 0.5), h - 1).astype(np.int64)
        bw, bh = c1 - c0 + 1, r1 - r0 + 1
        ids = np.flatnonzero((np.abs(area) > 1e-9) & (bw > 0) & (bh > 0))  # walls seen edge-on cover nothing
        sizes = (bw * bh)[ids]
        starts = np.concatenate([[0], np.cumsum(sizes)])
        first = 0
        while first < len(ids):
            last = max(int(np.searchsorted(starts, starts[first] + MAX_PAIRS, side="right")) - 1, first + 1)
            tri, counts = ids[first:last], sizes[first:last]
            owner = np.repeat(tri, counts)
            local = np.arange(counts.sum()) - np.repeat(starts[first:last] - starts[first], counts)
            cc, rr = c0[owner] + local % bw[owner], r0[owner] + local // bw[owner]
            px, py = cc + 0.5, rr + 0.5
            x0, x1, x2 = (col[owner, k] for k in range(3))
            y0, y1, y2 = (row[owner, k] for k in range(3))
            d = area[owner]
            l1 = ((px - x0) * (y2 - y0) - (x2 - x0) * (py - y0)) / d
            l2 = ((x1 - x0) * (py - y0) - (px - x0) * (y1 - y0)) / d
            l0 = 1 - l1 - l2
            inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
            owner, cc, rr, l0, l1, l2 = owner[inside], cc[inside], rr[inside], l0[inside], l1[inside], l2[inside]
            depth = (l0 * v[owner, 0, 1] + l1 * v[owner, 1, 1] + l2 * v[owner, 2, 1]).astype(np.float32)
            pixel = rr * w + cc
            order = np.lexsort((depth, pixel))  # per pixel, the highest surface first
            pixel, depth, owner, l0, l1, l2 = (x[order] for x in (pixel, depth, owner, l0, l1, l2))
            top = np.r_[True, pixel[1:] != pixel[:-1]]
            pixel, depth, owner, l0, l1, l2 = (x[top] for x in (pixel, depth, owner, l0, l1, l2))
            better = depth < best[pixel]
            pixel, owner = pixel[better], owner[better]
            best[pixel] = depth[better]
            weights = np.stack([l0[better], l1[better], l2[better]], 1)
            where[pixel] = np.einsum("nk,nkj->nj", weights, uv[owner])
            which[pixel] = number
            first = last
    rgb = np.zeros((w * h, 3), dtype=np.float32)
    for number, part in enumerate(parts):
        hit = np.flatnonzero(which == number)
        if not len(hit):
            continue
        with Image.open(part.texture) as image:
            texture = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        th, tw = texture.shape[:2]
        x = np.clip(where[hit, 0] * tw - 0.5, 0, tw - 1)
        y = np.clip(where[hit, 1] * th - 0.5, 0, th - 1)
        x0, y0 = np.minimum(x.astype(np.int64), tw - 2), np.minimum(y.astype(np.int64), th - 2)
        fx, fy = (x - x0)[:, None], (y - y0)[:, None]
        rgb[hit] = (texture[y0, x0] * (1 - fx) * (1 - fy) + texture[y0, x0 + 1] * fx * (1 - fy)
                    + texture[y0 + 1, x0] * (1 - fx) * fy + texture[y0 + 1, x0 + 1] * fx * fy)
    covered = (which >= 0).reshape(height, samples, width, samples)
    shaped = rgb.reshape(height, samples, width, samples, 3)
    count = covered.sum((1, 3))
    colour = shaped.sum((1, 3)) / np.maximum(count, 1)[..., None]
    top = best.reshape(height, samples, width, samples).min((1, 3))  # inf where nothing was drawn
    coverage = count / samples**2
    return colour, coverage, np.where(coverage > 0.5, top, np.nan)


COARSE = 0.05  # m: the grid on which the photo for each spot is chosen
CHOICE_BLUR = 1.0  # m: score smoothing, so the chosen photo changes in patches, not per cell
SEAM_BLEND = 0.4  # m: photos are cross-faded over about this distance where the choice changes


def render_photos(scene: Scene, surface: Any, west: float, north: float, width: int, height: int, resolution: float,
                  photos: Path | None = None) -> Any:
    """True orthophoto (RGB 0-1, NaN where no photo sees the ground) sampled straight from the photos.

    `surface` is the top surface's y on the COARSE grid (NaN where unknown). Every coarse cell gets the photo that sees
    it best: visible (depth test against the surface seen from that camera), looking most straight down, near the
    image centre, and sharp (motion blur differs per photo). Map cells then sample that photo bilinearly, from the
    full-resolution copies in `photos` (same names) when given, cross-fading neighbouring photos over SEAM_BLEND
    so exposure steps between photos do not show as hard seams.
    """
    import cv2
    import numpy as np
    from PIL import Image
    from scipy.ndimage import map_coordinates

    from . import colmap_model
    from .monitor import image_stats

    model = scene.model()
    if model is None:
        raise GausserError("No COLMAP model; run `gausser solve` first")
    cameras = colmap_model.read_cameras(model)
    images = [image for image in colmap_model.read_images(model) if (scene.images / image.name).is_file()]
    sharp = np.array([image_stats(scene.images / image.name, [])["sharpness"] for image in images])
    sharp = sharp / max(float(sharp.max()), 1e-6)

    ch, cw = surface.shape
    gx, gz = np.meshgrid(west + (np.arange(cw) + 0.5) * COARSE, north - (np.arange(ch) + 0.5) * COARSE)
    known = np.isfinite(surface)
    points = np.stack([gx[known], surface[known], gz[known]], 1)
    best = np.zeros(ch * cw, dtype=np.float32)
    choice = np.full(ch * cw, -1, dtype=np.int16)
    visible = np.zeros((len(images), ch * cw), dtype=bool)
    sigma = CHOICE_BLUR / COARSE
    for number, image in enumerate(images):
        camera = cameras[image.camera_id]
        local = points @ image.rotation.T + image.translation
        front = local[:, 2] > 0.5
        pixels = np.full((len(points), 2), -1.0)
        pixels[front] = camera.project(local[front])
        inside = front & (pixels[:, 0] >= 0) & (pixels[:, 0] < camera.width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < camera.height)
        # Depth test at 1/16 of the image: the nearest surface point per buffer pixel hides the ones behind it.
        bw, bh = camera.width // 16 + 1, camera.height // 16 + 1
        cell = (pixels[inside, 1] // 16).astype(np.int64) * bw + (pixels[inside, 0] // 16).astype(np.int64)
        distance = np.linalg.norm(local[inside], axis=1)
        nearest = np.full(bw * bh, np.inf)
        np.minimum.at(nearest, cell, distance)
        seen = np.zeros(len(points), dtype=bool)
        seen[np.flatnonzero(inside)] = distance <= nearest[cell] * 1.01 + 0.3
        ray = image.centre - points[seen]
        down = np.clip(-ray[:, 1] / np.linalg.norm(ray, axis=1), 0, 1)  # cos of the angle from straight down (y is down)
        offset = (pixels[seen] - [camera.width / 2, camera.height / 2]) / (np.hypot(camera.width, camera.height) / 2)
        centred = 1 - 0.5 * (offset**2).sum(1)
        score = np.zeros(ch * cw, dtype=np.float32)
        score[np.flatnonzero(known)[seen]] = down**4 * centred * sharp[number]
        mask = score > 0
        visible[number] = mask
        smooth = cv2.GaussianBlur(score.reshape(ch, cw), (0, 0), sigma).ravel() * mask
        better = smooth > best
        best[better], choice[better] = smooth[better], number
        print(f"photo {number + 1}/{len(images)}: sees {mask.mean():.0%} of the area", end="\r", flush=True)
    print()

    # Map cells: height and photo weights from the coarse grid.
    rows, cols = np.mgrid[0:height, 0:width]
    crow = np.minimum(((rows + 0.5) * resolution / COARSE).astype(np.int32), ch - 1)
    coarse = (crow * cw + np.minimum(((cols + 0.5) * resolution / COARSE).astype(np.int32), cw - 1)).ravel()
    del crow
    filled = np.where(known, surface, np.nanmedian(surface) if known.any() else 0)
    y = map_coordinates(filled, [((rows + 0.5) * resolution / COARSE - 0.5).ravel(), ((cols + 0.5) * resolution / COARSE - 0.5).ravel()],
                        order=1, mode="nearest").astype(np.float32)
    del rows, cols
    rgb = np.zeros((width * height, 3), dtype=np.float32)
    total = np.zeros(width * height, dtype=np.float32)
    for number, image in enumerate(images):
        chosen = (choice == number).reshape(ch, cw).astype(np.float32)
        if not chosen.any():
            continue
        weight = (cv2.GaussianBlur(chosen, (0, 0), SEAM_BLEND / COARSE).ravel() * visible[number])[coarse]
        cells = np.flatnonzero(weight > 0.01)
        weight = weight[cells]
        source = (photos / image.name) if photos and (photos / image.name).is_file() else scene.images / image.name
        with Image.open(source) as opened:
            photo = np.asarray(opened.convert("RGB"))
        camera = cameras[image.camera_id]
        scale = photo.shape[1] / camera.width
        world = np.stack([west + (cells % width + 0.5) * resolution, y[cells], north - (cells // width + 0.5) * resolution], 1)
        local = world @ image.rotation.T + image.translation
        uv = camera.project(local, scale) - 0.5  # pixel centres sit at +0.5
        for channel in range(3):
            rgb[cells, channel] += weight * map_coordinates(
                photo[..., channel], [uv[:, 1], uv[:, 0]], order=1, mode="nearest", output=np.float32,
            ) / 255
        total[cells] += weight
        print(f"sampled photo {number + 1}/{len(images)} ({scale:g}×)", end="\r", flush=True)
    print()
    rgb = np.where(total[:, None] > 0, rgb / np.maximum(total, 1e-6)[:, None], np.nan)
    return rgb.reshape(height, width, 3), y.reshape(height, width)


def _mesh_with_fill(scene: Scene, parts: list[MeshPart], west: float, north: float, width: int, height: int,
                    resolution: float, colour: bool = False) -> Any:
    """The mesh rendered straight down, its holes (glass, deep shadow, thin structures) filled from the newest splat.

    Returns RGB, coverage and surface y, plus the filled share when `colour`; otherwise None, coverage and y.
    """
    import numpy as np

    rgb, coverage, depth = render_mesh(parts, west, north, width, height, resolution)
    holes, filled = coverage <= 0.5, None
    if holes.any() and scene.exports():
        splat_source = scene.exports()[-1]
        print(f"Filling {holes.mean():.0%} of the cells the mesh leaves empty from {splat_source.name}", flush=True)
        _, data = ply.read(splat_source)
        fill = render({name: data[name] for name in data.dtype.names}, west, north, width, height, resolution)
        use = holes & (fill[1] > 0.5)
        rgb[use], coverage[use], depth[use] = fill[0][use], fill[1][use], fill[2][use]
        filled = round(float(np.mean(use)), 4)
    return (rgb, coverage, depth, filled) if colour else (None, coverage, depth)


def make(scene: Scene, source: Path | None = None, resolution: float = 0.03, kind: str = "splat",
         photos: Path | None = None) -> Path:
    """Write runs/map/<time>/: ortho.tif, dsm.tif (heights in the CRS's vertical frame of the GPS) and map.html.

    `kind` "splat" renders the newest (or given) splat export; "mesh" the newest (or given) textured OpenMVS
    mesh, whose colours come from the photos (sharper, and a true orthophoto); "photos" samples the photos
    themselves on the mesh surface (full-resolution copies in `photos`), the sharpest. "photos" covers the crop
    box when there is one, else the mesh.
    """
    import numpy as np

    record = scene.geo()
    if record is None:
        raise GausserError("The model is not georeferenced (no sparse/geo.json); solve photos or video frames with GPS")
    filled = None  # share of cells taken from the splat where the mesh has holes
    area = _crop_area(scene)
    if kind in ("mesh", "photos"):
        if source is None:
            meshes = scene.mesh_runs()
            if not meshes:
                raise GausserError("No mesh yet; run `gausser mesh` first")
            source = meshes[-1] / "mesh.glb"
        print(f"Reading {source}", flush=True)
        parts = read_mesh(source)
        west, east, south, north = mesh_extent(parts) if kind == "mesh" or area is None else area
        if kind == "photos":  # the surface on the coarse grid; the map cells sample the photos
            cw, ch = int(np.ceil((east - west) / COARSE)), int(np.ceil((north - south) / COARSE))
            _, coverage, surface = _mesh_with_fill(scene, parts, west, north, cw, ch, COARSE)
            width, height = _grid(west, east, south, north, resolution, f"{len(scene.image_files())} photos on the mesh surface")
            rgb, depth = render_photos(scene, surface, west, north, width, height, resolution, photos)
            coverage = np.isfinite(rgb[..., 0]).astype(np.float32)
            rgb = np.nan_to_num(rgb)
        else:
            width, height = _grid(west, east, south, north, resolution, f"{sum(len(part.faces) for part in parts):,} textured faces")
            rgb, coverage, depth, filled = _mesh_with_fill(scene, parts, west, north, width, height, resolution, colour=True)
    else:
        source = scene.splat(source)
        print(f"Reading {source}", flush=True)
        _, data = ply.read(source)
        splat = {name: data[name] for name in data.dtype.names}
        west, east, south, north = extent(splat)
        width, height = _grid(west, east, south, north, resolution, f"{len(splat['x']):,} splats")
        rgb, coverage, depth = render(splat, west, north, width, height, resolution)
    staged = scene.staging("map")
    origin = np.array(record["origin"], dtype=np.float64)
    corner = (origin[0] + west, origin[1] + north)  # the north-west corner in the CRS
    alpha = np.where(coverage > 0.5, 255, 0).astype(np.uint8)
    ortho = np.concatenate([np.round(rgb * 255).astype(np.uint8), alpha[..., None]], 2)
    dsm = (origin[2] - depth).astype(np.float32)  # NaN where less than half covered
    _write_geotiff(staged / "ortho.tif", ortho, corner, resolution, record["crs"])
    _write_geotiff(staged / "dsm.tif", dsm, corner, resolution, record["crs"])
    bounds, _ = _web_overlay(staged / "ortho.tif", staged / "ortho-web.webp")
    _, heights = _web_overlay(staged / "dsm.tif", staged / "dsm-web.webp")
    cameras = [list(position[:2]) for position in _camera_gps(scene)]
    (staged / "map.json").write_text(json.dumps({
        "crs": record["crs"], "resolution_m": resolution, "north_west": list(corner), "size": [width, height],
        "source": str(source), "kind": kind, "filled_from_splat": filled, "bounds_latlon": bounds, "dsm_range_m": heights,
    }, indent=1))
    (staged / "map.html").write_text(_html(bounds, cameras))
    run = scene.new_run("map")
    replace(staged, run)
    print(f"Orthophoto and DSM in {run}; open {run / 'map.html'}")
    return run


def _crop_area(scene: Scene) -> tuple[float, float, float, float] | None:
    """West, east, south, north of the crop box's corners in model coordinates, or None without a crop."""
    import itertools

    import numpy as np

    from . import crop as cropping

    box = cropping.load(scene)
    if box is None:
        return None
    corners = np.array(list(itertools.product(*zip(box.min, box.max, strict=True))))
    model = corners @ cropping.level_rotation(box.up)  # level = R · model, so model = Rᵀ · level
    return float(model[:, 0].min()), float(model[:, 0].max()), float(model[:, 2].min()), float(model[:, 2].max())


def _grid(west: float, east: float, south: float, north: float, resolution: float, what: str) -> tuple[int, int]:
    import numpy as np

    width, height = int(np.ceil((east - west) / resolution)), int(np.ceil((north - south) / resolution))
    print(f"Rendering {width}×{height} cells of {resolution * 100:g} cm ({east - west:.0f} × {north - south:.0f} m) "
          f"from {what}", flush=True)
    return width, height


def _write_geotiff(path: Path, data: Any, corner: tuple[float, float], resolution: float, crs: str) -> None:
    """A Cloud Optimized GeoTIFF: bands last in `data`, uint8 RGBA or float32 with NaN as no data."""
    import numpy as np
    from rasterio.enums import ColorInterp
    from rasterio.io import MemoryFile
    from rasterio.shutil import copy
    from rasterio.transform import from_origin

    bands = data.reshape(data.shape[0], data.shape[1], -1)
    profile = {
        "driver": "GTiff", "width": bands.shape[1], "height": bands.shape[0], "count": bands.shape[2],
        "dtype": str(bands.dtype), "crs": crs, "transform": from_origin(corner[0], corner[1], resolution, resolution),
    }
    if bands.dtype == np.float32:
        profile["nodata"] = float("nan")
    with MemoryFile() as memory:
        with memory.open(**profile) as dataset:
            dataset.write(np.moveaxis(bands, 2, 0))
            if bands.shape[2] == 4:
                dataset.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
        copy(memory.name, str(path), driver="COG", compress="DEFLATE")


def _web_overlay(raster: Path, image: Path, size: int = 4096) -> tuple[list[list[float]], list[float] | None]:
    """A GeoTIFF in Web Mercator as WebP (long edge `size`) for a web map, and its [[south, west], [north, east]] in degrees.

    RGBA rasters (the orthophoto) are reprojected as they are; a one-band DSM is coloured by height and hill-shaded,
    and its colour range (2nd to 98th percentile of height, metres) is returned as the second value.
    """
    import numpy as np
    import rasterio
    from PIL import Image
    from pyproj import Transformer
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    with rasterio.open(raster) as source:
        transform, cols, rows = calculate_default_transform(source.crs, "EPSG:3857", source.width, source.height, *source.bounds)
        assert cols is not None and rows is not None  # always set when the source size is given
        width, height = int(cols), int(rows)
        scale = max(width, height) / size
        if scale > 1:
            width, height = int(width / scale), int(height / scale)
            transform = transform * transform.scale(scale)
        bands = list(range(1, source.count + 1))
        out = np.full((len(bands), height, width), np.nan if source.count == 1 else 0, dtype=source.dtypes[0])
        reproject(
            rasterio.band(source, bands), out, dst_transform=transform, dst_crs="EPSG:3857",
            resampling=Resampling.average, dst_nodata=np.nan if source.count == 1 else None,
        )
    west, north = transform * (0, 0)
    east, south = transform * (width, height)
    lon, lat = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform([west, east], [south, north])
    heights = None
    if source.count == 1:
        pixel = transform.a * np.cos(np.radians((lat[0] + lat[1]) / 2))  # Web Mercator metres → ground metres
        rgba, heights = _shade(out[0], pixel)
    else:
        rgba = np.moveaxis(out, 0, 2)
    Image.fromarray(rgba, "RGBA").save(image, quality=85)  # 3 MB; a PNG was 20 MB
    return [[lat[0], lon[0]], [lat[1], lon[1]]], heights


# Low to high: the monitor's palette (deep, solve, lod, frames, train).
RAMP = ((23, 21, 32), (54, 249, 246), (114, 241, 184), (254, 222, 93), (255, 126, 219))


def _shade(z: Any, pixel: float) -> tuple[Any, list[float]]:
    """Heights as RGBA: colour from RAMP over the 2-98 % range, times a hill shade lit from the north-west."""
    import numpy as np

    valid = np.isfinite(z)
    if not valid.any():
        return np.zeros((*z.shape, 4), dtype=np.uint8), [0.0, 0.0]
    lo, hi = (float(v) for v in np.percentile(z[valid], [2, 98]))
    filled = np.where(valid, z, np.median(z[valid]))
    gy, gx = np.gradient(filled, pixel)  # rows run south
    shade = np.clip(0.65 + 0.35 * (gy - gx) / np.sqrt(1 + gx**2 + gy**2) * 2, 0.25, 1)
    t = np.clip((filled - lo) / max(hi - lo, 1e-6), 0, 1) * (len(RAMP) - 1)
    i = np.minimum(t.astype(int), len(RAMP) - 2)
    ramp = np.array(RAMP, dtype=np.float32)
    colour = ramp[i] + (ramp[i + 1] - ramp[i]) * (t - i)[..., None]
    rgba = np.dstack([colour * shade[..., None], np.where(valid, 255, 0)]).astype(np.uint8)
    return rgba, [round(lo, 2), round(hi, 2)]


def _camera_gps(scene: Scene) -> list[tuple[float, float, float]]:
    from . import geo

    return list(geo.image_positions(scene.image_files()).values())


def _html(bounds: list[list[float]], cameras: list[list[float]]) -> str:
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>gausser map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{height:100%;margin:0}#ctl{position:absolute;z-index:1000;top:10px;left:50px;background:#241b2f;color:#f0eff1;
font:13px mononoki,Menlo,monospace;padding:6px 10px;border-radius:4px}</style></head>
<body><div id="map"></div><div id="ctl">orthophoto <input id="op" type="range" min="0" max="1" step="0.05" value="1"></div>
<script>
const bounds = BOUNDS, cameras = CAMERAS;
const map = L.map("map", {maxZoom: 23});
const osm = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {maxNativeZoom: 19, maxZoom: 23, attribution: "© OpenStreetMap"});
const sat = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  {maxNativeZoom: 19, maxZoom: 23, attribution: "Esri World Imagery"}).addTo(map);
const ortho = L.imageOverlay("ortho-web.webp", bounds).addTo(map);
const cams = L.layerGroup(cameras.map(c => L.circleMarker(c, {radius: 3, color: "#ff7edb", weight: 1})));
L.control.layers({"Satellite": sat, "OpenStreetMap": osm}, {"Orthophoto": ortho, "Cameras (GPS)": cams}).addTo(map);
map.fitBounds(bounds);
document.getElementById("op").oninput = e => ortho.setOpacity(+e.target.value);
</script></body></html>
""".replace("BOUNDS", json.dumps(bounds)).replace("CAMERAS", json.dumps(cameras))
