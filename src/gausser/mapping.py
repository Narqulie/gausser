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


def make(scene: Scene, source: Path | None = None, resolution: float = 0.03) -> Path:
    """Write runs/map/<time>/: ortho.tif, dsm.tif (heights in the CRS's vertical frame of the GPS) and map.html."""
    import numpy as np

    record = scene.geo()
    if record is None:
        raise GausserError("The model is not georeferenced (no sparse/geo.json); solve photos or video frames with GPS")
    source = scene.splat(source)
    print(f"Reading {source}", flush=True)
    _, data = ply.read(source)
    splat = {name: data[name] for name in data.dtype.names}
    west, east, south, north = extent(splat)
    width, height = int(np.ceil((east - west) / resolution)), int(np.ceil((north - south) / resolution))
    print(f"Rendering {width}×{height} cells of {resolution * 100:g} cm ({east - west:.0f} × {north - south:.0f} m) "
          f"from {len(splat['x']):,} splats", flush=True)
    rgb, coverage, depth = render(splat, west, north, width, height, resolution)
    staged = scene.staging("map")
    origin = np.array(record["origin"], dtype=np.float64)
    corner = (origin[0] + west, origin[1] + north)  # the north-west corner in the CRS
    alpha = np.where(coverage > 0.5, 255, 0).astype(np.uint8)
    ortho = np.concatenate([np.round(rgb * 255).astype(np.uint8), alpha[..., None]], 2)
    dsm = (origin[2] - depth).astype(np.float32)  # NaN where less than half covered
    _write_geotiff(staged / "ortho.tif", ortho, corner, resolution, record["crs"])
    _write_geotiff(staged / "dsm.tif", dsm, corner, resolution, record["crs"])
    bounds = _web_overlay(staged / "ortho.tif", staged / "ortho-web.webp")
    cameras = [list(position[:2]) for position in _camera_gps(scene)]
    (staged / "map.json").write_text(json.dumps({
        "crs": record["crs"], "resolution_m": resolution, "north_west": list(corner), "size": [width, height],
        "source": str(source), "bounds_latlon": bounds,
    }, indent=1))
    (staged / "map.html").write_text(_html(bounds, cameras))
    run = scene.new_run("map")
    replace(staged, run)
    print(f"Orthophoto and DSM in {run}; open {run / 'map.html'}")
    return run


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


def _web_overlay(ortho: Path, image: Path, size: int = 4096) -> list[list[float]]:
    """The orthophoto in Web Mercator as WebP (long edge `size`), and its [[south, west], [north, east]] in degrees."""
    import numpy as np
    import rasterio
    from PIL import Image
    from pyproj import Transformer
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    with rasterio.open(ortho) as source:
        transform, cols, rows = calculate_default_transform(source.crs, "EPSG:3857", source.width, source.height, *source.bounds)
        assert cols is not None and rows is not None  # always set when the source size is given
        width, height = int(cols), int(rows)
        scale = max(width, height) / size
        if scale > 1:
            width, height = int(width / scale), int(height / scale)
            transform = transform * transform.scale(scale)
        out = np.zeros((4, height, width), dtype=np.uint8)
        reproject(rasterio.band(source, [1, 2, 3, 4]), out, dst_transform=transform, dst_crs="EPSG:3857", resampling=Resampling.average)
    Image.fromarray(np.moveaxis(out, 0, 2), "RGBA").save(image, quality=85)  # 3 MB; a PNG was 20 MB
    west, north = transform * (0, 0)
    east, south = transform * (width, height)
    lon, lat = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform([west, east], [south, north])
    return [[lat[0], lon[0]], [lat[1], lon[1]]]


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
