"""GPS of photos and drone video frames.

DJI drones write a subtitle file next to each video when "Video Subtitles" is on in DJI Fly
(<video stem>.SRT): one block per frame with its time in the video and the drone's position, e.g.
``[latitude: 61.498123] [longitude: 23.763456] [rel_alt: 50.000 abs_alt: 160.123]``. `frames` reads
it and writes each extracted frame's position into the frame's EXIF GPS tags, so video frames carry
their position the same way drone photos do, and COLMAP reads it from there.

`solve` georeferences a model whose images have GPS: it fits a similarity from the COLMAP camera centres
to their GPS positions in a projected CRS (EPSG:3067 in Finland, else the UTM zone) and moves the model
there. Coordinates stay small and in gausser's axes: x = east, y = down, z = north, in metres from the
origin in sparse/geo.json, so (E, N, H) = origin + (x, z, -y).
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from pathlib import Path

from typing import Any

from .scene import GausserError

_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->")
_LAT = re.compile(r"latitude\s*:\s*(-?[\d.]+)", re.I)
_LON = re.compile(r"longitude\s*:\s*(-?[\d.]+)", re.I)
_ABS = re.compile(r"abs_alt\s*:\s*(-?[\d.]+)", re.I)
_REL = re.compile(r"rel_alt\s*:\s*(-?[\d.]+)", re.I)
MAX_TILT = 10.0  # degrees between the GPS fit's up and the cameras' before georeference keeps the cameras'
_GPS = re.compile(r"GPS\s*\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)")  # older DJI: GPS(lon, lat, alt)


@dataclass
class Track:
    """A video's positions: time in seconds from the video's start, degrees, metres."""

    source: Path
    times: list[float]
    points: list[tuple[float, float, float]]  # latitude, longitude, altitude

    def at(self, time: float) -> tuple[float, float, float]:
        """The position at `time`, linear between samples, held at the ends."""
        i = bisect.bisect_left(self.times, time)
        if i == 0:
            return self.points[0]
        if i == len(self.times):
            return self.points[-1]
        t0, t1 = self.times[i - 1], self.times[i]
        w = (time - t0) / (t1 - t0) if t1 > t0 else 0.0
        a, b = self.points[i - 1], self.points[i]
        return (a[0] + w * (b[0] - a[0]), a[1] + w * (b[1] - a[1]), a[2] + w * (b[2] - a[2]))


def parse_srt(text: str, source: Path) -> Track | None:
    """The positions of a DJI subtitle file; blocks without a GPS fix (0, 0) are skipped."""
    times: list[float] = []
    points: list[tuple[float, float, float]] = []
    for block in re.split(r"\n\s*\n", text.replace("\r", "")):
        start = _TIME.search(block)
        lat, lon, alt = _LAT.search(block), _LON.search(block), _ABS.search(block) or _REL.search(block)
        if start and lat and lon:
            point = (float(lat[1]), float(lon[1]), float(alt[1]) if alt else 0.0)
        elif start and (gps := _GPS.search(block)):
            point = (float(gps[2]), float(gps[1]), float(gps[3]))
        else:
            continue
        if point[0] == 0 and point[1] == 0:
            continue
        h, m, s, fraction = start.groups()
        time = int(h) * 3600 + int(m) * 60 + int(s) + int(fraction) / 10 ** len(fraction)
        if times and time <= times[-1]:
            continue
        times.append(time)
        points.append(point)
    return Track(source, times, points) if times else None


def track(video: Path) -> Track | None:
    """The GPS track of a DJI video from its subtitle file, if there is one."""
    for subtitles in (video.with_suffix(".SRT"), video.with_suffix(".srt")):
        if subtitles.is_file() and not subtitles.name.startswith("._"):
            return parse_srt(subtitles.read_text(errors="replace"), subtitles)
    return None


def _dms(value: float) -> tuple[float, float, float]:
    value = abs(value)
    degrees = int(value)
    minutes = int((value - degrees) * 60)
    return float(degrees), float(minutes), round((value - degrees - minutes / 60) * 3600, 4)


def add_gps(jpeg: Path, latitude: float, longitude: float, altitude: float) -> None:
    """Give a JPEG without EXIF (ffmpeg's frames) EXIF GPS tags, without re-encoding its pixels."""
    from PIL import ExifTags, Image

    exif = Image.Exif()
    exif.get_ifd(ExifTags.IFD.GPSInfo).update({
        ExifTags.GPS.GPSLatitudeRef: "N" if latitude >= 0 else "S",
        ExifTags.GPS.GPSLatitude: _dms(latitude),
        ExifTags.GPS.GPSLongitudeRef: "E" if longitude >= 0 else "W",
        ExifTags.GPS.GPSLongitude: _dms(longitude),
        ExifTags.GPS.GPSAltitudeRef: 0 if altitude >= 0 else 1,
        ExifTags.GPS.GPSAltitude: round(abs(altitude), 3),
    })
    payload = exif.tobytes()  # b"Exif\0\0" + TIFF
    data = jpeg.read_bytes()
    if data[:2] != b"\xff\xd8":
        raise GausserError(f"{jpeg.name} is not a JPEG")
    jpeg.write_bytes(data[:2] + b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload + data[2:])


def crs_for(latitude: float, longitude: float) -> str:
    """EPSG:3067 (ETRS-TM35FIN) in Finland, else the WGS 84 UTM zone."""
    if 59 <= latitude <= 70.5 and 19 <= longitude <= 32:
        return "EPSG:3067"
    zone = int((longitude + 180) // 6) % 60 + 1
    return f"EPSG:{32600 + zone if latitude >= 0 else 32700 + zone}"


def project(points: Any, crs: str) -> Any:
    """(latitude, longitude, altitude) rows → (east, north, height) in `crs`; altitude passes through."""
    import numpy as np
    from pyproj import Transformer

    points = np.asarray(points, dtype=np.float64)
    east, north = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(points[:, 1], points[:, 0])
    return np.stack([east, north, points[:, 2]], 1)


def to_local(enh: Any, origin: Any) -> Any:
    """(east, north, height) → gausser's local axes (x east, y down, z north)."""
    import numpy as np

    d = np.asarray(enh, dtype=np.float64) - origin
    return np.stack([d[:, 0], -d[:, 2], d[:, 1]], 1)


def similarity(source: Any, target: Any) -> tuple[float, Any, Any]:
    """Umeyama: scale s, rotation R, translation t minimising |s·R·source + t − target|."""
    import numpy as np

    mu_s, mu_t = source.mean(0), target.mean(0)
    a, b = source - mu_s, target - mu_t
    u, sigma, vt = np.linalg.svd(np.einsum("ni,nj->ij", b, a) / len(a))
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(u) * np.linalg.det(vt))])
    r = np.einsum("ij,jk,kl->il", u, d, vt)
    s = float(np.trace(np.diag(sigma) @ d) / a.var(0).sum())
    return s, r, mu_t - s * np.einsum("ij,j->i", r, mu_s)


def level_similarity(source: Any, target: Any, up: Any) -> tuple[float, Any, Any]:
    """Similarity that turns the model's `up` onto gravity (-y) and fits only heading, scale and position."""
    import numpy as np

    from .crop import level_rotation

    level = level_rotation(-np.asarray(up, dtype=float))  # takes up onto -y
    a = np.einsum("ij,nj->ni", level, source)
    a0, b0 = a - a.mean(0), target - target.mean(0)
    cross = float((a0[:, 0] * b0[:, 2] - a0[:, 2] * b0[:, 0]).sum())
    dot = float((a0[:, 0] * b0[:, 0] + a0[:, 2] * b0[:, 2]).sum())
    c, s_ = np.cos(np.arctan2(cross, dot)), np.sin(np.arctan2(cross, dot))
    r = np.array([[c, 0.0, -s_], [0.0, 1.0, 0.0], [s_, 0.0, c]]) @ level
    s = float(np.hypot(cross, dot) / (a0[:, [0, 2]] ** 2).sum())
    return s, r, target.mean(0) - s * np.einsum("ij,j->i", r, source.mean(0))


def georeference(
    images_bin: Path, positions: dict[str, tuple[float, float, float]], up: Any = None,
) -> tuple[str, dict[str, Any], str]:
    """`colmap model_transformer` similarity from a model onto its images' GPS, the geo.json record, and a log line.

    Images whose GPS disagrees with the model by more than 3× the median (or 1 m) are left out of the fit,
    twice over, so a few bad fixes do not tilt the model. When the full fit tilts the model's `up` (from its
    cameras) more than MAX_TILT off gravity, the fit keeps that up and takes only heading, scale and position
    from GPS: a phone's GPS along a walk cannot fix the roll about the walk (37° off on a 14 × 2 m walk).
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    from .orbit import read_cameras

    centres, _, _, names = read_cameras(images_bin)
    pairs = [(centre, positions[name]) for centre, name in zip(centres, names, strict=True) if name in positions]
    if len(pairs) < 3:
        raise GausserError(f"Only {len(pairs)} solved images have GPS; cannot georeference")
    source = np.array([pair[0] for pair in pairs])
    gps = np.array([pair[1] for pair in pairs])
    crs = crs_for(*gps.mean(0)[:2])
    enh = project(gps, crs)
    origin = np.round(enh.mean(0))
    target = to_local(enh, origin)
    if np.ptp(target[:, [0, 2]], axis=0).max() < 5:
        raise GausserError("The GPS positions span less than 5 m; too little to georeference")

    def robust(fit: Any) -> tuple[float, Any, Any, Any]:
        keep = np.ones(len(source), dtype=bool)
        s, r, t = fit(source, target)
        for _ in range(2):
            error = np.linalg.norm(s * np.einsum("ij,nj->ni", r, source) + t - target, axis=1)
            keep = error <= max(1.0, 3 * float(np.median(error)))
            s, r, t = fit(source[keep], target[keep])
        return s, r, t, keep

    s, r, t, keep = robust(similarity)
    tilt, up_source = None, "gps"
    if up is not None:
        fitted = np.einsum("ij,j->i", r, up) / np.linalg.norm(up)
        tilt = float(np.degrees(np.arccos(np.clip(-fitted[1], -1.0, 1.0))))
        if tilt > MAX_TILT:
            s, r, t, keep = robust(lambda a, b: level_similarity(a, b, up))
            up_source = "cameras"
    residual = s * np.einsum("ij,nj->ni", r, source[keep]) + t - target[keep]
    horizontal = float(np.sqrt(np.mean(residual[:, 0] ** 2 + residual[:, 2] ** 2)))
    vertical = float(np.sqrt(np.mean(residual[:, 1] ** 2)))
    x, y, z, w = Rotation.from_matrix(r).as_quat()
    text = f"{s:.12g} {w:.12g} {x:.12g} {y:.12g} {z:.12g} {t[0]:.12g} {t[1]:.12g} {t[2]:.12g}\n"
    record = {
        "crs": crs, "origin": origin.tolist(), "axes": "x east, y down, z north; metres from origin (east, north, height)",
        "images": int(keep.sum()), "rms_horizontal_m": round(horizontal, 3), "rms_vertical_m": round(vertical, 3),
        "up": up_source, "gps_tilt_deg": None if tilt is None else round(tilt, 1),
    }
    note = (
        f"{crs}, {int(keep.sum())}/{len(pairs)} cameras within {horizontal:.2f} m horizontally and "
        f"{vertical:.2f} m vertically RMS of their GPS"
    )
    if up_source == "cameras":
        note += f"; kept the cameras' up (GPS alone tilted the model {tilt:.0f}°)"
    return text, record, note


def image_positions(images: list[Path]) -> dict[str, tuple[float, float, float]]:
    """The GPS of every image that has it, by file name."""
    found = {image.name: read_gps(image) for image in images}
    return {name: position for name, position in found.items() if position}


def from_drone(image: Path) -> bool:
    """A DJI photo, or a video frame `frames` gave GPS from a DJI .SRT (no EXIF Make): not a phone photo."""
    from PIL import ExifTags, Image

    with Image.open(image) as opened:
        return str(opened.getexif().get(ExifTags.Base.Make, "DJI")).strip().upper().startswith("DJI")


def read_gps(image: Path) -> tuple[float, float, float] | None:
    """Latitude, longitude and altitude from an image's EXIF GPS tags, if it has them."""
    from PIL import ExifTags, Image

    try:
        with Image.open(image) as opened:
            gps = opened.getexif().get_ifd(ExifTags.IFD.GPSInfo)
    except OSError:
        return None
    try:
        lat = sum(float(v) / 60**i for i, v in enumerate(gps[ExifTags.GPS.GPSLatitude]))
        lon = sum(float(v) / 60**i for i, v in enumerate(gps[ExifTags.GPS.GPSLongitude]))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if gps.get(ExifTags.GPS.GPSLatitudeRef) == "S":
        lat = -lat
    if gps.get(ExifTags.GPS.GPSLongitudeRef) == "W":
        lon = -lon
    alt = float(gps.get(ExifTags.GPS.GPSAltitude, 0.0))
    if gps.get(ExifTags.GPS.GPSAltitudeRef) in (1, b"\x01"):
        alt = -alt
    return (lat, lon, alt) if (lat, lon) != (0.0, 0.0) else None
