"""Kandao QooCam 8K raw video: two fisheye streams plus the camera's own lens calibration.

The raw .MOV holds two 3840×3840 HEVC streams (lens 0 and lens 1) and a top-level ``kfix`` box
whose ``LENS`` entry is one hex line per lens of little-endian float32: 6 header values, f, cx,
cy (lens 1 in side-by-side pixels), equidistant fisheye k1..k4, R (3×3, row-major) and t (cm),
with x_cam = R·X + t. The camera stores every lens image (video and photo) flipped vertically,
i.e. mirrored, relative to that calibration: the two lenses agree to 0.2° where they overlap only
with the flip (SIFT matches on a real clip), and splats built from the stored images came out
mirrored. `frames` therefore flips each lens image back on ingest and uses the calibration as is.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any

from .scene import GausserError

SIDE = 3840  # pixels per lens image
ENTRY = 48  # kfix/kvar entry header: 32-byte name, 8-byte type, u32 count, u32 size
# The body's grip end, in the rig frame (lens 0's camera frame): the operator's hand, or a stick,
# is at the right edge of lens 0 and the left edge of lens 1.
GRIP = (1.0, 0.0, 0.0)


def _box(path: Path, name: bytes) -> bytes | None:
    """The payload of the top-level MP4 box ``name``, or None."""
    with path.open("rb") as f:
        size = f.seek(0, 2)
        offset = 0
        while offset + 8 <= size:
            f.seek(offset)
            length, kind = struct.unpack(">I4s", f.read(8))
            header = 8
            if length == 1:
                (length,) = struct.unpack(">Q", f.read(8))
                header = 16
            elif length == 0:
                length = size - offset
            if length < header:
                return None
            if kind == name:
                return f.read(length - header)
            offset += length
    return None


def _lens_floats(line: str) -> list[float]:
    """The float32 values of one LENS line: a short prefix ("=1" or "2"), then hex."""
    for skip in (1, 2):
        try:
            data = bytes.fromhex(line[skip:][: (len(line) - skip) // 8 * 8])
        except ValueError:
            continue
        values = list(struct.unpack(f"<{len(data) // 4}f", data))
        if len(values) >= 25 and 500 < values[6] < 3000:  # f is the 7th value
            return values
    raise ValueError(f"unreadable LENS line {line[:20]!r}")


def _lens_text(path: Path) -> str | None:
    """The LENS calibration text of a raw video (kfix box) or a photo (``LENS=`` in its JPEG header)."""
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        with path.open("rb") as f:
            header = f.read(1 << 16)
        match = re.search(rb"QOOCAM 8K.{0,200}?LENS=([0-9a-fA-F=\r\n]+)", header, re.S)
        return match.group(1).decode() if match else None
    if suffix not in (".mov", ".mp4"):
        return None
    kfix = _box(path, b"kfix")
    if kfix is None or (at := kfix.find(b"LENS\0")) < 0:
        return None
    count, length = struct.unpack("<II", kfix[at + 40 : at + ENTRY])
    return kfix[at + ENTRY : at + ENTRY + count * length].split(b"\0")[0].decode("ascii", "replace")


def side_by_side(photo: Path, out: Path) -> None:
    """A QooCam 8K photo (lens 0 above lens 1, 3840×7680) as the video's side-by-side 7680×3840 frame.

    The halves are the same lens images as the video streams: matching the two lenses where they
    overlap fits best unturned, and the grip shows at lens 0's right edge as in video.
    """
    from PIL import Image

    with Image.open(photo) as image:
        side = image.width
        if image.height != 2 * side:
            raise GausserError(f"{photo.name} is {image.width}x{image.height}, not two lens images one above the other")
        frame = Image.new("RGB", (2 * side, side))
        for index in (0, 1):  # un-mirror each lens image (see the module docstring)
            lens = image.crop((0, index * side, side, (index + 1) * side)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            frame.paste(lens, (index * side, 0))
    frame.save(out, quality=95)


def calibration(path: Path) -> dict[str, Any] | None:
    """The lens rig of a QooCam raw video or photo in `pano.lenses` form (rig = lens 0's frame), or None.

    Photos carry the same LENS lines as videos, with lens 1's cx also in side-by-side pixels.
    """
    import numpy as np

    text = _lens_text(path)
    if text is None:
        return None
    raw = []
    for index, line in enumerate(line for line in text.split("\r\n") if line.strip()):
        v = _lens_floats(line)
        f, cx, cy, k = v[6], v[7] - index * SIDE, v[8], v[9:13]
        rotation = np.array(v[13:22]).reshape(3, 3)
        translation = np.array(v[22:25]) / 100  # cm → m
        raw.append((f, cx, cy, k, rotation, translation))
    if len(raw) != 2:
        return None
    # Express every lens relative to lens 0: lens_from_rig = lens_from_world · (lens0_from_world)⁻¹.
    r0, t0 = raw[0][4], raw[0][5]
    lenses = []
    for index, (f, cx, cy, k, rotation, translation) in enumerate(raw):
        relative = rotation @ r0.T
        lenses.append({
            "x0": index * SIDE, "size": SIDE, "f": f, "c": [cx, cy], "k": list(k), "max_deg": 97.0,
            "rotation": relative.tolist(), "translation": (translation - relative @ t0).tolist(),
        })
    return {"camera": "QooCam 8K", "lenses": lenses, "grip": list(GRIP)}
