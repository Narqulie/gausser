"""Kandao QooCam 8K raw video: two fisheye streams plus the camera's own lens calibration.

The raw .MOV holds two 3840×3840 HEVC streams (lens 0 and lens 1) and a top-level ``kfix`` box
whose ``LENS`` entry is one hex line per lens of little-endian float32: 6 header values, f, cx,
cy (lens 1 in side-by-side pixels), equidistant fisheye k1..k4, R (3×3, row-major) and t (cm),
with x_cam = R·X + t. The stored frames are the calibration images flipped vertically, so
cy becomes 3839 − cy and x_cam = M·(R·X + t), M = diag(1, −1, 1). Verified on a real clip by
SIFT matches between the two lenses where they overlap: the calibration then agrees to 0.2°.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

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


def calibration(path: Path) -> dict[str, Any] | None:
    """The lens rig of a QooCam raw video in `pano.lenses` form (rig = lens 0's frame), or None."""
    import numpy as np

    if path.suffix.lower() not in (".mov", ".mp4"):
        return None
    kfix = _box(path, b"kfix")
    if kfix is None or (at := kfix.find(b"LENS\0")) < 0:
        return None
    count, length = struct.unpack("<II", kfix[at + 40 : at + ENTRY])
    text = kfix[at + ENTRY : at + ENTRY + count * length].split(b"\0")[0].decode("ascii", "replace")
    flip = np.diag([1.0, -1.0, 1.0])
    raw = []
    for index, line in enumerate(line for line in text.split("\r\n") if line.strip()):
        v = _lens_floats(line)
        f, cx, cy, k = v[6], v[7] - index * SIDE, v[8], v[9:13]
        rotation = flip @ np.array(v[13:22]).reshape(3, 3)
        translation = flip @ np.array(v[22:25]) / 100  # cm → m
        raw.append((f, cx, SIDE - 1 - cy, k, rotation, translation))
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
