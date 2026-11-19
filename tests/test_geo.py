from pathlib import Path

import pytest
from PIL import Image

from gausser import geo

# Two blocks as a DJI Mini 4 Pro writes them (DJI_20261114103819_0003_D.SRT), plus one without a fix.
SRT = """1
00:00:00,000 --> 00:00:00,033
<font size="28">FrameCnt: 1, DiffTime: 33ms
2026-11-14 10:38:21.018
[iso: 110] [shutter: 1/1000.0] [fnum: 1.7] [ev: -0.3] [color_md: default] [focal_len: 24.00] [latitude: 61.522042] [longitude: 23.688751] [rel_alt: 49.600 abs_alt: 352.428] [ct: 5769] </font>

2
00:00:01,000 --> 00:00:01,033
<font size="28">FrameCnt: 31, DiffTime: 33ms
2026-11-14 10:38:22.018
[iso: 110] [shutter: 1/1000.0] [fnum: 1.7] [ev: -0.3] [color_md: default] [focal_len: 24.00] [latitude: 61.522142] [longitude: 23.688951] [rel_alt: 49.600 abs_alt: 354.428] [ct: 5769] </font>

3
00:00:02,000 --> 00:00:02,033
<font size="28">[latitude: 0.000000] [longitude: 0.000000] [rel_alt: 0.000 abs_alt: 0.000]</font>
"""


def test_a_dji_subtitle_file_gives_positions_by_video_time():
    track = geo.parse_srt(SRT, Path("x.SRT"))
    assert track is not None
    assert track.times == [0.0, 1.0]  # the block without a fix is skipped
    assert track.at(0.5) == pytest.approx((61.522092, 23.688851, 353.428))
    assert track.at(-1) == track.points[0] and track.at(9) == track.points[-1]


def test_gps_written_into_a_frame_reads_back_and_leaves_the_pixels(tmp_path):
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (32, 24), (200, 40, 10)).save(frame)
    before = frame.read_bytes()
    geo.add_gps(frame, 61.5220423, -23.6887514, 352.428)
    lat, lon, alt = geo.read_gps(frame)  # type: ignore[misc]
    assert (lat, lon, alt) == pytest.approx((61.5220423, -23.6887514, 352.428), abs=1e-7)
    assert frame.read_bytes().endswith(before[2:])  # the JPEG data itself is unchanged


def test_an_image_without_gps_has_none(tmp_path):
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (8, 8)).save(frame)
    assert geo.read_gps(frame) is None


def test_georeference_recovers_a_similarity_onto_gps_and_ignores_a_bad_fix(tmp_path, monkeypatch):
    import numpy as np
    from scipy.spatial.transform import Rotation

    # Cameras on a 3×4 grid 50 m up, in metres east/north of a point near Tampere.
    rng = np.random.default_rng(1)
    grid = np.array([(e, n) for e in (0, 30, 60) for n in (0, 25, 50, 75)], dtype=float)
    crs = geo.crs_for(61.52, 23.69)
    assert crs == "EPSG:3067"
    from pyproj import Transformer

    to_gps = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    e0, n0 = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(23.69, 61.52)
    lon, lat = to_gps.transform(e0 + grid[:, 0], n0 + grid[:, 1])
    positions = {f"f{i}.jpg": (lat[i], lon[i], 350.0 + rng.normal(0, 0.05)) for i in range(len(grid))}
    positions["f3.jpg"] = (lat[3] + 0.001, lon[3], 350.0)  # a GPS jump of about 110 m
    # The model: the same layout in COLMAP's arbitrary frame (rotated, scaled 1/20, shifted).
    true = np.stack([grid[:, 0], np.full(len(grid), -50.0), grid[:, 1]], 1)
    r = Rotation.from_euler("xyz", [20, -35, 70], degrees=True).as_matrix()
    model = (true - true.mean(0)) @ r.T / 20 + [3, -1, 2]
    monkeypatch.setattr("gausser.orbit.read_cameras", lambda _: (model, None, None, list(positions)))
    text, record, _ = geo.georeference(tmp_path / "images.bin", positions)
    s, qw, qx, qy, qz, *t = map(float, text.split())
    rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    moved = s * model @ rot.T + t
    local = geo.to_local(geo.project(np.array([positions[f"f{i}.jpg"] for i in range(len(grid))]), crs), record["origin"])
    good = [i for i in range(len(grid)) if i != 3]
    assert record["crs"] == "EPSG:3067" and record["images"] == len(grid) - 1
    assert np.abs(moved[good] - local[good]).max() < 0.3
    assert s == pytest.approx(20, rel=0.01)


def test_georeference_keeps_the_cameras_up_when_phone_gps_along_a_walk_would_tilt_the_model(tmp_path, monkeypatch):
    """Handheld photos on two passes 3 m apart along a market hut (37° tilt on the real set): the phone's GPS
    put both passes on one line, the second 1.5 m higher, and nothing else fixes the roll about the walk."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(2)
    crs = geo.crs_for(61.498, 23.76)
    from pyproj import Transformer

    e0, n0 = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(23.76, 61.498)
    along = np.r_[np.linspace(0, 14, 16), np.linspace(14, 2, 15)]
    second = np.arange(31) >= 16
    walk = np.c_[along, np.where(second, 3.0, 0.0)]  # metres east, north
    gps = np.c_[along, np.zeros(31)] + rng.normal(0, 0.3, (31, 2))
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(e0 + gps[:, 0], n0 + gps[:, 1])
    positions = {f"f{i}.jpg": (lat[i], lon[i], 90.0 + 1.5 * second[i] + rng.normal(0, 0.3)) for i in range(31)}
    true = np.stack([walk[:, 0], np.full(31, -1.5), walk[:, 1]], 1)  # x east, y down, z north
    r = Rotation.from_euler("xyz", [20, -35, 70], degrees=True).as_matrix()
    model = (true - true.mean(0)) @ r.T / 20 + [3, -1, 2]
    monkeypatch.setattr("gausser.orbit.read_cameras", lambda _: (model, None, None, list(positions)))

    text, record, note = geo.georeference(tmp_path / "images.bin", positions, up=r @ [0, -1.0, 0])

    s, qw, qx, qy, qz, *_ = map(float, text.split())
    up = Rotation.from_quat([qx, qy, qz, qw]).as_matrix() @ r @ [0, -1.0, 0]
    assert record["up"] == "cameras" and record["gps_tilt_deg"] > 10 and "kept the cameras' up" in note
    assert up == pytest.approx([0, -1.0, 0], abs=1e-9)
    assert s == pytest.approx(20, rel=0.15)  # the GPS merged the passes, so the walk looks narrower
