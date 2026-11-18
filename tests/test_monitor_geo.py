import numpy as np
import pytest
from pyproj import Transformer
from scipy.spatial.transform import Rotation

from gausser import monitor
from test_arkit import write_model

RECORD = {"crs": "EPSG:3067", "origin": [323957.0, 6825418.0, 417.0]}


def camera(r_world_to_cam, centre):
    return r_world_to_cam, -r_world_to_cam @ np.asarray(centre, float)


def test_geo_model_gives_lat_lon_heading_and_pitch_in_gausser_axes(tmp_path):
    # gausser axes: x east, y down, z north. A camera's view direction is the third row of world→camera.
    down = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # optical axis +y: straight down
    north_45 = Rotation.from_euler("x", 45, degrees=True).as_matrix()  # optical axis: north, tipped 45° down (+y)
    cams = [("nadir.jpg", *camera(down, [10, -50, 0]), []), ("oblique.jpg", *camera(north_45, [0, -50, 0]), [])]
    write_model(tmp_path / "0", cams, {1: np.array([0.0, 0.0, 20.0])})

    geo = monitor.geo_model(tmp_path / "0", RECORD)

    nadir, oblique = geo["cameras"]
    assert nadir["pitch"] == pytest.approx(-90, abs=0.1)
    assert oblique["pitch"] == pytest.approx(-45, abs=0.1) and oblique["heading"] == pytest.approx(0, abs=0.1)
    lon, lat = Transformer.from_crs("EPSG:3067", "EPSG:4326", always_xy=True).transform(323967.0, 6825418.0)
    assert (nadir["lat"], nadir["lon"]) == pytest.approx((lat, lon), abs=1e-7)  # 10 m east of the origin
    _, north = Transformer.from_crs("EPSG:4326", "EPSG:3067", always_xy=True).transform(geo["points"]["latlon"][1], geo["points"]["latlon"][0])
    assert north == pytest.approx(6825438.0, abs=0.01)  # the point is 20 m north
