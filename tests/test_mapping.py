import numpy as np
import pytest

from gausser import mapping


def splats(xyz, colour, size=0.1, opacity=0.999):
    n = len(xyz)
    out = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    out.update({f"scale_{i}": np.full(n, np.log(size)) for i in range(3)})
    out.update({"rot_0": np.ones(n), "rot_1": np.zeros(n), "rot_2": np.zeros(n), "rot_3": np.zeros(n)})
    out["opacity"] = np.full(n, np.log(opacity / (1 - opacity)))
    for c in range(3):
        out[f"f_dc_{c}"] = (np.asarray(colour)[:, c] - 0.5) / mapping.SH_C0
    return out


def test_the_top_splat_hides_the_one_below_and_the_dsm_is_its_height():
    # A red splat 2 m above (y = -2, y points down) a blue one, both at x = 1, z = 1.
    red_over_blue = splats(np.array([[1.0, -2.0, 1.0], [1.0, 0.0, 1.0]]), [[1, 0, 0], [0, 0, 1]], size=0.3)
    rgb, coverage, depth = mapping.render(red_over_blue, west=0, north=2, width=20, height=20, resolution=0.1)
    centre = (10, 10)  # row 10 = z 1 (north 2 minus 1 m), column 10 = x 1
    assert coverage[centre] > 0.99
    assert rgb[centre] == pytest.approx([1, 0, 0], abs=0.05)  # alpha is capped at 0.99, as in 3DGS
    assert depth[centre] == pytest.approx(-2, abs=0.05)
    assert coverage[0, 0] < 0.05  # 1.4 m from the splats: nearly empty


def test_north_is_up_and_east_is_right():
    pair = splats(np.array([[1.5, 0.0, 1.5], [0.5, 0.0, 0.5]]), [[0, 1, 0], [1, 1, 0]], size=0.05)
    rgb, coverage, _ = mapping.render(pair, west=0, north=2, width=20, height=20, resolution=0.1)
    assert rgb[5, 15] == pytest.approx([0, 1, 0], abs=0.02)  # north-east splat: top right
    assert rgb[15, 5] == pytest.approx([1, 1, 0], abs=0.02)


def test_chunked_blending_matches_one_pass(monkeypatch):
    rng = np.random.default_rng(0)
    many = splats(rng.uniform([0, -3, 0], [2, 0, 2], (300, 3)), rng.uniform(0, 1, (300, 3)), size=0.08, opacity=0.6)
    whole = mapping.render(many, 0, 2, 40, 40, 0.05)
    monkeypatch.setattr(mapping, "MAX_PAIRS", 500)
    chunked = mapping.render(many, 0, 2, 40, 40, 0.05)
    for a, b in zip(whole, chunked, strict=True):
        assert np.allclose(a, b, atol=1e-9, equal_nan=True)


def test_a_splat_wider_than_max_radius_is_left_out_rather_than_cut_to_a_box(monkeypatch):
    # Cut to its bounding box, a wide haze splat drew straight seams across the orthophoto.
    monkeypatch.setattr(mapping, "MAX_RADIUS", 8)
    haze = splats(np.array([[1.0, 0.0, 1.0]]), [[1, 1, 1]], size=1.0, opacity=0.5)
    _, coverage, _ = mapping.render(haze, west=0, north=2, width=20, height=20, resolution=0.1)
    assert coverage.max() == 0
