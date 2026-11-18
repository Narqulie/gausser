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


def write_glb(path, quads):
    """A .glb like TextureMesh's: per quad (corners xyz, RGB colour) one primitive with its own 4×4 PNG texture."""
    import json
    import struct

    from PIL import Image

    binary, views, accessors, meshes, materials, images = b"", [], [], [], [], []
    for i, (corners, colour) in enumerate(quads):
        Image.new("RGB", (4, 4), colour).save(path.parent / f"tex{i}.png")
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
        for data, kind, comp in ((np.asarray(corners, np.float32), "VEC3", 5126), (uv, "VEC2", 5126),
                                 (np.array([0, 1, 2, 0, 2, 3], np.uint32), "SCALAR", 5125)):
            views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": data.nbytes})
            accessors.append({"bufferView": len(views) - 1, "componentType": comp, "count": len(data), "type": kind})
            binary += data.tobytes()
        meshes.append({"attributes": {"POSITION": 3 * i, "TEXCOORD_0": 3 * i + 1}, "indices": 3 * i + 2, "material": i})
        materials.append({"pbrMetallicRoughness": {"baseColorTexture": {"index": i}}})
        images.append({"uri": f"tex{i}.png"})
    gltf = {"meshes": [{"primitives": meshes}], "materials": materials, "textures": [{"source": i} for i in range(len(quads))],
            "images": images, "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(binary)}]}
    text = json.dumps(gltf).encode().ljust((len(json.dumps(gltf)) + 3) // 4 * 4)
    chunks = struct.pack("<I4s", len(text), b"JSON") + text + struct.pack("<I4s", len(binary), b"BIN\0") + binary
    path.write_bytes(b"glTF" + struct.pack("<II", 2, 12 + len(chunks)) + chunks)


def test_mesh_map_shows_the_highest_surface_with_its_photo_texture(tmp_path):
    # Ground (y = 0, green) under a 2 m high roof (y = -2, red) over the middle; north up, east right.
    ground = ([[0, 0, 0], [2, 0, 0], [2, 0, 2], [0, 0, 2]], (0, 255, 0))
    roof = ([[0.5, -2, 0.5], [1.5, -2, 0.5], [1.5, -2, 1.5], [0.5, -2, 1.5]], (255, 0, 0))
    write_glb(tmp_path / "mesh.glb", [ground, roof])

    parts = mapping.read_mesh(tmp_path / "mesh.glb")
    assert [len(p.faces) for p in parts] == [2, 2]
    assert mapping.mesh_extent(parts) == pytest.approx((0, 2, 0, 2))
    rgb, coverage, depth = mapping.render_mesh(parts, west=0, north=2, width=20, height=20, resolution=0.1)
    assert rgb[10, 10] == pytest.approx([1, 0, 0], abs=0.01) and depth[10, 10] == pytest.approx(-2)  # the roof wins
    assert rgb[1, 1] == pytest.approx([0, 1, 0], abs=0.01) and depth[1, 1] == pytest.approx(0)
    assert coverage.min() == 1


def test_camera_projection_applies_radial_distortion_and_scales_to_full_resolution_copies():
    from gausser.colmap_model import Camera

    camera = Camera(2, 4032, 2268, (3000.0, 2016.0, 1134.0, 0.05))  # SIMPLE_RADIAL: f, cx, cy, k
    point = np.array([[0.2, -0.1, 1.0]])  # x/z = 0.2, y/z = -0.1, r² = 0.05
    u, v = camera.project(point)[0]
    assert (u, v) == pytest.approx((3000 * 0.2 * 1.0025 + 2016, 3000 * -0.1 * 1.0025 + 1134))
    assert camera.project(point, scale=2)[0] == pytest.approx((2 * u, 2 * v))  # 8064-pixel originals of a 4032 model


def test_photo_map_area_is_the_crop_box_in_model_coordinates(tmp_path):
    import json

    from gausser.scene import Scene

    scene = Scene(tmp_path)
    scene.sparse.mkdir()
    # A georeferenced model has y down, so the level frame is (x, -y, -z).
    scene.crop.write_text(json.dumps({"up": [0, -1, 0], "min": [-54, -53, -46], "max": [53, -25, 45]}))
    assert mapping._crop_area(scene) == pytest.approx((-54, 53, -45, 46))
