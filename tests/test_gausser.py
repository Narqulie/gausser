from __future__ import annotations

import json
import os
import sqlite3
import struct
import sys
import time
import types
from contextlib import closing
from pathlib import Path

import pytest
from PIL import Image

import gausser.scene

from gausser import cli, keyframes, monitor, msplat, pano, ply, qoocam, steps
from gausser.scene import GausserError, Scene, registered_images, run_tool


def record(calls: list, call: object, result: object) -> object:
    """For fakes: note the call and return what the real function would."""
    calls.append(call)
    return result


def make_scene(root: Path, images: int = 3) -> Scene:
    (root / "images").mkdir(parents=True)
    for index in range(images):
        (root / "images" / f"img{index}.jpg").write_bytes(b"jpg")
    return Scene(root)


def colmap_database(path, scene: Scene, size: int) -> None:
    """What `colmap feature_extractor --ImageReader.single_camera 1` writes (COLMAP 4): one camera, and
    each image in its own frame of the one rig whose sensor is that camera."""
    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE cameras (camera_id, model, width, height, params, prior_focal_length)")
        db.execute("INSERT INTO cameras VALUES (1, 0, ?, ?, x'00', 1)", (size, size))
        db.execute("CREATE TABLE images (image_id, name, camera_id)")
        db.execute("CREATE TABLE rigs (rig_id INTEGER PRIMARY KEY AUTOINCREMENT, ref_sensor_id, ref_sensor_type)")
        db.execute("CREATE TABLE frames (frame_id, rig_id)")
        db.execute("CREATE TABLE frame_data (frame_id, data_id, sensor_id, sensor_type)")
        db.execute("INSERT INTO rigs VALUES (1, 1, 0)")
        for image_id, name in enumerate(sorted(p.name for p in scene.image_files()), start=1):
            db.execute("INSERT INTO images VALUES (?, ?, 1)", (image_id, name))
            db.execute("INSERT INTO frames VALUES (?, 1)", (image_id,))
            db.execute("INSERT INTO frame_data VALUES (?, ?, 1, 0)", (image_id, image_id))
        db.commit()


def write_model(model: Path, registered: int) -> None:
    model.mkdir(parents=True)
    (model / "images.bin").write_bytes(struct.pack("<Q", registered))
    (model / "cameras.bin").write_bytes(b"")
    (model / "points3D.bin").write_bytes(b"")


def write_rig_model(model: Path, rig: dict | None, rig_from_world: list, centres: list) -> None:
    """A binary images.bin of 360° frames: every view of frame i, posed through the rig layout."""
    np = pytest.importorskip("numpy")
    rotation_type = pytest.importorskip("scipy.spatial.transform").Rotation
    model.mkdir(parents=True)
    records: list[bytes] = []
    for index, (rig_rotation, centre) in enumerate(zip(rig_from_world, centres, strict=True)):
        for view, (cam_from_rig, translation, _) in enumerate(pano.layout(rig)):
            r = cam_from_rig @ rig_rotation
            t = translation - r @ np.asarray(centre, dtype=float)
            x, y, z, w = rotation_type.from_matrix(r).as_quat()
            name = f"c{view:02d}_f{index}.jpg".encode()
            records.append(struct.pack("<idddddddi", len(records) + 1, w, x, y, z, *t, 1) + name + b"\0" + struct.pack("<Q", 0))
    (model / "images.bin").write_bytes(struct.pack("<Q", len(records)) + b"".join(records))
    (model / "cameras.bin").write_bytes(b"")
    (model / "points3D.bin").write_bytes(b"")


def copy_model(command: list[str]) -> None:
    """Fake `colmap model_transformer`: output = input."""
    import shutil

    source, target = (Path(command[command.index(flag) + 1]) for flag in ("--input_path", "--output_path"))
    for path in source.iterdir():
        shutil.copy(path, target / path.name)


def write_model_files(model: Path, registered: int) -> None:
    """Fill an existing folder, as colmap image_deleter does (it needs the output folder to exist)."""
    assert model.is_dir()
    write_model(model / "tmp", registered)
    for path in (model / "tmp").iterdir():
        path.rename(model / path.name)
    (model / "tmp").rmdir()


def write_database(scene: Scene) -> None:
    with sqlite3.connect(scene.database) as database:
        database.execute("CREATE TABLE images (name TEXT)")
        database.executemany("INSERT INTO images VALUES (?)", [(image.name,) for image in scene.image_files()])


def solved_and_trained(root: Path) -> Scene:
    """A scene whose model and export are up to date."""
    scene = make_scene(root)
    write_model(scene.sparse / "0", 3)
    write_database(scene)
    export = scene.runs / "brush" / "a" / "exports" / "x.ply"
    export.parent.mkdir(parents=True)
    export.write_bytes(b"ply")
    model_time = os.stat(scene.sparse / "0" / "images.bin").st_mtime
    os.utime(export, (model_time + 10, model_time + 10))
    return scene


@pytest.fixture
def tools(monkeypatch):
    """Pretend every external tool is installed and record the commands run."""
    commands: list[list[str]] = []
    monkeypatch.setattr(steps, "require_tool", lambda name: name)

    def record(handler):
        def run_tool(command, *, cwd, env=None):
            commands.append(command)
            handler(command)

        monkeypatch.setattr(steps, "run_tool", run_tool)

    record(lambda command: None)
    return types.SimpleNamespace(commands=commands, on_run=record)


def test_solve_publishes_the_largest_colmap_model_as_sparse_0(tmp_path, tools):
    scene = make_scene(tmp_path)
    (scene.colmap_masks).mkdir(parents=True)

    def fake_colmap(command):
        if command[1] == "feature_extractor":
            Path(command[command.index("--database_path") + 1]).write_bytes(b"db")
        if command[1] == "mapper":
            output = Path(command[command.index("--output_path") + 1])
            write_model(output / "0", 2)
            write_model(output / "1", 3)

    tools.on_run(fake_colmap)
    steps.solve(scene)

    assert sorted(path.name for path in scene.sparse.iterdir()) == ["0"]
    assert registered_images(scene.sparse / "0") == 3
    assert scene.database.read_bytes() == b"db"
    extract, match, _mapper = tools.commands
    assert extract[extract.index("--ImageReader.mask_path") + 1] == str(scene.colmap_masks)
    assert match[1] == "exhaustive_matcher"  # auto: few images


def test_solve_no_masks_ignores_existing_colmap_masks(tmp_path, tools):
    scene = make_scene(tmp_path)
    scene.colmap_masks.mkdir(parents=True)

    def fake_colmap(command):
        if command[1] == "feature_extractor":
            Path(command[command.index("--database_path") + 1]).write_bytes(b"db")
        if command[1] == "mapper":
            write_model(Path(command[command.index("--output_path") + 1]) / "0", 3)

    tools.on_run(fake_colmap)
    steps.solve(scene, no_masks=True)

    assert "--ImageReader.mask_path" not in tools.commands[0]
    assert (scene.sparse / steps.NO_MASKS).is_file()  # later masks must not throw this model away


def test_solve_gives_colmap_only_the_real_images_not_appledouble_sidecars(tmp_path, tools):
    scene = make_scene(tmp_path)
    (scene.images / "._img0.jpg").write_bytes(b"junk")
    (scene.images / "._frames.json").write_bytes(b"junk")
    listed: list[str] = []

    def fake_colmap(command):
        if command[1] == "feature_extractor":
            Path(command[command.index("--database_path") + 1]).write_bytes(b"db")
            listed.extend(Path(command[command.index("--image_list_path") + 1]).read_text().split())
        if command[1] == "mapper":
            write_model(Path(command[command.index("--output_path") + 1]) / "0", 3)

    tools.on_run(fake_colmap)
    steps.solve(scene)

    assert listed == ["img0.jpg", "img1.jpg", "img2.jpg"]


def test_solve_gives_each_source_video_its_own_camera(tmp_path, tools):
    """A crowd capture mixes phones; one shared camera model blurred every multi-contributor scene."""
    scene = make_scene(tmp_path, images=4)
    sources = {"img0.jpg": "anna.mp4", "img1.jpg": "anna.mp4", "img2.jpg": "bert.mp4"}  # img3 added by hand
    (scene.images / "frames.json").write_text(json.dumps({"items": [{"name": n, "source": s} for n, s in sources.items()]}))
    cameras: dict[str, int] = {}

    def fake_colmap(command):
        database = command[command.index("--database_path") + 1]
        if command[1] == "feature_extractor":
            colmap_database(database, scene, 16)
        if command[1] == "exhaustive_matcher":
            with sqlite3.connect(database) as db:
                cameras.update(db.execute("SELECT name, camera_id FROM images").fetchall())
                assert db.execute("SELECT count(*) FROM cameras").fetchone() == (2,)
                # COLMAP 4's mapper aborts unless each image's frame belongs to a rig of its own camera
                rigs = db.execute(
                    "SELECT DISTINCT i.camera_id, d.sensor_id, r.ref_sensor_id FROM images i JOIN frame_data d ON d.data_id = i.image_id "
                    "JOIN frames f ON f.frame_id = d.frame_id JOIN rigs r ON r.rig_id = f.rig_id"
                ).fetchall()
                assert sorted(rigs) == [(1, 1, 1), (2, 2, 2)]
        if command[1] == "mapper":
            write_model(Path(command[command.index("--output_path") + 1]) / "0", 4)

    tools.on_run(fake_colmap)
    steps.solve(scene)

    assert cameras == {"img0.jpg": 1, "img1.jpg": 1, "img2.jpg": 2, "img3.jpg": 1}


def test_solve_stops_the_pipeline_when_most_images_are_not_placed(tmp_path, tools):
    scene = make_scene(tmp_path, images=5)

    def fake_colmap(command):
        if command[1] == "feature_extractor":
            Path(command[command.index("--database_path") + 1]).write_bytes(b"db")
        if command[1] == "mapper":
            write_model(Path(command[command.index("--output_path") + 1]) / "0", 2)

    tools.on_run(fake_colmap)
    with pytest.raises(GausserError, match="only 2 of 5"):
        steps.solve(scene)
    assert registered_images(scene.sparse / "0") == 2  # kept, so `gausser train` can still use it


def test_failed_solve_keeps_the_previous_model(tmp_path, tools):
    scene = make_scene(tmp_path)
    write_model(scene.sparse / "0", 3)

    def failing(command):
        if command[1] == "mapper":
            raise GausserError("mapper failed")

    tools.on_run(failing)
    with pytest.raises(GausserError):
        steps.solve(scene)
    assert registered_images(scene.sparse / "0") == 3


def test_new_frames_clear_everything_built_from_the_old_images(tmp_path, tools):
    scene = make_scene(tmp_path)
    scene.masks.mkdir()
    scene.colmap_masks.mkdir(parents=True)
    write_model(scene.sparse / "0", 3)
    scene.database.write_bytes(b"db")
    (scene.runs / "brush" / "old").mkdir(parents=True)

    photos = tmp_path / "photos"
    photos.mkdir()
    Image.new("RGB", (40, 30)).save(photos / "frame1.jpg")
    steps.frames(scene, photos)

    assert [path.name for path in scene.image_files()] == ["frame1.jpg"]
    assert not scene.masks.exists() and not scene.colmap_masks.exists()
    assert not scene.database.exists() and not scene.sparse.exists()
    assert (scene.runs / "brush" / "old").is_dir()  # training history is kept


def test_360_frames_become_rig_views_that_colmap_solves_as_one_rig(tmp_path, tools):
    scene = Scene(tmp_path)
    (scene.masks_exclude).mkdir()  # an old operator mask is cleared; a new one masks features
    rig_names: list[set[str]] = []

    def fake_tools(command):
        if command[1] == "feature_extractor":
            colmap_database(command[command.index("--database_path") + 1], scene, 16)
        elif command[1] == "rig_configurator":
            with sqlite3.connect(command[command.index("--database_path") + 1]) as db:
                cameras = db.execute("SELECT camera_id, group_concat(substr(name, 5)) FROM images GROUP BY camera_id").fetchall()
            rig_names.extend(set(names.split(",")) for _, names in cameras)
        elif command[1] == "mapper":
            identity = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
            write_rig_model(Path(command[command.index("--output_path") + 1]) / "0", None, [identity] * 3, [[0, 0, 0], [1, 0, 0], [0, 0, 1]])
        elif command[1] == "model_transformer":
            copy_model(command)

    tools.on_run(fake_tools)
    pano_photos = tmp_path / "panos-in"
    pano_photos.mkdir()
    for name in ("a", "b"):
        Image.new("RGB", (64, 32)).save(pano_photos / f"{name}.jpg")
    steps.frames(scene, pano_photos)
    assert sorted(p.name for p in scene.panos.iterdir()) == ["a.jpg", "b.jpg"]
    assert len(scene.image_files()) == 2 * pano.VIEWS and not scene.masks_exclude.exists()
    scene.masks_exclude.mkdir()
    Image.new("L", (16, 16)).save(scene.masks_exclude / "c04_a.png")  # all black: no features
    steps.solve(scene)

    assert rig_names == [{"a.jpg", "b.jpg"}] * pano.VIEWS  # one camera per view, shared by both panoramas
    extract = next(c for c in tools.commands if c[1:2] == ["feature_extractor"])
    masks = Path(extract[extract.index("--ImageReader.mask_path") + 1])
    assert extract[extract.index("--ImageReader.camera_params") + 1] == "8,8,8"
    match = next(c for c in tools.commands if c[1:2] == ["exhaustive_matcher"])
    assert "--FeatureMatching.skip_image_pairs_in_same_frame" in match
    assert scene.model() is not None and not masks.exists()  # staging is cleaned up


def _qoocam_lens() -> bytes:
    """The QooCam 8K's LENS calibration text (values from a real camera)."""
    def line(prefix: str, f: float, cx: float, cy: float, rotation: list[float]) -> str:
        values = [825.52, 3.05, 0, 0, 0, 0, f, cx, cy, 0.022634, 0.013984, -0.01102, 0.001205, *rotation, 0, 0, -1.476]
        return prefix + struct.pack(f"<{len(values)}f", *values).hex()

    return "\r\n".join([
        line("=1", 1111.97, 1946.25, 1903.3, [0, 1, 0, -1, 0, 0, 0, 0, 1]),
        line("2", 1110.64, 5794.61, 1890.57, [0.0183, -0.9998, 0.0019, -0.9998, -0.0184, -0.0012, 0.0013, -0.0019, -1]),
    ]).encode() + b"\r\n"


def _qoocam_file(path: Path) -> None:
    """A .MOV with the QooCam 8K's kfix LENS calibration and no video."""
    lens = _qoocam_lens()
    entry = b"LENS".ljust(32, b"\0") + b"CHAR".ljust(8, b"\0") + struct.pack("<II", 512, 1) + lens.ljust(512, b"\0")
    kfix = b"PRODUCT".ljust(32, b"\0") + b"CHAR".ljust(8, b"\0") + struct.pack("<II", 16, 1) + b"QooCam 8K".ljust(16, b"\0") + entry
    path.write_bytes(struct.pack(">I4s", 16, b"ftyp") + b"qt  \0\0\0\0" + struct.pack(">I4s", 8 + len(kfix), b"kfix") + kfix)


def test_frames_takes_ranges_and_spacing_options():
    args = cli.build_parser().parse_args(["frames", "clip.MOV", "--range", "1.5-32.5", "--range", "0:57-1:20", "--spacing", "high", "--max-frames", "300"])
    assert args.ranges == [(1.5, 32.5), (57.0, 80.0)] and args.spacing == "high" and args.max_frames == 300


def test_qoocam_photos_are_used_whole_with_lens_0_above_lens_1(tmp_path, tools):
    (tmp_path / "scene").mkdir()
    scene = Scene(tmp_path / "scene")
    photos = tmp_path / "Kitchen"
    photos.mkdir()
    for name in ("Q360_1", "Q360_2"):
        image = Image.new("RGB", (64, 128), "red")
        image.paste((0, 0, 255), (0, 64, 64, 128))  # lens 1 below
        image.save(photos / f"{name}.JPG", exif=b"Exif\0\0" + b"PRODUCT=QOOCAM 8K LENS=" + _qoocam_lens())
    steps.frames(scene, photos)

    assert not tools.commands  # every photo is kept
    assert pano.lenses(scene.panos) == qoocam.calibration(photos / "Q360_1.JPG") is not None
    with Image.open(scene.panos / "Q360_1.jpg") as frame:
        left, right = frame.crop((0, 0, 64, 64)).resize((1, 1)), frame.crop((64, 0, 128, 64)).resize((1, 1))
    assert frame.size == (128, 64)
    assert left.tobytes()[0] > 200 and right.tobytes()[2] > 200  # lens 0 (red) left, lens 1 (blue) right
    assert len(scene.image_files()) == 2 * pano.VIEWS


def test_qoocam_video_becomes_per_lens_views_solved_as_a_calibrated_two_lens_rig(tmp_path, tools, monkeypatch):
    scene = Scene(tmp_path)
    rig_config: list = []
    ffmpeg_commands: list[list[str]] = []

    def fake_ffmpeg(command, *, cwd):
        ffmpeg_commands.append([*command, (cwd / "graph.txt").read_text()])
        for index in range(2):  # two 64-pixel lens images side by side
            Image.new("RGB", (128, 64), "white").save(cwd / f"{index:07d}.jpg")

    monkeypatch.setattr(keyframes, "run_tool", fake_ffmpeg)
    monkeypatch.setattr(keyframes, "probe", lambda ffprobe, video, span: keyframes.VideoInfo(3840, 3840, 30.0, 100, "bt709"))
    monkeypatch.setattr(keyframes, "analyse", lambda *args: [[keyframes.Pick(5, 0.1, 9.0), keyframes.Pick(40, 0.1, 9.0)]] * len(keyframes.LEVELS))

    def fake_tools(command):
        if command[1] == "feature_extractor":
            colmap_database(command[command.index("--database_path") + 1], scene, 32)
        elif command[1] == "rig_configurator":
            rig_config.extend(json.loads(Path(command[command.index("--rig_config_path") + 1]).read_text())[0]["cameras"])
        elif command[1] == "mapper":
            upright = [[0.0, 1, 0], [-1, 0, 0], [0, 0, 1]]  # grip (+x) down
            write_rig_model(Path(command[command.index("--output_path") + 1]) / "0", rig, [upright] * 3, [[0, 0, 0], [1, 0, 0], [0, 0, 1]])
        elif command[1] == "model_transformer":
            copy_model(command)

    tools.on_run(fake_tools)
    _qoocam_file(tmp_path / "Q360.MOV")
    rig = qoocam.calibration(tmp_path / "Q360.MOV")
    steps.frames(scene, tmp_path / "Q360.MOV", [(1.5, 32.5), (57.0, 80.0)])

    cuts = ffmpeg_commands
    assert [c[c.index("-ss") + 1 : c.index("-to") + 2] for c in cuts] == [["1.5", "-to", "32.5"], ["57", "-to", "80"]]
    # stored lens images are mirrored; both lenses select the same frame numbers
    assert all("[0:0]select='(eq(n,5)+eq(n,40))',vflip[a];[0:1]select='(eq(n,5)+eq(n,40))',vflip[b];[a][b]hstack[out]" in c[-1] for c in cuts)
    assert sorted(p.name for p in scene.panos.glob("*.jpg")) == ["p01_frame_000005.jpg", "p01_frame_000040.jpg", "p02_frame_000005.jpg", "p02_frame_000040.jpg"]
    rig = pano.lenses(scene.panos)
    assert rig is not None and rig["camera"] == "QooCam 8K"
    assert len(scene.image_files()) == 4 * pano.VIEWS  # 2 parts × 2 frames
    keep = sorted(scene.masks_exclude.glob("*.png"))
    assert len(keep) == 4 * pano.VIEWS  # lens circle and grip, so Brush and COLMAP skip them
    with Image.open(scene.masks_exclude / "c01_p01_frame_000005.png") as mask:  # tilted toward the grip
        black, _ = mask.convert("L").getextrema()
        histogram = mask.convert("L").histogram()
    assert black == 0 and 0 < histogram[0] < 32 * 32

    steps.solve(scene)
    lens1 = [camera for camera in rig_config if camera["image_prefix"] >= "c07_"]
    assert len(rig_config) == pano.VIEWS and len(lens1) == 7
    assert all(abs(sum(t * t for t in camera["cam_from_rig_translation"]) ** 0.5 - 0.02952) < 1e-4 for camera in lens1)


@pytest.mark.parametrize("fisheye", [False, True])
def test_level_transform_turns_a_tilted_360_model_so_gravity_is_plus_y(tmp_path, fisheye):
    np = pytest.importorskip("numpy")
    rotation_type = pytest.importorskip("scipy.spatial.transform").Rotation
    rig = None
    if fisheye:
        _qoocam_file(tmp_path / "Q360.MOV")
        rig = qoocam.calibration(tmp_path / "Q360.MOV")
    # Upright frames walking a loop at one height in a world where gravity is +y; the rig's own
    # down (pano +y, or the fisheye grip +x) points along gravity. Then the whole world is tilted.
    base = np.eye(3) if rig is None else np.array([[0.0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    angles = np.radians(np.arange(0, 360, 30))
    frames = [base @ rotation_type.from_euler("y", a).as_matrix() for a in angles]
    centres = [np.array([2 * np.cos(a), 0.05 * np.sin(3 * a), 3 * np.sin(a)]) for a in angles]
    tilt = rotation_type.from_euler("xyz", [50, -20, 110], degrees=True).as_matrix()
    write_rig_model(tmp_path / "m", rig, [r @ tilt.T for r in frames], [tilt @ c for c in centres])

    scale, w, x, y, z, *_ = map(float, pano.level_transform(tmp_path / "m" / "images.bin", rig).split())
    level = rotation_type.from_quat([x, y, z, w]).as_matrix()
    assert scale == 1 and np.allclose(level @ tilt @ [0, 1, 0], [0, 1, 0], atol=0.01)


def test_mask_keeps_whole_image_on_a_miss_but_fails_when_nothing_matches(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    image_module = pytest.importorskip("PIL.Image")
    scene = Scene(tmp_path)
    scene.images.mkdir()
    for name in ("hit.png", "miss.png"):
        image_module.new("RGB", (4, 2)).save(scene.images / name)
    hits = {"hit.png"}

    class Predictor:
        def __init__(self, **_kwargs):
            pass

        def predict(self, image, prompt, threshold, all_instances):
            name = Path(image.filename).name
            return (np.zeros((2, 4), dtype=bool), 0.9) if name in hits else None

    def save_masks(mask, image, brush_dir, colmap_dir, dilation, feather, colmap_mask=None):
        (brush_dir / f"{image.stem}.png").write_bytes(b"white" if mask.all() else b"subject")
        (colmap_dir / f"{image.name}.png").write_bytes(b"")

    monkeypatch.setitem(sys.modules, "gausser.sam", types.SimpleNamespace(Predictor=Predictor, save_masks=save_masks))
    write_model(scene.sparse / "0", 2)

    steps.mask(scene, "statue")

    assert (scene.masks / "hit.png").read_bytes() == b"subject"
    assert (scene.masks / "miss.png").read_bytes() == b"white"
    assert (scene.colmap_masks / "miss.png.png").exists()
    assert not scene.sparse.exists()  # COLMAP features depend on the masks

    write_model(scene.sparse / "0", 2)
    (scene.sparse / steps.NO_MASKS).touch()
    scene.database.write_bytes(b"db")
    steps.mask(scene, "statue")
    assert scene.sparse.exists() and scene.database.exists()  # a --no-masks model never read the masks

    hits.clear()
    with pytest.raises(GausserError, match="no 'statue'"):
        steps.mask(scene, "statue")
    assert (scene.masks / "hit.png").read_bytes() == b"subject"  # previous masks untouched


def test_mask_miss_drop_writes_black_brush_mask_and_white_colmap_mask(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    image_module = pytest.importorskip("PIL.Image")
    scene = Scene(tmp_path)
    scene.images.mkdir()
    for name in ("hit.png", "miss.png"):
        image_module.new("RGB", (4, 2)).save(scene.images / name)

    class Predictor:
        def __init__(self, **_kwargs):
            pass

        def predict(self, image, prompt, threshold, all_instances):
            return (np.zeros((2, 4), dtype=bool), 0.9) if Path(image.filename).name == "hit.png" else None

    saved = {}

    def save_masks(mask, image, brush_dir, colmap_dir, dilation, feather, colmap_mask=None):
        saved[image.name] = (mask.any(), mask.all() if colmap_mask is None else colmap_mask.all())
        (brush_dir / f"{image.stem}.png").write_bytes(b"")
        (colmap_dir / f"{image.name}.png").write_bytes(b"")

    monkeypatch.setitem(sys.modules, "gausser.sam", types.SimpleNamespace(Predictor=Predictor, save_masks=save_masks))

    steps.mask(scene, "statue", miss="drop")

    # (Brush mask has any subject pixel, COLMAP mask is all white)
    assert saved == {"hit.png": (False, False), "miss.png": (False, True)}


def test_mask_save_masks_colmap_override_is_independent_of_the_brush_mask(tmp_path):
    np = pytest.importorskip("numpy")
    image_module = pytest.importorskip("PIL.Image")
    pytest.importorskip("torch")
    from gausser import sam

    brush, colmap = tmp_path / "b", tmp_path / "c"
    brush.mkdir()
    colmap.mkdir()
    black = np.zeros((2, 4), dtype=bool)
    sam.save_masks(black, Path("a.jpg"), brush, colmap, 0, 0, ~black)

    with image_module.open(brush / "a.png") as b, image_module.open(colmap / "a.jpg.png") as c:
        assert b.getextrema() == (0, 0) and c.getextrema() == (255, 255)


def test_save_masks_negative_dilation_shrinks_the_brush_mask_only(tmp_path):
    np = pytest.importorskip("numpy")
    image_module = pytest.importorskip("PIL.Image")
    pytest.importorskip("torch")
    from gausser import sam

    brush, colmap = tmp_path / "b", tmp_path / "c"
    brush.mkdir()
    colmap.mkdir()
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    sam.save_masks(mask, Path("a.jpg"), brush, colmap, -2, 0)

    with image_module.open(brush / "a.png") as b, image_module.open(colmap / "a.jpg.png") as c:
        assert (np.asarray(b) > 127).sum() == 6 * 6 and (np.asarray(c) > 127).sum() == 100


def test_lod_uses_newest_export_and_current_splat_transform_flags(tmp_path, tools):
    scene = make_scene(tmp_path)
    export = scene.runs / "brush" / "a" / "exports" / "export_25000.ply"
    export.parent.mkdir(parents=True)
    export.write_bytes(b"ply")

    def fake_splat_transform(command):
        Path(command[-1]).write_text("{}")

    tools.on_run(fake_splat_transform)
    run = steps.lod(scene)

    assert (run / "lod-meta.json").is_file()
    assert run.parent == scene.runs / "lod"
    bundle = tools.commands[-1]
    assert bundle[1:5] == ["--lod-chunk-count", "512", "--lod-chunk-extent", "16"]
    assert bundle[5:7] == [str(export), "-l"]
    assert not list(scene.tmp.glob("lod-levels-*"))


@pytest.fixture(autouse=True)
def no_caffeinate(monkeypatch):
    monkeypatch.setattr(cli, "keep_awake", lambda: None)


@pytest.fixture(autouse=True)
def own_index(monkeypatch, tmp_path_factory):
    """Keep tests out of the real index of all scenes."""
    monkeypatch.setenv("GAUSSER_HOME", str(tmp_path_factory.mktemp("gausser-home")))


@pytest.fixture
def fake_steps(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(steps, "solve", lambda scene, matcher, no_masks: calls.append("solve"))
    monkeypatch.setattr(steps, "train_brush", lambda scene, **kwargs: record(calls, "train", scene.runs / "brush" / "x"))
    return calls


def test_run_skips_up_to_date_steps_and_from_redoes_the_rest(tmp_path, fake_steps):
    solved_and_trained(tmp_path)

    assert cli.main(["run", "-s", str(tmp_path)]) == 0
    assert fake_steps == []

    assert cli.main(["run", "-s", str(tmp_path), "--from", "solve"]) == 0
    assert fake_steps == ["solve", "train"]


def test_run_redoes_solve_when_images_changed(tmp_path, fake_steps):
    scene = solved_and_trained(tmp_path)
    (scene.images / "new.jpg").write_bytes(b"jpg")

    assert cli.main(["run", "-s", str(tmp_path)]) == 0
    assert fake_steps == ["solve", "train"]


def test_run_retrains_when_exports_are_older_than_the_model(tmp_path, fake_steps):
    scene = solved_and_trained(tmp_path)
    os.utime(scene.exports()[-1], (0, 0))  # e.g. a re-solve finished but training was interrupted

    assert cli.main(["run", "-s", str(tmp_path)]) == 0
    assert fake_steps == ["train"]


def test_brush_only_sees_current_inputs(tmp_path, tools):
    scene = make_scene(tmp_path)
    write_model(scene.sparse / "0", 3)
    scene.masks.mkdir()
    scene.colmap_masks.mkdir(parents=True)
    write_model(scene.runs / "msplat" / "old" / "dataset", 9)  # a stale model Brush must not find
    seen: list[set[str]] = []

    def fake_brush(command):
        dataset = Path(command[1])
        seen.append({path.name for path in dataset.iterdir()})
        export = Path(command[command.index("--export-path") + 1]) / "export_10.ply"
        export.parent.mkdir(parents=True)
        export.write_bytes(b"ply")

    tools.on_run(fake_brush)
    run = steps.train_brush(scene, iterations=10, export_every=5, extra=["--alpha-mode", "transparent"])

    assert seen == [{"images", "sparse", "masks"}]
    assert tools.commands[-1][-2:] == ["--alpha-mode", "transparent"]
    assert sorted(path.name for path in run.iterdir()) == ["exports", "init-sparse", "masked"]


def test_brush_leaves_out_images_with_an_empty_mask(tmp_path, tools):
    """With --alpha-mode transparent, an all-black mask would teach the subject to vanish."""
    scene = make_scene(tmp_path)
    write_model(scene.sparse / "0", 3)
    scene.masks.mkdir()
    for index, colour in enumerate((255, 0, 255)):
        Image.new("L", (4, 4), colour).save(scene.masks / f"img{index}.png")
    seen: list[list[str]] = []

    def fake_tools(command):
        if command[1] == "image_deleter":
            assert Path(command[command.index("--image_names_path") + 1]).read_text() == "img1.jpg\n"
            out = Path(command[command.index("--output_path") + 1])
            out.rmdir()  # COLMAP needs it to exist; write_model makes it
            write_model(out, 2)
            return
        seen.append(sorted(path.name for path in (Path(command[1]) / "images").iterdir()))
        export = Path(command[command.index("--export-path") + 1]) / "export_10.ply"
        export.parent.mkdir(parents=True)
        export.write_bytes(b"ply")

    tools.on_run(fake_tools)
    run = steps.train_brush(scene, iterations=10, export_every=5)

    assert seen == [["img0.jpg", "img2.jpg"]]
    assert sorted(path.name for path in run.iterdir()) == ["exports", "init-sparse", "masked"]


def test_brush_no_masks_leaves_masks_out_of_its_folder(tmp_path, tools):
    scene = make_scene(tmp_path)
    write_model(scene.sparse / "0", 3)
    scene.masks.mkdir()
    seen: list[set[str]] = []

    def fake_brush(command):
        seen.append({path.name for path in Path(command[1]).iterdir()})
        export = Path(command[command.index("--export-path") + 1]) / "export_10.ply"
        export.parent.mkdir(parents=True)
        export.write_bytes(b"ply")

    tools.on_run(fake_brush)
    run = steps.train_brush(scene, iterations=10, export_every=5, no_masks=True)

    assert seen == [{"images", "sparse"}]
    assert (run / "unmasked").is_file() and not (run / "masked").exists()


def test_run_passes_no_masks_to_solve_and_train(tmp_path, monkeypatch):
    make_scene(tmp_path)
    seen = []
    monkeypatch.setattr(steps, "solve", lambda scene, matcher, no_masks: seen.append(("solve", no_masks)))
    monkeypatch.setattr(
        steps, "train_brush", lambda scene, **kwargs: record(seen, ("train", kwargs["no_masks"]), scene.runs / "brush" / "x"),
    )

    assert cli.main(["run", "-s", str(tmp_path), "--no-masks"]) == 0
    assert seen == [("solve", True), ("train", True)]


def test_status_runs_on_an_empty_folder(tmp_path, capsys):
    assert cli.main(["status", "-s", str(tmp_path)]) == 0
    assert "images  0" in capsys.readouterr().out


def test_a_scene_is_named_after_its_folder_or_the_saved_name(tmp_path, capsys):
    root = tmp_path / "atlas" / "scene"
    root.mkdir(parents=True)
    summary = monitor.summary(str(root), 1.0)
    assert (summary["name"], summary["group"]) == ("atlas", str(tmp_path))  # `scene` folders take the parent's name
    assert cli.main(["name", "-s", str(root), "Atlas-tyttö"]) == 0
    assert monitor.summary(str(root), 1.0)["name"] == "Atlas-tyttö"
    assert cli.main(["name", "-s", str(root), "--clear"]) == 0
    assert capsys.readouterr().out.splitlines() == ["Atlas-tyttö", "atlas"]
    assert monitor.summary(str(tmp_path / "gone"), 1.0)["name"] == "gone"


def test_mask_counts_skip_exfat_sidecars(tmp_path):
    (tmp_path / "masks").mkdir()
    for name in ("a.png", "._a.png"):
        (tmp_path / "masks" / name).write_bytes(b"")
    assert monitor.state(Scene(tmp_path))["masks"]["count"] == 1


def test_missing_scene_is_a_clean_error(tmp_path, capsys):
    assert cli.main(["solve", "-s", str(tmp_path / "nope")]) == 1
    assert "does not exist" in capsys.readouterr().err


def test_a_second_command_on_the_same_scene_fails_fast(tmp_path):
    scene = Scene(tmp_path)
    with scene.lock(["train"]):
        with pytest.raises(GausserError, match="already working"):
            with Scene(tmp_path).lock():
                pass
        held = monitor.holder(scene)
        assert held and held["argv"] == ["train"]  # the refused command did not erase the holder
    with scene.lock():  # released again
        pass


def test_large_image_sets_use_sequential_matching_with_loop_detection(tmp_path, tools):
    scene = make_scene(tmp_path, images=201)

    def fake_colmap(command):
        if command[1] == "feature_extractor":
            Path(command[command.index("--database_path") + 1]).write_bytes(b"db")
        if command[1] == "mapper":
            write_model(Path(command[command.index("--output_path") + 1]) / "0", 201)

    tools.on_run(fake_colmap)
    steps.solve(scene)

    match = tools.commands[1]
    assert match[1] == "sequential_matcher"
    assert match[-2:] == ["--SequentialMatching.loop_detection", "1"]


def test_msplat_composites_masked_out_pixels_to_black(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    image, mask, output = tmp_path / "a.jpg", tmp_path / "a.png", tmp_path / "out.png"
    image_module.new("RGB", (2, 1), (200, 100, 50)).save(image, quality=100)
    mask_image = image_module.new("L", (2, 1))
    mask_image.putpixel((0, 0), 255)
    mask_image.save(mask)

    msplat._composite(image, mask, output)

    with image_module.open(output) as result:
        kept, removed = result.getpixel((0, 0)), result.getpixel((1, 0))
    assert removed == (0, 0, 0)
    assert all(abs(a - b) <= 3 for a, b in zip(kept, (200, 100, 50)))


def test_menu_recommends_the_next_step_and_hides_impossible_ones(tmp_path):
    from gausser import guide

    empty = Scene(tmp_path)
    assert [command for _label, command in guide.menu(empty, None)] == ["frames", "name", "status"]
    assert guide.menu(empty, tmp_path / "clip.mp4")[0] == ("Run the pipeline (skips what is up to date)  ← next", "run")

    photos = make_scene(tmp_path / "photos")
    assert [command for _label, command in guide.menu(photos, None)] == ["run", "frames", "mask", "solve", "name", "status"]

    done = solved_and_trained(tmp_path / "done")
    assert guide.menu(done, None)[0][1] == "view"
    assert "cameras 3/3 ✓" in guide.summary(done) and "trained ✓" in guide.summary(done)


def test_gausser_video_opens_the_guide_on_a_scene_next_to_the_video(tmp_path, monkeypatch):
    from gausser import guide

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    seen: list[tuple[Path, Path]] = []
    monkeypatch.setattr(cli, "interactive", lambda: True)
    monkeypatch.setattr(guide, "guided", lambda scene, video: record(seen, (scene.root, video), ["status", "-s", str(scene.root)]))

    assert cli.main([str(video)]) == 0
    assert seen == [((tmp_path / "clip").resolve(), video.resolve())]

    monkeypatch.setattr(cli, "interactive", lambda: False)
    assert cli.main([str(video)]) == 2  # no terminal, no menu


def test_cancelling_the_guide_removes_the_new_empty_scene_folder(tmp_path, monkeypatch):
    from gausser import guide

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    monkeypatch.setattr(cli, "interactive", lambda: True)

    def cancel(scene, video):
        raise KeyboardInterrupt

    monkeypatch.setattr(guide, "guided", cancel)
    assert cli.main([str(video)]) == 130
    assert not (tmp_path / "clip").exists()


def test_loose_photos_are_explained_and_imported_on_request(tmp_path, capsys, fake_steps):
    (tmp_path / "a.jpg").write_bytes(b"jpg")
    (tmp_path / "b.jpg").write_bytes(b"jpg")

    assert cli.main(["run", "-s", str(tmp_path)]) == 1
    assert "--import" in capsys.readouterr().err

    assert cli.main(["run", "-s", str(tmp_path), "--import"]) == 0
    assert [path.name for path in Scene(tmp_path).image_files()] == ["a.jpg", "b.jpg"]
    assert fake_steps == ["solve", "train"]

    (tmp_path / "a.jpg").write_bytes(b"again")
    with pytest.raises(GausserError, match="already has a.jpg"):
        Scene(tmp_path).import_loose_photos()


def test_unknown_words_still_get_the_command_list(capsys):
    with pytest.raises(SystemExit):
        cli.main(["stauts"])
    assert "invalid choice" in capsys.readouterr().err


def test_exclude_mask_is_the_inverse_of_the_grown_hit_with_the_subject_protected():
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    hit = np.zeros((20, 20), dtype=bool)
    hit[8:12, 8:12] = True
    protect = np.zeros_like(hit)
    protect[8:10, 8:10] = True

    white = steps.exclude_mask(hit, protect, 0)
    assert (~white).sum() == 12 and white[8:10, 8:10].all()  # the 4 px square minus the protected corner

    grown = steps.exclude_mask(hit, np.zeros_like(hit), 2)
    assert (~grown).sum() == 8 * 8  # 4 px square grown by 2 px on every side


def test_mask_exclude_writes_only_masks_exclude_with_the_subject_protected(tmp_path, monkeypatch, capsys):
    np = pytest.importorskip("numpy")
    image_module = pytest.importorskip("PIL.Image")
    scene = Scene(tmp_path)
    scene.images.mkdir()
    for name in ("people.png", "none.png"):
        image_module.new("RGB", (4, 2)).save(scene.images / name)
    scene.masks.mkdir()
    subject = np.zeros((2, 4), dtype=np.uint8)
    subject[:, 3] = 255  # the statue is the last column
    image_module.fromarray(subject, mode="L").save(scene.masks / "people.png")
    write_model(scene.sparse / "0", 2)
    calls = []

    class Predictor:
        def __init__(self, **_kwargs):
            pass

        def predict(self, image, prompt, threshold, all_instances):
            calls.append(all_instances)
            return (np.ones((2, 4), dtype=bool), 0.9) if Path(image.filename).name == "people.png" else None

    fake_sam = types.SimpleNamespace(Predictor=Predictor)
    monkeypatch.setitem(sys.modules, "gausser.sam", fake_sam)
    monkeypatch.setattr("gausser.sam", fake_sam, raising=False)  # `from . import sam` prefers the package attribute

    steps.mask(scene, "person", exclude=True)  # a scene where no image matches is fine too

    with image_module.open(scene.masks_exclude / "people.png") as m:
        assert (np.asarray(m) > 127).tolist() == [[False, False, False, True]] * 2
    with image_module.open(scene.masks_exclude / "none.png") as m:
        assert m.getextrema() == (255, 255)
    assert calls == [True, True]  # always every instance
    assert (scene.masks / "people.png").is_file() and not scene.colmap_masks.exists()
    assert (scene.sparse / "0").is_dir()  # COLMAP inputs untouched
    assert "Excluded 'person' in 1/2 images" in capsys.readouterr().out


def test_brush_exclude_masks_stages_masks_exclude_and_marks_the_run(tmp_path, tools):
    scene = make_scene(tmp_path)
    write_model(scene.sparse / "0", 3)
    scene.masks.mkdir()
    with pytest.raises(GausserError, match="No masks-exclude/"):
        steps.train_brush(scene, exclude_masks=True)
    scene.masks_exclude.mkdir()
    seen = []

    def fake_brush(command):
        masks = Path(command[1]) / "masks"
        seen.append(masks.resolve())
        export = Path(command[command.index("--export-path") + 1]) / "export_10.ply"
        export.parent.mkdir(parents=True)
        export.write_bytes(b"ply")

    tools.on_run(fake_brush)
    run = steps.train_brush(scene, iterations=10, export_every=5, exclude_masks=True)

    assert seen == [scene.masks_exclude.resolve()]
    assert (run / "excluded").is_file() and not (run / "masked").exists()


def test_train_exclude_masks_conflicts_are_clean_errors(tmp_path, capsys):
    make_scene(tmp_path)
    assert cli.main(["train", "-s", str(tmp_path), "--exclude-masks", "--no-masks"]) == 1
    assert cli.main(["train", "-s", str(tmp_path), "--exclude-masks", "--trainer", "msplat"]) == 1
    assert capsys.readouterr().err.count("--exclude-masks is for Brush") == 2


def test_mask_command_passes_exclude(tmp_path, monkeypatch):
    make_scene(tmp_path)
    seen = []
    monkeypatch.setattr(steps, "mask", lambda scene, prompt, **kw: seen.append((prompt, kw["exclude"], kw["dilation"])))
    assert cli.main(["mask", "person", "-s", str(tmp_path), "--exclude", "--dilation", "8"]) == 0
    assert cli.main(["mask", "statue", "-s", str(tmp_path)]) == 0
    assert seen == [("person", True, 8), ("statue", False, 0)]


def test_a_step_logs_its_output_and_tool_output_and_records_its_outcome(tmp_path):
    scene = Scene(tmp_path)
    with scene.step("solve", matcher="auto"):
        print("python says hi")
        run_tool(["sh", "-c", "printf 'tool line\\n'"], cwd=tmp_path)
    with pytest.raises(GausserError):
        with scene.step("train"):
            run_tool(["sh", "-c", "exit 3"], cwd=tmp_path)

    solve, train = monitor.history(scene)
    assert (solve["step"], solve["ok"], solve["matcher"]) == ("solve", True, "auto")
    assert (train["ok"], train["error"]) == (False, "`sh` failed with exit code 3")
    log = (scene.logs / solve["log"]).read_text()
    assert "python says hi" in log and "tool line" in log
    assert sys.stdout.__class__.__name__ != "_Tee"  # restored


def write_colmap_model(model: Path) -> None:
    """One PINHOLE camera, two images 2 units apart looking along +z, three points."""
    model.mkdir(parents=True)
    (model / "cameras.bin").write_bytes(struct.pack("<QiiQQ4d", 1, 1, 1, 640, 480, 500.0, 500.0, 320.0, 240.0))
    images = struct.pack("<Q", 2)
    for image_id, tx, name, observed in ((1, 0.0, b"a.jpg", (0, -1)), (2, -2.0, b"b.jpg", (1, 2))):
        images += struct.pack("<idddddddi", image_id, 1, 0, 0, 0, tx, 0, 0, 1) + name + b"\0"
        images += struct.pack("<Q", len(observed)) + b"".join(struct.pack("<ddq", 0, 0, p) for p in observed)
    (model / "images.bin").write_bytes(images)
    points = struct.pack("<Q", 3)
    for point_id in range(3):
        points += struct.pack("<QdddBBBdQ", point_id, point_id, 0, 5, 255, 0, 0, 0.5, 2) + struct.pack("<iiii", 1, 0, 2, 0)
    (model / "points3D.bin").write_bytes(points)


def test_monitor_reads_a_binary_colmap_model(tmp_path):
    write_colmap_model(tmp_path / "0")
    model = monitor.read_model(tmp_path / "0")
    a, b = model["cameras"]
    assert (a["name"], a["center"], a["points"], a["fx"]) == ("a.jpg", [0, 0, 0], 1, 500.0)
    assert (b["name"], b["center"], b["points"]) == ("b.jpg", [2, 0, 0], 2)
    assert model["up"] == pytest.approx([0, -1, 0])  # image-down is +y in COLMAP's camera axes
    assert model["points"]["xyz"] == [0, 0, 5, 1, 0, 5, 2, 0, 5]
    assert model["stats"]["points"] == 3 and model["stats"]["track_lengths"] == [[2, 3]]


def test_monitor_shows_the_running_step_and_its_live_snapshot(tmp_path):
    scene = make_scene(tmp_path)
    with scene.lock(["solve"]), scene.step("solve"):
        write_colmap_model(scene.tmp / "solve-20990101-000000" / "snapshots" / "1")
        state = monitor.state(scene)
    assert state["holder"]["argv"] == ["solve"]
    assert state["current"]["step"] == "solve"
    assert state["current"]["live"]["model"]["registered"] == 2
    assert monitor.state(scene)["current"] is None  # ended


def test_monitor_cleans_terminal_output_and_serves_only_the_scene(tmp_path):
    assert monitor.clean(b"\x1b[32mok\x1b[0m\n10%\r50%\r100%\r\n") == "ok\n100%\n"
    scene = Scene(tmp_path)
    with pytest.raises(PermissionError):
        monitor.Monitor().path(scene, "../../etc/passwd")
    with pytest.raises(FileNotFoundError):  # only scenes in the index are served
        monitor.Monitor().scene("/etc")


def test_monitor_api_answers_bad_requests_with_status_codes(tmp_path):
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.error import HTTPError
    from urllib.parse import quote
    from urllib.request import urlopen

    scene = make_scene(tmp_path)
    monitor.register(scene)
    server = ThreadingHTTPServer(("127.0.0.1", 0), monitor.make_handler(monitor.Monitor()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/api/"

    def status(route: str) -> int:
        try:
            with urlopen(base + route) as response:
                return int(response.status)
        except HTTPError as error:
            return error.code

    try:
        s = quote(str(scene.root))
        assert status(f"state?scene={s}") == 200
        assert status("state?scene=/nope") == 404
        assert status(f"log?scene={s}&name=none.log") == 404
        assert status(f"log?scene={s}&name=none.log&offset=abc") == 400  # malformed before missing
        assert status(f"evals?scene={s}") == 400  # run= missing
        assert status(f"thumb?scene={s}&path=../outside.jpg") == 403
    finally:
        server.shutdown()


def test_monitor_ignores_a_lock_whose_pid_was_reused_after_a_reboot(tmp_path):
    scene = Scene(tmp_path)
    (scene.root / ".gausser").mkdir()
    lock = scene.root / ".gausser" / "lock"
    lock.write_text(json.dumps({"pid": os.getpid(), "argv": ["train"], "start": 1_000_000.0}))  # 1970: before this process
    assert monitor.holder(scene) is None
    lock.write_text(json.dumps({"pid": os.getpid(), "argv": ["train"], "start": time.time()}))
    assert monitor.holder(scene)


def test_monitor_does_not_index_a_folder_that_only_holds_photos(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "photo.jpg")
    assert monitor.register(Scene(tmp_path)) is None and not monitor.known_scenes()
    gausser.scene.index(Scene(tmp_path), {"opened": 1.0})  # what `--monitor` does before the command imports the photos
    assert monitor.register(Scene(tmp_path))


def test_monitor_cache_drops_the_oldest_entries_past_its_byte_budget(monkeypatch):
    monkeypatch.setattr(monitor, "CACHE_BYTES", 10)
    app = monitor.Monitor()
    assert app.cached("a", lambda: b"12345678") == b"12345678"
    app.cached("b", lambda: b"12345678")
    assert list(app.cache) == ["b"] and app.cache_bytes == 8
    assert app.cached("b", lambda: b"x") == b"12345678"  # a hit never calls make()


def test_the_index_lists_every_scene_with_its_runs_and_eval_renders(tmp_path):
    scene = Scene(tmp_path)
    scene.images.mkdir()
    Image.new("RGB", (8, 8), (100, 100, 100)).save(scene.images / "a.jpg")
    with scene.step("train") as made:
        made["run"] = "runs/brush/1"
    renders = scene.runs / "brush" / "1" / "exports" / "eval_1000" / "images"
    renders.mkdir(parents=True)
    Image.new("RGB", (4, 4), (110, 100, 100)).save(renders / "a.png")

    assert list(monitor.known_scenes()) == [str(scene.root)]
    (run,) = monitor.recent_runs()
    assert (run["step"], run["run"], run["ok"]) == ("train", "runs/brush/1", True)
    app = monitor.Monitor()
    (evaluation,) = app.evals(app.scene(str(scene.root)), "runs/brush/1")
    (image,) = evaluation["images"]
    assert evaluation["step"] == 1000 and image["photo"] == "images/a.jpg"
    assert 25 < image["psnr"] < 35  # 10/255 off in one channel


# `gausser dense` and `gausser mesh` (OpenMVS), and Brush starting from the dense cloud.


def write_dense_ply(path: Path, xyz: list[tuple[float, float, float]], views: list[int]) -> None:
    """An OpenMVS dense cloud: xyz, rgb, normals, then the variable-length view and weight lists."""
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float32 x\nproperty float32 y\nproperty float32 z\n"
        "property uint8 red\nproperty uint8 green\nproperty uint8 blue\n"
        "property float32 nx\nproperty float32 ny\nproperty float32 nz\n"
        "property list uint8 uint32 view_indices\nproperty list uint8 float32 view_weights\n"
        "end_header\n"
    )
    body = b"".join(
        struct.pack("<3f3B3f", *point, index, 2 * index, 3 * index, 0, 0, 1)
        + struct.pack(f"<B{n}I", n, *range(n)) + struct.pack(f"<B{n}f", n, *[1.0] * n)
        for index, (point, n) in enumerate(zip(xyz, views, strict=True))
    )
    path.write_bytes(header.encode() + body)


def dense_scene(root: Path, points: int = 4):
    scene = solved_and_trained(root)
    scene.dense.mkdir()
    (scene.dense / "scene_dense.mvs").write_bytes(b"mvs")
    write_dense_ply(scene.dense / "scene_dense.ply", [(i, i + 0.5, -i) for i in range(points)], [2, 0, 7, 3][:points])
    return scene


def test_dense_undistorts_imports_and_densifies_into_dense(tmp_path, tools):
    scene = make_scene(tmp_path)
    write_model(scene.sparse / "0", 3)

    def fake(command):
        if command[0] == "colmap":
            cwd = Path(command[command.index("--output_path") + 1])
            (cwd / "images").mkdir()
            (cwd / "stereo").mkdir()
            (cwd / "run-colmap-geometric.sh").write_text("")
        elif command[0] == "DensifyPointCloud":
            staged = next(scene.tmp.glob("dense-*"))
            (staged / "scene_dense.mvs").write_bytes(b"mvs")
            (staged / "DensifyPointCloud-1.log").write_text("")
            write_dense_ply(staged / "scene_dense.ply", [(0, 0, 0)], [2])

    tools.on_run(fake)
    steps.dense(scene)

    assert [command[0] for command in tools.commands] == ["colmap", "InterfaceCOLMAP", "DensifyPointCloud"]
    assert tools.commands[0][1] == "image_undistorter"
    assert tools.commands[2][1:] == ["scene.mvs", "--resolution-level", "1", "--remove-dmaps", "1"]
    assert sorted(path.name for path in scene.dense.iterdir()) == ["images", "scene_dense.mvs", "scene_dense.ply"]
    assert scene.dense_done()


def test_a_new_model_removes_the_dense_cloud_built_from_the_old_one(tmp_path, tools):
    scene = dense_scene(tmp_path)

    def fake(command):
        if command[1] == "feature_extractor":
            Path(command[command.index("--database_path") + 1]).write_bytes(b"db")
        if command[1] == "mapper":
            write_model(Path(command[command.index("--output_path") + 1]) / "0", 3)

    tools.on_run(fake)
    steps.solve(scene)

    assert scene.dense_cloud() is None


def test_read_dense_cloud_walks_the_variable_length_view_lists(tmp_path):
    path = tmp_path / "scene_dense.ply"
    write_dense_ply(path, [(1, 2, 3), (4, 5, 6), (7, 8, 9)], [2, 0, 5])

    xyz, rgb = ply.read_dense_cloud(path, limit=10)

    assert xyz.tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert rgb.tolist() == [[0, 0, 0], [1, 2, 3], [2, 4, 6]]
    assert len(ply.read_dense_cloud(path, limit=2)[0]) == 2


def test_brush_starts_from_the_dense_cloud_when_there_is_one(tmp_path, tools):
    scene = dense_scene(tmp_path)
    (scene.sparse / "0" / "._cameras.bin").write_bytes(b"exFAT sidecar")  # must not be linked
    seen = {}

    def fake_brush(command):
        model = Path(command[1]) / "sparse" / "0"
        data = (model / "points3D.bin").read_bytes()
        seen["count"] = struct.unpack_from("<Q", data)[0]
        seen["first"] = struct.unpack_from("<Q3d3B", data, 8)
        seen["cameras"] = (model / "cameras.bin").resolve() == (scene.sparse / "0" / "cameras.bin").resolve()
        seen["start"] = seen.get("start", 0) + 1
        seen["sidecar"] = (model / "._cameras.bin").exists()
        export = Path(command[command.index("--export-path") + 1]) / "export_10.ply"
        export.parent.mkdir(parents=True)
        export.write_bytes(b"ply")

    tools.on_run(fake_brush)
    run = steps.train_brush(scene, iterations=10)

    assert seen == {"count": 4, "first": (1, 0.0, 0.5, 0.0, 0, 0, 0), "cameras": True, "start": 1, "sidecar": False}
    assert (run / "init-dense").is_file()
    assert (scene.sparse / "0" / "points3D.bin").read_bytes() == b""  # the scene's own model is untouched

    def plain_brush(command):
        seen["sparse"] = (Path(command[1]) / "sparse").resolve() == scene.sparse.resolve()
        export = Path(command[command.index("--export-path") + 1]) / "export_10.ply"
        export.parent.mkdir(parents=True)
        export.write_bytes(b"ply")

    tools.on_run(plain_brush)
    run = steps.train_brush(scene, iterations=10, init="sparse")
    assert (run / "init-sparse").is_file() and seen["sparse"]

    scene.masks.mkdir()  # the dense cloud ignores masks, so a masked run starts from COLMAP's points
    seen.clear()
    run = steps.train_brush(scene, iterations=10)
    assert (run / "init-sparse").is_file() and seen["sparse"]


@pytest.mark.parametrize("masked", [False, True])
def test_brush_starts_from_the_lidar_points_unless_it_gets_masks(tmp_path, tools, masked):
    np = pytest.importorskip("numpy")
    scene = Scene(tmp_path)
    scene.images.mkdir()
    model = scene.sparse / "0"
    model.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (model / name).write_bytes(b"")
    scene.arkit.mkdir()
    ply.write_points(scene.arkit / "points.ply", np.array([[0.0, 1, 2], [3, 4, 5]]), np.array([[9, 8, 7], [6, 5, 4]]))
    if masked:
        scene.masks.mkdir()
    seen = {}

    def brush(command):
        data = (Path(command[1]) / "sparse" / "0" / "points3D.bin").read_bytes()
        seen["first"] = struct.unpack_from("<Q3d3B", data, 8) if data else 0  # sparse: the empty test model
        export = Path(command[command.index("--export-path") + 1]) / "export_10.ply"
        export.parent.mkdir(parents=True)
        export.write_bytes(b"ply")

    tools.on_run(brush)
    run = steps.train_brush(scene, iterations=10)

    assert (run / ("init-sparse" if masked else "init-lidar")).is_file()
    assert seen["first"] == (0 if masked else (1, 0.0, 1.0, 2.0, 9, 8, 7))


def test_mesh_reconstructs_refines_decimates_and_textures_into_a_run(tmp_path, tools):
    scene = dense_scene(tmp_path)

    def fake(command):
        out = Path(command[command.index("-o") + 1])
        if command[0] in ("ReconstructMesh", "RefineMesh"):
            out.with_suffix(".ply").write_text("ply\nformat binary_little_endian 1.0\nelement face 1000000\nend_header\n")
        if command[0] == "TextureMesh":
            out.with_suffix(".glb").write_bytes(b"glb")
            (out.parent / "mesh-textured_diffuse00.png").write_bytes(b"png")
        out.write_bytes(b"mvs")

    tools.on_run(fake)
    run = steps.mesh(scene, faces=30000)

    assert [command[0] for command in tools.commands] == ["ReconstructMesh", "RefineMesh", "TextureMesh"]
    assert tools.commands[2][tools.commands[2].index("--decimate") + 1] == "0.030000"
    assert sorted(path.name for path in run.iterdir()) == ["mesh-textured_diffuse00.png", "mesh.glb", "mesh.ply", "refined.ply"]
    assert scene.mesh_done()


def test_run_mesh_builds_the_dense_cloud_before_training_and_meshes_last(tmp_path, monkeypatch):
    scene = solved_and_trained(tmp_path)
    order = []
    monkeypatch.setattr(steps, "dense", lambda scene, level: order.append("dense"))
    monkeypatch.setattr(steps, "train_brush", lambda scene, **kw: record(order, f"train-{kw['init']}", scene.runs / "brush" / "y"))
    monkeypatch.setattr(steps, "mesh", lambda scene, **kw: record(order, "mesh", scene.runs / "mesh" / "z"))

    assert cli.main(["run", "-s", str(scene.root), "--mesh"]) == 0
    assert order == ["dense", "train-auto", "mesh"]


def test_solve_with_gps_matches_spatially_maps_globally_and_georeferences(tmp_path, tools, monkeypatch):
    scene = make_scene(tmp_path)
    positions = {f"img{i}.jpg": (61.52 + i * 1e-4, 23.69 + (i % 2) * 2e-4, 350.0) for i in range(3)}
    monkeypatch.setattr(steps.geo, "image_positions", lambda images: positions)
    monkeypatch.setattr(steps.geo, "georeference", lambda images_bin, found: ("1 1 0 0 0 0 0 0\n", {"crs": "EPSG:3067"}, "ok"))

    def fake_colmap(command):
        if command[1] == "feature_extractor":
            Path(command[command.index("--database_path") + 1]).write_bytes(b"db")
        if command[1] == "global_mapper":
            write_model(Path(command[command.index("--output_path") + 1]) / "0", 3)
        if command[1] == "model_transformer":
            copy_model(command)

    tools.on_run(fake_colmap)
    steps.solve(scene)

    assert [command[1] for command in tools.commands] == ["feature_extractor", "spatial_matcher", "global_mapper", "model_transformer"]
    assert scene.geo() == {"crs": "EPSG:3067"}
