from __future__ import annotations

import os
import sqlite3
import struct
import sys
import types
from pathlib import Path

import pytest

from gausser import cli, msplat, steps
from gausser.scene import GausserError, Scene, registered_images


def make_scene(root: Path, images: int = 3) -> Scene:
    (root / "images").mkdir(parents=True)
    for index in range(images):
        (root / "images" / f"img{index}.jpg").write_bytes(b"jpg")
    return Scene(root)


def write_model(model: Path, registered: int) -> None:
    model.mkdir(parents=True)
    (model / "images.bin").write_bytes(struct.pack("<Q", registered))
    (model / "cameras.bin").write_bytes(b"")
    (model / "points3D.bin").write_bytes(b"")


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
    monkeypatch.setattr(steps, "require_tool", lambda name, hint: name)

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
    (scene.images / "._selected_metadata.json").write_bytes(b"junk")
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

    def fake_sharp_frames(command):
        output = Path(command[2])
        output.mkdir(parents=True)
        (output / "frame1.jpg").write_bytes(b"jpg")

    tools.on_run(fake_sharp_frames)
    steps.frames(scene, tmp_path / "images", [])

    assert [path.name for path in scene.image_files()] == ["frame1.jpg"]
    assert not scene.masks.exists() and not scene.colmap_masks.exists()
    assert not scene.database.exists() and not scene.sparse.exists()
    assert (scene.runs / "brush" / "old").is_dir()  # training history is kept


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


@pytest.fixture
def fake_steps(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(steps, "solve", lambda scene, matcher, no_masks: calls.append("solve"))
    monkeypatch.setattr(steps, "train_brush", lambda scene, **kwargs: calls.append("train"))
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
    assert sorted(path.name for path in run.iterdir()) == ["exports", "masked"]


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
    monkeypatch.setattr(steps, "train_brush", lambda scene, **kwargs: seen.append(("train", kwargs["no_masks"])))

    assert cli.main(["run", "-s", str(tmp_path), "--no-masks"]) == 0
    assert seen == [("solve", True), ("train", True)]


def test_status_runs_on_an_empty_folder(tmp_path, capsys):
    assert cli.main(["status", "-s", str(tmp_path)]) == 0
    assert "images  0" in capsys.readouterr().out


def test_missing_scene_is_a_clean_error(tmp_path, capsys):
    assert cli.main(["solve", "-s", str(tmp_path / "nope")]) == 1
    assert "does not exist" in capsys.readouterr().err


def test_a_second_command_on_the_same_scene_fails_fast(tmp_path):
    scene = Scene(tmp_path)
    with scene.lock():
        with pytest.raises(GausserError, match="already working"):
            with Scene(tmp_path).lock():
                pass
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
    assert [command for _label, command in guide.menu(empty, None)] == ["frames", "status"]
    assert guide.menu(empty, tmp_path / "clip.mp4")[0] == ("Run the pipeline (skips what is up to date)  ← next", "run")

    photos = make_scene(tmp_path / "photos")
    assert [command for _label, command in guide.menu(photos, None)] == ["run", "frames", "mask", "solve", "status"]

    done = solved_and_trained(tmp_path / "done")
    assert guide.menu(done, None)[0][1] == "view"
    assert "cameras 3/3 ✓" in guide.summary(done) and "trained ✓" in guide.summary(done)


def test_gausser_video_opens_the_guide_on_a_scene_next_to_the_video(tmp_path, monkeypatch):
    from gausser import guide

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    seen = []
    monkeypatch.setattr(cli, "interactive", lambda: True)
    monkeypatch.setattr(guide, "guided", lambda scene, video: seen.append((scene.root, video)) or ["status", "-s", str(scene.root)])

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


def make_subset_source(root: Path) -> Scene:
    """A solved scene with drone and iphone images, masks for both, and a COLMAP database."""
    scene = make_scene(root, images=0)
    names = ["frame_1.jpg", "frame_2.jpg", "iphone_1.jpg"]
    for name in names:
        (scene.images / name).write_bytes(b"jpg")
    (scene.images / "._frame_1.jpg").write_bytes(b"junk")
    scene.masks.mkdir()
    scene.colmap_masks.mkdir(parents=True)
    for name in names:
        (scene.masks / f"{Path(name).stem}.png").write_bytes(b"m")
        (scene.colmap_masks / f"{name}.png").write_bytes(b"c")
    write_model(scene.sparse / "0", 3)
    with sqlite3.connect(scene.database) as database:
        for statement in (
            "CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT)",
            "CREATE TABLE keypoints (image_id INTEGER)",
            "CREATE TABLE descriptors (image_id INTEGER)",
            "CREATE TABLE matches (pair_id INTEGER)",
            "CREATE TABLE two_view_geometries (pair_id INTEGER)",
            "CREATE TABLE pose_priors (corr_data_id INTEGER)",
            "CREATE TABLE frames (frame_id INTEGER)",
            "CREATE TABLE frame_data (frame_id INTEGER, data_id INTEGER)",
        ):
            database.execute(statement)
        for image_id, name in enumerate(names, start=1):
            database.execute("INSERT INTO images VALUES (?, ?)", (image_id, name))
            database.execute("INSERT INTO keypoints VALUES (?)", (image_id,))
            database.execute("INSERT INTO frames VALUES (?)", (image_id,))
            database.execute("INSERT INTO frame_data VALUES (?, ?)", (image_id, image_id))
        for pair in ((1, 2), (1, 3), (2, 3)):
            database.execute("INSERT INTO matches VALUES (?)", (pair[0] * 2147483647 + pair[1],))
    return scene


def test_subset_copies_matching_images_and_masks_and_filters_the_model(tmp_path, tools):
    source = make_subset_source(tmp_path / "source")

    def fake_image_deleter(command):
        write_model_files(Path(command[command.index("--output_path") + 1]), 2)

    tools.on_run(fake_image_deleter)
    result = steps.subset(source, ["frame_*"], tmp_path / "new")

    assert [path.name for path in result.image_files()] == ["frame_1.jpg", "frame_2.jpg"]
    assert not (result.images / "._frame_1.jpg").exists()
    assert sorted(path.name for path in result.masks.iterdir()) == ["frame_1.png", "frame_2.png"]
    assert sorted(path.name for path in result.colmap_masks.iterdir()) == ["frame_1.jpg.png", "frame_2.jpg.png"]
    (command,) = tools.commands
    assert command[1:5] == ["image_deleter", "--input_path", str(source.sparse / "0"), "--output_path"]
    assert command[command.index("--image_names_path") + 1].endswith("delete.txt")
    assert registered_images(result.sparse / "0") == 2
    assert result.solve_done(result.image_files())
    with sqlite3.connect(result.database) as database:
        assert database.execute("SELECT count(*) FROM matches").fetchone() == (1,)  # only frame_1 ↔ frame_2
        assert database.execute("SELECT count(*) FROM frame_data").fetchone() == (2,)
    assert len(source.image_files()) == 3  # source untouched


def test_subset_deletes_exactly_the_non_matching_images(tmp_path, tools):
    source = make_subset_source(tmp_path / "source")
    deleted: list[str] = []

    def fake_image_deleter(command):
        deleted.extend(Path(command[command.index("--image_names_path") + 1]).read_text().split())
        write_model_files(Path(command[command.index("--output_path") + 1]), 1)

    tools.on_run(fake_image_deleter)
    steps.subset(source, ["iphone_*"], tmp_path / "new")

    assert deleted == ["frame_1.jpg", "frame_2.jpg"]


def test_subset_errors_when_nothing_matches(tmp_path, tools):
    source = make_subset_source(tmp_path / "source")

    with pytest.raises(GausserError, match="matches nope"):
        steps.subset(source, ["nope"], tmp_path / "new")

    assert not (tmp_path / "new").exists()


def test_subset_refuses_a_scene_that_already_has_images(tmp_path, tools):
    source = make_subset_source(tmp_path / "source")
    existing = make_scene(tmp_path / "new")

    with pytest.raises(GausserError, match="not overwriting"):
        steps.subset(source, ["frame_*"], existing.root)

    assert len(existing.image_files()) == 3


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
