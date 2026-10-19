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

    def save_masks(mask, image, brush_dir, colmap_dir, dilation, feather):
        (brush_dir / f"{image.stem}.png").write_bytes(b"white" if mask.all() else b"subject")
        (colmap_dir / f"{image.name}.png").write_bytes(b"")

    monkeypatch.setitem(sys.modules, "gausser.sam", types.SimpleNamespace(Predictor=Predictor, save_masks=save_masks))
    write_model(scene.sparse / "0", 2)

    steps.mask(scene, "statue")

    assert (scene.masks / "hit.png").read_bytes() == b"subject"
    assert (scene.masks / "miss.png").read_bytes() == b"white"
    assert (scene.colmap_masks / "miss.png.png").exists()
    assert not scene.sparse.exists()  # COLMAP features depend on the masks

    hits.clear()
    with pytest.raises(GausserError, match="no 'statue'"):
        steps.mask(scene, "statue")
    assert (scene.masks / "hit.png").read_bytes() == b"subject"  # previous masks untouched


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
    monkeypatch.setattr(steps, "solve", lambda scene, matcher: calls.append("solve"))
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
    run = steps.train_brush(scene, iterations=10, export_every=5)

    assert seen == [{"images", "sparse", "masks"}]
    assert sorted(path.name for path in run.iterdir()) == ["exports"]


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
