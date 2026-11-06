# gausser: agent instructions

## What this is

A single `gausser` CLI (`gausser.cli:main`) that turns photos or video into
Gaussian splats: `frames → mask → solve → [dense] → train → lod`, plus `mesh`. It is deliberately
small. Do not reintroduce manifests, scene locks, invalidation archives, a JSON
engine, or a GUI without an explicit request from the Captain. The one GUI is
`gausser monitor` (2026-11-05, requested): a read-only web page that never drives
the pipeline. The native
Studio app, `gausser-engine`, and the separate `sammer`/`colmapper`/`brusher`/…
commands were removed on 2026-10-19 (branch `simplify`); git history has them.

## Layout

```
src/gausser/
├── cli.py     argparse subcommands, `run` pipeline, `status`, `gausser [PATH]` routing
├── guide.py   scene-aware guided menu: summary, recommended next step, minimal questions
├── keyframes.py frames: one ffmpeg analysis pass (focus + DIS optical flow, tracked grid, ladder of thresholds) picks frame numbers, one pass extracts them
├── steps.py   frames (calls keyframes), mask, solve (COLMAP), dense + mesh (OpenMVS), subset (colmap image_deleter + filtered database), train_brush, lod (splat-transform)
├── sam.py     SAM3 predictor + mask writers; imported only by `mask` (needs torch)
├── pano.py    360° frames: 14 pinhole views per equirect (pycolmap.panorama's 12 + zenith/nadir) or per dual fisheye (7 per lens), partition masks, COLMAP rig
├── qoocam.py  QooCam 8K raw .MOV: reads the kfix LENS calibration into panos/lenses.json form
├── msplat.py  experimental Msplat trainer with checkpoint resume; masks via black compositing
├── composite.py `gausser composite`: visual-hull filter of the statue splat, voxel carve of the context, PLY merge
├── align.py   `gausser align`: fits a similarity (multi-start trimmed ICP) of a splat onto a published one; splat-transform applies it
├── view.py    `gausser view`: serves a SuperSplat build plus one splat on 127.0.0.1 (`serve()` is shared with orbit)
├── orbit.py   `gausser orbit`: camera path and up from sparse/0 (cameras' level right axes); runs orbit.mjs with node
├── orbit.mjs  Playwright driver: sets camera keys in SuperSplat and saves its `render.video` download
├── monitor.py `gausser monitor` / `--monitor`: read-only HTTP JSON API + the built page in web/ (gitignored)
web/           the monitor page: Vite + React + three/R3F + Spark (splats) + recharts; `npm run build` → src/gausser/web/
└── scene.py   Scene paths, image listing, replace()/remove(), run_tool()
```

## Rules

- Files are the state. A step writes under `.gausser/tmp/` and `replace()`s
  its output into place only after success. `run` skips a step that is up to
  date (`Scene.masks_done/solve_done/train_done/lod_done`: masks per image stem,
  database image names == current images, export mtime > model mtime, LOD mtime >
  export mtime). Once any step runs, every later step runs too. Don't use folder
  mtimes, because Finder's `.DS_Store` writes change them.
- Changing an input deletes what was built from it. New frames delete the masks,
  `database.db`, `sparse/` and `dense/`; new masks delete `database.db`, `sparse/` and `dense/`
  unless `sparse/no-masks` marks a `solve --no-masks` model; `solve` deletes `dense/`.
  Never delete `runs/`.
- `--no-masks` (`run`, `solve`, `train`) ignores existing masks: no COLMAP `mask_path`, no `masks` link in
  Brush's staging folder, no Msplat compositing. `mask --miss drop` writes a black Brush mask and a white
  COLMAP mask for an image SAM3 misses (default `keep`: white in both).
- `frames` (keyframes.py): the analysis and extraction ffmpeg passes use the same input options (`-ss/-to`, `-i`,
  autorotate, `-fps_mode passthrough`), so the `select` frame numbers match; a test checks this against pixel
  content. Frame names are `frame_<n>.jpg` (n relative to the range start; `pNN_` prefix with several parts).
  The ffmpeg filter graph is read from a file (`-/filter_complex`, ffmpeg ≥ 7). No hwaccel: it gave no speedup.
- Explicit step commands always run. Only `run` skips steps.
- Entry: `gausser`, `gausser FOLDER` or `gausser VIDEO` (one argument that isn't a
  command and that exists or looks like a path) opens `guide.guided` in a terminal.
  It returns an argv, which is printed as `→ gausser …` and then parsed like typed
  input, so the menu never adds behaviour the CLI lacks. A video gets a scene
  folder `<stem>/` next to it, which is removed again if the menu is cancelled
  while it is still empty. Without a TTY the menu refuses (exit 2).
- `solve` publishes the model, then raises if fewer than half the images were
  registered, so `run` stops before a wasted training run.
- Every command except `status`, `view`, `orbit`, `monitor` and `align` (which has no scene) holds `Scene.lock()` (flock on
  `.gausser/lock`). A second command fails fast; it never waits. The lock file holds the holder's
  pid/argv as JSON (opened `a+`, so a refused command does not erase it).
- `gausser monitor` only reads. It must never take the lock (not even briefly to probe it): a running
  holder is the lock file's pid being alive. Each locked step runs inside `Scene.step()`, which appends
  start/end records to `.gausser/history.jsonl` and tees everything printed, and `run_tool` output
  (through a pty on a terminal, so progress bars still draw), into `.gausser/logs/<time>-<step>.log`.
  Nothing in the pipeline reads these. Live data comes from staging folders: Brush's
  `.gausser/tmp/brush-*/exports/*.ply` and the COLMAP mapper's `snapshots/` (~20 per solve,
  `--Mapper.snapshot_frames_freq`). Scenes run by a gausser without these files show the step only
  from its recently written staging folder (`monitor.legacy_step`).
- One monitor (port 8737, `~/.local/share/gausser/monitor.json`) serves every scene in the index
  `~/.local/share/gausser/history.jsonl` (`GAUSSER_HOME`; `scene.index_path()`): every `Scene.step()`
  record is also appended there with its `scene`, and `gausser monitor` adds an `opened` record. API
  requests name their scene (`?scene=`); folders not in the index are refused. Tests set `GAUSSER_HOME`
  to a temporary folder (autouse fixture). The end record of `train`/`lod` carries `run` (the run folder).
- `train --eval-split N` adds `--eval-save-to-disk`: Brush writes `exports/eval_<step>/images/<stem>.png`;
  the monitor computes PSNR against `images/<stem>.*` itself (Brush logs only the mean).
- Log parsing for progress and stats lives in `web/src/parse.ts` only; the CLI still passes tool output
  straight through.
- `orbit` renders in a headed Chromium via `playwright-core` (installed under `~/.local/share/gausser/node`,
  `GAUSSER_NODE_DIR` overrides; Playwright's pinned Chromium revision must be installed, or set `GAUSSER_CHROME`).
  It takes no lock because it only adds `runs/orbit/<time>/`; the driver is not run by tests.
- Brush scans every file under its input folder and picks the `cameras.bin` with
  the most registered images (on a tie, the lowest path). It also matches any
  `masks` folder. That includes Msplat dataset copies, failed attempts and the old
  `.gausser/invalidated` archives. `train_brush` therefore passes a staging folder
  that only links `images`, `sparse` and `masks`; never pass the scene root.
- External tools are found on PATH with `require_tool()` and run through
  `run_tool()`, with output passed straight through. Don't add output parsers or
  progress UIs to the CLI; the monitor page parses the step logs.

- OpenMVS (`dense`, `mesh`; built by `scripts/install-openmvs.sh` at a pinned commit, tools linked into
  `~/.local/bin`) runs with `cwd` = the folder of its `.mvs` project: image paths in it are relative.
  `dense` passes `--remove-dmaps 1` (the depth maps were 7.5 GB for 350 frames). Its PLY ends each vertex in
  two variable-length lists (views, weights); `steps.read_dense_cloud` walks them. Brush takes initial points
  only from a COLMAP `points3D.bin`, so dense init writes one (empty tracks) into Brush's staging `sparse/0`,
  next to links to the real cameras; the run gets an `init-dense`/`init-sparse` marker.

## Hard-won constraints

- Verified end to end on 2026-10-19 with 158 fountain frames: SAM3 masked 157/158,
  COLMAP 4.2.1 registered 157/158, Brush and Msplat trained, and Msplat Ctrl+C/--resume,
  `frames` and `lod` all worked.
- SAM3 is pinned to commit `f6e51f59500a87c576c2df2323ce56b9fd7a12de`. The old pin
  `d7a33bb` no longer exists upstream. Upstream is CUDA-only (it imports `triton` and
  uses `device="cuda"`), so after every SAM3 install apply `patches/sam3-macos.patch`
  with `patch -p1` inside the installed `sam3/` package. The `sam` extra lists SAM3's
  undeclared imports: einops, pycocotools, psutil and scipy. numpy must stay <2.
  The default checkpoint is `~/.cache/gausser/sam3.pt`, a 3.2 GB hard link to
  `~/Developer/Automask-GS/models/sam3.pt`; without it SAM3 needs a Hugging Face login.
  `sam.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` before importing torch;
  without it, SAM3 misses every image on MPS.
- A masked Brush run on an unmasked solve (`mask --miss drop`, `solve --no-masks`, `train`) starts from
  the whole scene's COLMAP points and keeps ~⅓ of its splats as opaque background garbage that masked loss
  never removes, so `composite` must filter the statue by `hull_keep` (inside the dilated mask in ≥80% of
  ≥5 views, black masks skipped) before carving the context. `train_brush` writes an empty `masked` or
  `unmasked` file into each run for `composite`'s default selection. Negative `--dilation` erodes Brush masks only.
- `mask P --exclude` writes only `masks-exclude/` (black = P grown by `--dilation`, white inside `masks/`); `train --exclude-masks` links it as Brush's `masks` and marks the run `excluded`, which `composite` accepts as a context run like `unmasked`. `composite` defaults: `--slack 2` (hull dilation, px at 1/4 size), `--grow 2` (carve voxels).
- Mask naming: Brush `masks/<stem>.png`, COLMAP `colmap/masks/<name><ext>.png`.
  Image stems must be unique.
- Brush is pinned in `scripts/install-macos.sh` (`BRUSH_REV`, currently 1.0.0 @1388f74c);
  gausser's flags are verified against it. Keep the 1.5M `--max-splats` default,
  because unbounded density stalls Metal. Brush exports at the final step; the old
  "export-every < iterations" rule is obsolete. Masks put Brush in
  `alpha-mode masked` (loss ignored outside the mask, not trained transparent).
  This was verified: all-black masks mean no growth and opacity falls to 0.
  `train_brush` sets `RUST_LOG=brush_cli=info` unless RUST_LOG is already set;
  without it a non-TTY run prints nothing, including the `--eval-split` PSNR/SSIM lines.
- COLMAP: `feature_extractor` gets `--image_list_path` (the filtered image names), because exFAT drives
  add `._*` AppleDouble sidecars that COLMAP would otherwise try to read as images (hundreds of
  `BITMAP_ERROR`s). The mapper needs no list: it reads only what is in the database.
  `ImageReader.single_camera=1`, GPU off on macOS (Homebrew has no CUDA),
  largest sub-model chosen by the `images.bin` header becomes `sparse/0`.
  Exhaustive matching on >200 images takes hours, so `auto` switches to sequential.
  Sequential always adds `--SequentialMatching.loop_detection 1`; COLMAP 4.2 downloads
  the vocabulary tree itself into `~/.cache/colmap`.
- splat-transform ≥3.10: use `--lod-chunk-count/--lod-chunk-extent`. `-C` now
  means `--filter-cluster`.
- `splat-transform` `-r/-t` act in a frame turned 180° about z: file rotation M is given as euler of
  D·M·D and translation as (-tx,-ty,tz), with D = diag(-1,-1,1); `-s -r -t` run in order, so they give
  x' = s·R·x + t. `align` re-reads its output and checks it against numpy, so a CLI change fails loudly.
- 360° scenes (`panos/` exists): views are `images/cNN_<frame>.jpg`; COLMAP's rig groups a frame's views by
  stripping the `cNN_` image prefix. `rig_configurator` needs one camera per prefix, so `solve` extracts with one
  shared SIMPLE_PINHOLE camera and `pano.split_cameras` copies it per view index before configuring. Feature
  masks = the view's partition (each pano pixel in one view, as pycolmap does) AND colmap/masks AND masks-exclude.
  `pano.py` uses einsum, not `@`: numpy 1.26's Accelerate matmul emits spurious divide-by-zero warnings on macOS.
- QooCam 8K calibration (kfix `LENS`, also `LENS=` in photo JPEG headers): x_cam = R·X + t, t in cm. The camera
  stores every lens image vertically flipped, i.e. MIRRORED, relative to it (the lenses agree to 0.2° across their
  overlap only with that flip; ~5° without). `frames` vflips each lens image on ingest (ffmpeg `vflip`, photos in
  `qoocam.side_by_side`); treating the stored images as true via a flipped calibration gave mirrored splats. Rig frame = lens 0's camera frame; the
  grip is +x there. A fisheye rig keeps VIEWS = 14 (2 × 7) so view names and counts match equirect scenes.
- `solve` levels every 360° model (`pano.level_transform` → `colmap model_transformer`, format
  "scale qw qx qy qz tx ty tz"): down = normal of the camera-centre plane, signed by the frames' own down
  (pano +y, fisheye grip); gravity becomes +y. `orbit.level_up` (also used by the monitor) falls back to -y
  when the cameras' right axes are not level, as with fisheye ring views.
- Msplat requires exactly `msplat==1.1.4` and a binary model. It has no mask input.
  When every image has a mask, `msplat.train` writes composited copies (black
  background) into the run's `dataset/`, sets `bg_color` to black and records
  `"masked": true` in `settings.json`.

## Checks

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
.venv/bin/mypy src
(cd web && npm run build)   # type-checks and builds the monitor page into src/gausser/web/
```

`cd web && npm run dev` serves the page with hot reload; point its `/api` proxy (vite.config.ts) at the running monitor.

Set up everything with `scripts/install-macos.sh`. It is safe to rerun, and it
creates `.venv` with Homebrew Python 3.13, because msplat has only cp312/cp313
wheels. There is no CI. Tests stub external tools
through `steps.require_tool`/`steps.run_tool`; they never need COLMAP, Brush or SAM3.
