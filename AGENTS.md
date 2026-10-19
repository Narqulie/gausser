# gausser: agent instructions

## What this is

A single `gausser` CLI (`gausser.cli:main`) that turns photos or video into
Gaussian splats: `frames → mask → solve → train → lod`. It is deliberately
small. Do not reintroduce manifests, scene locks, invalidation archives, a JSON
engine, or a GUI without an explicit request from the Captain. The native
Studio app, `gausser-engine`, and the separate `sammer`/`colmapper`/`brusher`/…
commands were removed on 2026-10-19 (branch `simplify`); git history has them.

## Layout

```
src/gausser/
├── cli.py     argparse subcommands, `run` pipeline, `status`, guided questionary menu
├── steps.py   frames (sharp-frames), mask, solve (COLMAP), train_brush, lod (splat-transform)
├── sam.py     SAM3 predictor + mask writers; imported only by `mask` (needs torch)
├── msplat.py  experimental Msplat trainer with checkpoint resume; masks via black compositing
├── view.py    `gausser view`: serves a SuperSplat build plus one splat on 127.0.0.1
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
  `database.db` and `sparse/`; new masks delete `database.db` and `sparse/`.
  Never delete `runs/`.
- Explicit step commands always run. Only `run` skips steps.
- Every command except `status` and `view` holds `Scene.lock()` (flock on
  `.gausser/lock`). A second command fails fast; it never waits.
- Brush scans every file under its input folder and picks the `cameras.bin` with
  the most registered images (on a tie, the lowest path). It also matches any
  `masks` folder. That includes Msplat dataset copies, failed attempts and the old
  `.gausser/invalidated` archives. `train_brush` therefore passes a staging folder
  that only links `images`, `sparse` and `masks`; never pass the scene root.
- External tools are found on PATH with `require_tool()` and run through
  `run_tool()`, with output passed straight through. Don't add output parsers or
  progress UIs for tool output.

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
- COLMAP: `ImageReader.single_camera=1`, GPU off on macOS (Homebrew has no CUDA),
  largest sub-model chosen by the `images.bin` header becomes `sparse/0`.
  Exhaustive matching on >200 images takes hours, so `auto` switches to sequential.
  Sequential always adds `--SequentialMatching.loop_detection 1`; COLMAP 4.2 downloads
  the vocabulary tree itself into `~/.cache/colmap`.
- splat-transform ≥3.10: use `--lod-chunk-count/--lod-chunk-extent`. `-C` now
  means `--filter-cluster`.
- Msplat requires exactly `msplat==1.1.4` and a binary model. It has no mask input.
  When every image has a mask, `msplat.train` writes composited copies (black
  background) into the run's `dataset/`, sets `bg_color` to black and records
  `"masked": true` in `settings.json`.

## Checks

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
.venv/bin/mypy src
```

Set up everything with `scripts/install-macos.sh`. It is safe to rerun, and it
creates `.venv` with Homebrew Python 3.13, because msplat has only cp312/cp313
wheels. There is no CI. Tests stub external tools
through `steps.require_tool`/`steps.run_tool`; they never need COLMAP, Brush or SAM3.
