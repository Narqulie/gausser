# gausser

Scene-owned CLI stages that turn photos into 3D Gaussian Splatting `.ply` files.

```
sharpframes (optional) → sammer → colmapper → brusher
                                      └→ msplatter (experimental, unmasked)
```

I built this because the existing pipelines were monolithic scripts that did
everything at once and were a pain to debug when something went wrong mid-way
through a 40-minute run. These three tools each do one stage, share a directory
layout, and you run them in sequence by hand. If masking works but COLMAP fails
you just fix the images and re-run colmapper, not the whole thing.

## What the three tools do

**sammer** — runs SAM3 (Meta's segment-anything model) on every photo in your
scene folder. You give it a text prompt like "fountain" or "statue" and it
writes a mask for each image showing where that thing is. Writes two sets of
masks because COLMAP and Brush want different naming conventions.

**colmapper** — wraps COLMAP's SfM pipeline. Extracts features, matches them
between images, builds a sparse point cloud. Picks the largest reconstruction
if COLMAP finds multiple (which happens more than you'd think).

**brusher** — runs Brush 3DGS training on the COLMAP output. Writes `.ply`
checkpoints every N iterations so you can stop early and still have a result.

**sharpframes** — optional pre-processing that selects frames from a video or
image source into `images/` transactionally before the reconstruction stages.

**msplatter** — an experimental Apple-Silicon Msplat backend. It creates an
isolated run with a PLY export and checkpoint, but is deliberately **not** a
replacement for Brush: upstream Msplat has no mask-input support.

Each one has a progress bar that actually shows what's happening, not just a
spinning cursor.

## When you'd use this

You have a pile of photos of something (a statue, a building, a fountain) and
you want a 3D model you can view in a splat viewer. You point sammer at the
photos, colmapper figures out where the camera was for each one, and brusher
trains the gaussian splat.

## Quick start

```bash
cd ~/scenes/my_fountain

sammer --prompt "fountain" --dilation 8 --feather 12
colmapper --matcher sequential
brusher
```

If you're working with video frames (like I mostly am) the `--matcher
sequential` flag is important — the default exhaustive matcher pairs every
image with every other image and on 500 frames that's 125k pairs, which just
hangs on MPS.

## What you need installed

- Python 3.10+ with torch
- COLMAP on PATH (`brew install colmap` or download from the website)
- Brush on PATH (needs a Rust toolchain: `cargo build --release` in the brush
  repo)
- SAM3 for sammer (`pip install
  git+https://github.com/facebookresearch/sam3.git@d7a33bb3a6a023db88a9` then
  `huggingface-cli login`)
- Optional Sharp Frames (`pip install sharp-frames`) plus `ffmpeg` and
  `ffprobe` on PATH
- Optional Msplat (`pip install msplat`) on Apple Silicon running macOS 14+;
  it requires COLMAP **binary** sparse output

The pinned SAM3 commit avoids a triton dependency issue on macOS. The latest
commit tries to import triton (CUDA-only) at module load time and crashes on
Apple Silicon.

## Textual control surface

Install the optional UI and launch it from the scene root:

```bash
pip install -e '.[tui]'
cd ~/scenes/my_fountain
gausser
```

The terminal UI shows scene/tool readiness and can run SAM3, COLMAP, Brush, an
ordered pipeline, or experimental Msplat. Enter the SAM3 prompt, then press a
stage action twice to confirm it. Msplat requires its own second confirmation:
it does not consume SAM masks and therefore deliberately trains unmasked. `Esc`
requests cancellation and waits for the shared runner to clean up the complete
process tree. `r` refreshes readiness and `q` quits. The same scene-owned CLI
commands remain available for headless use.

## Directory layout the tools expect

```
my_scene/
├── images/              ← put your photos here
├── masks/               ← created by sammer for Brush
├── colmap/
│   └── masks/           ← created by sammer for COLMAP
├── database.db          ← created by colmapper
├── sparse/
│   └── 0/
│           ├── cameras.bin
│           ├── images.bin
│           └── points3D.bin
├── runs/
│   ├── brush/<run-id>/exports/  ← created by brusher
│   └── msplat/<run-id>/          ← experimental PLY export + checkpoint
└── .gausser/
    ├── manifest.json
    └── staging/         ← incomplete work; never consumed
```

Each tool reads from this structure and writes to it. You cd into the scene
folder and run the commands. No config files, no yaml, no pipeline state.

## CLI reference

```
sammer --prompt "fountain" [--scene SCENE] [--path ./images/]
       [--score-threshold 0.5] [--device auto]
       [--dilation 8] [--feather 12] [--all-instances] [--force]

colmapper [--scene SCENE] [--path ./images/] [--output ./sparse/]
          --matcher sequential|exhaustive|vocab_tree [--vocab-tree FILE]
          [--existing error|resume|replace] [--no-gpu] [--mask-path DIR]

brusher [SCENE] [--total-train-iters 30000] [--export-every 5000]
        [--eval-every 1000] [--refine-every 200]
        [--max-splats N] [--max-resolution N] [--eval-split-every N]
        [--export-path DIR]

sharpframes SOURCE [--scene SCENE] [--force] [SHARP-FRAMES OPTIONS]

msplatter [SCENE] [--iterations 7000] [--num-downscales 0]
           [--downscale-factor N] [--no-eval] --allow-unmasked
```

## Why the name

I wanted short commands you type a lot. `-er` suffix for tools that do one
thing. They share a repo as gausser because they're always used together and
they share some UI code.

## A real run

On 527 video frames of a fountain (the dataset I built this for):

```
sammer --prompt "Fountain" --dilation 8 --feather 12
  524/527 masked, 99.4% hit rate, 50 min

colmapper --matcher sequential
  527/527 features, 527/527 matched, 524/527 registered, 25 min

brusher
  writes export_30000.ply, ~15 min
```

Three commands, about 90 minutes total, you get a .ply you can drag into
supersplat or play with in the brush viewer.

## Known quirks

- COLMAP's mapper can split your images into multiple sub-models if it thinks
  they don't connect. colmapper picks the largest one automatically.
- On macOS without `PYTORCH_ENABLE_MPS_FALLBACK=1` set, SAM3 crashes on every
  image because of an unsupported MPS operator. sammer sets this automatically
  at startup.
- Brush's `--export-every` must be less than `--total-train-iters`. It
  iterates 0..N-1 so exporting at step N produces nothing.
- macOS external volumes may contain `._*` AppleDouble metadata sidecars. Gausser
  ignores them when counting or processing images; raw file counts do not.
- Msplat cannot consume `masks/` or `colmap/masks/`. `msplatter` requires
  `--allow-unmasked`, including on scenes that contain SAM masks, so that this
  semantic difference is never implicit. It supports only upstream binary
  COLMAP models, not text sparse models. Msplat loads all source images into
  shared GPU memory; for large scenes use `--downscale-factor N` to load at
  one-Nth linear resolution (for example, `4` for quarter-resolution inputs).
