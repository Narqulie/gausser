# gausser

Photos or video in, Gaussian splats out. One command wraps five steps:

```
frames → mask → solve → train → lod
(video)  (SAM3)  (COLMAP) (Brush/Msplat) (splat-transform)
```

## Quick start

The guided way. It reads the scene, recommends the next step, asks only what that
step needs, and prints the equivalent command before running it:

```bash
gausser clip.mp4           # makes clip/ next to the video, then guides you
gausser ~/photos/statue    # a folder of photos; offers to move them into images/
gausser                    # the current folder
```

The direct way, for scripts and overnight runs:

```bash
mkdir -p statue/images && cd statue
cp ~/photos/statue/*.jpg images/        # or: gausser frames walkaround.mp4
gausser run --prompt "statue"            # mask → solve → train
gausser view                             # open the result in SuperSplat
```

A scene is a folder with its photos in `images/`. If the photos are in the
folder itself, `gausser run --import` moves them into `images/`.

`gausser run` is unattended. It skips steps that are already up to date, so you
can rerun it after a failure or after adding photos. Use `--from STEP` to redo a
step and everything after it. For example, `gausser run --from solve --matcher sequential`.
If COLMAP places fewer than half the images, `run` stops before training: the
images don't overlap enough. `gausser train` trains on them anyway.

While a command runs, gausser keeps the Mac awake with `caffeinate`, because a
sleeping Mac pauses training. Closing a laptop's lid still puts it to sleep unless
an external display is attached.

Without `--prompt`, no new masks are made, but existing masks are still used. Add
`--video clip.mp4` to extract frames first, and `--lod` to finish with a streamed
LOD set.

## Commands

| Command | Does | Writes |
|---|---|---|
| `gausser frames SOURCE [-- args]` | Sharp Frames picks sharp frames from a video | `images/` |
| `gausser mask PROMPT [--miss drop]` | SAM3 masks the subject | `masks/`, `colmap/masks/` |
| `gausser solve [--no-masks]` | COLMAP finds the camera poses | `database.db`, `sparse/0/` |
| `gausser train [--no-masks] [-- args]` | Brush (default) or `--trainer msplat` trains splats; args after `--` go to Brush | `runs/brush/<time>/exports/*.ply` |
| `gausser lod [PLY]` | splat-transform builds streamed LOD (newest export by default) | `runs/lod/<time>/` |
| `gausser view [PLY]` | Opens a splat (newest export by default) in a local SuperSplat | — |
| `gausser orbit [PLY] [--seconds 12] [--size 1920x1080] [-o OUT.mp4]` | SuperSplat renders a 360° orbit video (newest export by default) | `runs/orbit/<time>/orbit.mp4` |
| `gausser status` | Shows what the scene has and which tools are installed | — |

Every command takes `-s/--scene DIR` (default: current folder) and `--help`.
Only one command at a time can change a scene; a second one stops with an error
instead of overwriting the first one's work. `status`, `view` and `orbit` always run
(`orbit` only adds a new run folder).

`gausser orbit` needs `sparse/0` and a Brush export. The camera circles the point nearest all
camera viewing axes, at the median camera distance times `--zoom` (1.3) and aimed `--drop` (0.35, splat
units) below that point, with "up" from the PLY's `Vertical axis` header. It drives a visible Chromium
window through Playwright (installed by `scripts/install-macos.sh` into `~/.local/share/gausser/node`;
override with `GAUSSER_NODE_DIR`, and `GAUSSER_CHROME` picks another browser executable), so leave the window alone
while it renders.

## Subject plus context from one solve

Mask the subject, but solve COLMAP on the whole images: masks that restrict COLMAP
to the subject lose registrations and leave no background points. Then train twice on
that one model, so the two splats share coordinates and can be merged later:

```bash
gausser mask statue --miss drop   # frames SAM3 misses are ignored by Brush, not trained whole
gausser solve --no-masks          # COLMAP sees the whole image
gausser train --no-masks          # context splat: unmasked
gausser train                     # statue splat: masked
```

`gausser run --prompt statue --no-masks` makes the masks and then solves and trains
without them (the first two steps of the recipe, minus `--miss drop`).

## How state works

The files are the state. There is no database or manifest.

- Each step writes into `.gausser/tmp/` and moves the result into place only when
  it succeeds. A failed step leaves the previous output untouched.
- A step counts as up to date when its output matches its input:
  - masks: every image has a mask;
  - solve: the COLMAP database lists exactly the current images;
  - train: the newest export is newer than the model;
  - lod: the newest LOD set is newer than the newest export.

  `gausser status` marks anything out of date.
- New frames delete the masks and the COLMAP model, and new masks delete the
  COLMAP model, because both were built from what changed. Training runs in
  `runs/` are never deleted.
- `.gausser/tmp/` keeps failed attempts for inspection. `gausser status` shows their
  size; delete the folder whenever nothing is running.
- Brush gets a folder that holds only links to `images/`, `sparse/` and `masks/`.
  Brush scans everything in the folder it is given and would otherwise pick up
  stale COLMAP models and images from `runs/` or `.gausser/`.

## Notes that cost time to learn

- **Masks.** Brush reads `masks/<stem>.png`. COLMAP reads `colmap/masks/<name><ext>.png`;
  the double extension is required. If SAM3 finds nothing in an image, that image
  gets a white mask, so it is kept whole and the run continues (`--miss keep`, the
  default). With `--miss drop` its Brush mask is black, so Brush ignores that image,
  and its COLMAP mask stays white, so it still registers. Use `drop` when a whole-image
  fallback would train people, water or sky into the subject. Try
  `--threshold 0.3` if many images miss. `--no-masks` on `solve` and `train` ignores
  existing masks; masks are still deleted when new ones are made.
- **Matching.** `--matcher auto` uses exhaustive matching up to 200 images and
  sequential matching with loop detection above that. Exhaustive matching on 500+
  images runs for hours. Loop detection also matches distant images that look
  alike, so a walk that ends where it started, or a set of unordered photos, still
  connects. COLMAP downloads the vocabulary tree it needs once, into `~/.cache/colmap`.
  For large unordered photo sets, `--matcher vocab_tree` is the dedicated option.
- **COLMAP on macOS.** Homebrew COLMAP has no CUDA, so GPU is off on macOS. All
  images share one camera (`single_camera=1`). If COLMAP splits the scene, the
  largest model becomes `sparse/0`.
- **Brush.** `--max-splats` defaults to 1.5M, because unbounded growth stalls
  Metal on Apple Silicon. Brush exports at the final step too: 5000 iterations with
  `--export-every 2500` writes `export_2500.ply` and `export_5000.ply`.
  Masks are confirmed in use: with all-black masks Brush stops adding splats and
  their opacity falls to zero. Masked-out areas are ignored during training, not
  made transparent.
- **Quality.** `gausser train --eval-split 8` holds out every 8th image and
  prints `Eval iter N: PSNR …, ssim …` every 1000 steps and at the end. Compare
  runs with the same split.
- **Msplat** is experimental. It has no mask input, so with masks gausser
  composites each image onto black and trains against a black background. The
  masked-out areas become empty space (so `--miss drop` images become all black
  there; prefer Brush for that). Ctrl+C saves a checkpoint; continue with
  `gausser train --resume <run>`.

## Install (macOS, Apple Silicon)

```bash
scripts/install-macos.sh
```

The script is safe to rerun. It installs:

- COLMAP, ffmpeg, Python 3.13, Node and Rust (Homebrew);
- splat-transform (npm);
- a `.venv` with gausser, Sharp Frames, SAM3 (patched) and Msplat;
- Brush at a pinned commit (`cargo install`);
- a SuperSplat build for `gausser view` and `gausser orbit`, and Playwright with Chromium for `orbit`;
- `gausser` and `sharp-frames` in `~/.local/bin`.

It ends with `gausser status`. After you reinstall SAM3 by hand, rerun the script:
`status` reports an unpatched SAM3.

SAM3 is pinned to commit `f6e51f5`; the older pin `d7a33bb` no longer exists
upstream. Upstream SAM3 assumes CUDA: it imports `triton` and hardcodes
`device="cuda"`. `patches/sam3-macos.patch` adds CPU and MPS fallbacks; it is the
same patch the earlier Automask-GS setup used.

Gausser loads `~/.cache/gausser/sam3.pt` when it exists and no `--sam-checkpoint`
is given. Otherwise it downloads the gated model from Hugging Face. Gausser sets
`PYTORCH_ENABLE_MPS_FALLBACK=1` itself; without it, SAM3 misses every image on
Apple GPUs.

## Development

```bash
pytest -q && ruff check src tests && mypy src
```
