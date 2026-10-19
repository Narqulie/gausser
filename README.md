# gausser

Photos or video in, Gaussian splats out. One command wraps five steps:

```
frames → mask → solve → train → lod
(video)  (SAM3)  (COLMAP) (Brush/Msplat) (splat-transform)
```

## Quick start

```bash
mkdir -p statue/images && cd statue
cp ~/photos/statue/*.jpg images/        # or: gausser frames walkaround.mp4
gausser run --prompt "statue"            # mask → solve → train
gausser                                  # or answer a few questions instead
```

`gausser run` is unattended. It skips steps that are already up to date, so you
can rerun it after a failure or after adding photos. Use `--from STEP` to redo a
step and everything after it. For example, `gausser run --from solve --matcher sequential`.

Without `--prompt`, no new masks are made, but existing masks are still used. Add
`--video clip.mp4` to extract frames first, and `--lod` to finish with a streamed
LOD set.

## Commands

| Command | Does | Writes |
|---|---|---|
| `gausser frames SOURCE [-- args]` | Sharp Frames picks sharp frames from a video | `images/` |
| `gausser mask PROMPT` | SAM3 masks the subject | `masks/`, `colmap/masks/` |
| `gausser solve` | COLMAP finds the camera poses | `database.db`, `sparse/0/` |
| `gausser train` | Brush (default) or `--trainer msplat` trains splats | `runs/brush/<time>/exports/*.ply` |
| `gausser lod [PLY]` | splat-transform builds streamed LOD (newest export by default) | `runs/lod/<time>/` |
| `gausser status` | Shows what the scene has and which tools are installed | — |

Every command takes `-s/--scene DIR` (default: current folder) and `--help`.

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
  gets a white mask, so it is kept whole and the run continues. Try
  `--threshold 0.3` if many images miss.
- **Matching.** `--matcher auto` uses exhaustive matching up to 200 images and
  sequential above that. Exhaustive matching on 500+ video frames runs for hours.
- **COLMAP on macOS.** Homebrew COLMAP has no CUDA, so GPU is off on macOS. All
  images share one camera (`single_camera=1`). If COLMAP splits the scene, the
  largest model becomes `sparse/0`.
- **Brush.** `--max-splats` defaults to 1.5M, because unbounded growth stalls
  Metal on Apple Silicon. Brush 0.3.0 exports at the final step too: 5000
  iterations with `--export-every 2500` writes `export_2500.ply` and `export_5000.ply`.
- **Msplat** is experimental and ignores masks. Ctrl+C saves a checkpoint;
  continue with `gausser train --resume <run>`.

## Install (macOS, Apple Silicon)

```bash
brew install python@3.13                  # msplat wheels exist only for 3.12/3.13
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'                    # the CLI and tests

brew install colmap ffmpeg
pip install sharp-frames
npm install -g @playcanvas/splat-transform
# Brush: build github.com/ArthurBrussee/brush with `cargo build --release`, put `brush` on PATH

# SAM3 masking (optional, a few GB):
pip install -e '.[sam]'
pip install git+https://github.com/facebookresearch/sam3.git@f6e51f59500a87c576c2df2323ce56b9fd7a12de
patch -p1 -d "$(python -c 'import os, importlib.util as u; print(os.path.dirname(u.find_spec("sam3").origin))')" \
  < patches/sam3-macos.patch               # upstream SAM3 is CUDA-only
# then either put the checkpoint at ~/.cache/gausser/sam3.pt, or run `hf auth login`

# Msplat (optional):
pip install -e '.[msplat]'
```

SAM3 is pinned to commit `f6e51f5`; the older pin `d7a33bb` no longer exists
upstream. Upstream SAM3 assumes CUDA: it imports `triton` and hardcodes
`device="cuda"`. `patches/sam3-macos.patch` adds CPU and MPS fallbacks; it is the
same patch the earlier Automask-GS setup used.

Gausser loads `~/.cache/gausser/sam3.pt` when it exists and no `--sam-checkpoint`
is given. Otherwise it downloads the gated model from Hugging Face. Gausser sets
`PYTORCH_ENABLE_MPS_FALLBACK=1` itself; without it, SAM3 misses every image on
Apple GPUs.

Run `gausser status` to check what is installed.

## Development

```bash
pytest -q && ruff check src tests && mypy src
```
