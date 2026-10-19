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

`gausser run` is unattended. It skips any step whose output already exists, so
you can rerun it after a failure. Use `--from STEP` to redo a step and everything
after it. For example, `gausser run --from solve --matcher sequential`.

Without `--prompt`, nothing is masked. Add `--video clip.mp4` to extract frames
first, and `--lod` to finish with a streamed LOD set.

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
- New frames delete the masks and the COLMAP model, and new masks delete the
  COLMAP model, because both were built from what changed. Training runs in
  `runs/` are never deleted.
- `.gausser/tmp/` keeps failed attempts for inspection. Delete it whenever
  nothing is running.

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
- **Brush.** `--export-every` must be smaller than `--iterations`, because Brush
  never reaches step N. `--max-splats` defaults to 1.5M, because unbounded growth
  stalls Metal on Apple Silicon.
- **Msplat** is experimental and ignores masks. Ctrl+C saves a checkpoint;
  continue with `gausser train --resume <run>`.

## Install (macOS, Apple Silicon)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'                    # the CLI and tests

brew install colmap ffmpeg
pip install sharp-frames
npm install -g @playcanvas/splat-transform
# Brush: build github.com/ArthurBrussee/brush with `cargo build --release`, put `brush` on PATH

# SAM3 masking (optional, about 2 GB):
pip install -e '.[sam]'
pip install git+https://github.com/facebookresearch/sam3.git@d7a33bb3a6a023db88a9
hf auth login

# Msplat (optional):
pip install -e '.[msplat]'
```

SAM3 must stay at commit `d7a33bb`, because newer commits import the CUDA-only
`triton` and crash on macOS. Gausser sets `PYTORCH_ENABLE_MPS_FALLBACK=1` itself.
Without it, SAM3 misses every image on Apple GPUs.

Run `gausser status` to check what is installed.

## Development

```bash
pytest -q && ruff check src tests && mypy src
```
