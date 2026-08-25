# gausser — agent instructions

## What this is

Scene-owned stages that turn photos into 3D Gaussian Splat `.ply` files:
`sammer` → `colmapper` → `brusher`, with optional `sharpframes`, experimental
unmasked `msplatter`, and a Textual `gausser` controller. Each shares one scene
layout and can run independently.

## Entry points

| Command | Entry | What it does |
|---------|-------|-------------|
| `sammer` | `sammer.cli:main` | SAM3 text-prompted masking → `masks/` + `colmap/masks/` |
| `colmapper` | `colmapper.cli:main` | COLMAP SfM → `sparse/0/` |
| `brusher` | `brusher.cli:main` | Brush 3DGS training → `runs/brush/<run-id>/exports/` |
| `sharpframes` | `sharpframes.cli:main` | Optional selection into `images/` |
| `msplatter` | `msplatter.cli:main` | Experimental unmasked Msplat → `runs/msplat/<run-id>/` |
| `gausser` | `gausser.tui:main` | Textual scene preflight/controller |

All commands are registered in `pyproject.toml` via `[project.scripts]`.
`src/` layout, setuptools build. No `sam3` in pyproject deps — it is installed
manually; Msplat is also optional.

## Critical quirks

- **`PYTORCH_ENABLE_MPS_FALLBACK=1`** — Without this, SAM3 silently fails on every
  MPS image (0/527 hits). sammer/cli.py sets it at module import time (before torch
  loads). The `~/.cargo/bin/sammer` wrapper also sets it. If running sammer outside
  the wrapper, the env var must be set.

- **SAM3 pin** — Must use `@d7a33bb3a6a023db88a9` commit. Newer commits import
  `triton` (CUDA-only) at module load time and crash on macOS.

- **Brush `--export-every` must be < `--total-train-iters`** — Brush iterates
  `0..N-1`. Exporting at step `N` produces no file.

- **COLMAP multi-sub-model** — colmapper reads `images.bin` header (`struct.unpack("Q", ...)[0]`)
  to pick the largest model by registered image count, not alphabetically. Moves
  chosen model to `sparse/0/` for Brush compatibility.

- **`ImageReader.single_camera=1`** — Hardcoded in colmapper pipeline. No flag to
  disable it.

- **Mask naming** — Brush: `masks/<stem>.png` (stem match, any source extension).
  COLMAP: `colmap/masks/<name><ext>.png` (double extension, required by
  `--ImageReader.mask_path`).

- **`--matcher sequential` for video frames** — Exhaustive on 500+ images generates
  125k+ pairs and hangs for hours on MPS. colmapper warns on >200 images with
  exhaustive.

## Running

```bash
cd my_scene
sammer --prompt "fountain" --dilation 8 --feather 12
colmapper --matcher sequential
brusher
```

`sammer` skips a complete matching mask set unless `--force`; `colmapper`
requires explicit `--existing resume|replace` for owned artifacts; Brush and
Msplat always create fresh run directories. Use `--score-threshold 0.3` if
SAM hit rate is low.

## Code layout

```
src/
├── gausser/ui.py           shared: IMAGE_EXTENSIONS, console, count_images, make_summary_table
├── sammer/{cli,engine}.py  SAM3 masking
├── colmapper/{cli,pipeline,runner}.py  COLMAP subprocess runner + output parsing
└── brusher/{cli,runner}.py  Brush subprocess runner + output parsing
```

Pytest coverage lives in `tests/`; use `pytest -q`. There is no CI, linter, or
type checker. Source changes take effect immediately after `pip install -e .`.

## External deps (must be on PATH)

- `colmap` — `brew install colmap`
- `brush` — Rust workspace, `cargo build --release`
- `sam3` — `pip install git+https://github.com/facebookresearch/sam3.git@d7a33bb3a6a023db88a9`
- `sharp-frames` plus `ffmpeg`/`ffprobe` — optional Sharp Frames stage
- `msplat` — optional, Apple Silicon/macOS 14+ only

SAM3 also needs `huggingface-cli login`. Msplat does not consume SAM/Brush
masks; `msplatter` is explicitly unmasked and requires binary COLMAP output.

## Wrapper scripts

`~/.cargo/bin/{sammer,colmapper,brusher}` are thin wrappers that activate the
Automask-GS venv before running. They take priority over pyenv shims because
`~/.cargo/bin` is earlier on PATH.

## What to check before editing

- Read `/Users/narqulie/AGENTS.md` for workstation conventions (pyenv, Python version).
- `gausser/ui.py` has shared helpers used by all three tools.
- `pip install -e ~/Developer/gausser` to apply changes.
