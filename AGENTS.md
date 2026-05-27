# gausser — agent instructions

## Project

Three CLI tools in one `pip install`:
- `sammer` — SAM3 segmentation (text-prompted masks)
- `colmapper` — COLMAP SfM wrapper
- `brusher` — Brush 3DGS training wrapper

`sammer` → `colmapper` → `brusher`. Each reads from/writes to the current directory.

## Install

```bash
pip install -e ~/Developer/gausser
```

Package uses `src/` layout with setuptools. Entry points in `pyproject.toml:18-21` map `sammer` → `sammer.cli:main`, etc.

## External dependencies (must be on PATH)

- `colmap` — `brew install colmap` or prebuilt binary
- `brush` — `cargo build --release` from the brush repo (25-crate Rust workspace)

## SAM3 (optional, for `sammer` only)

```bash
pip install git+https://github.com/facebookresearch/sam3.git@d7a33bb3a6a023db88a9
```

Pinned commit avoids `triton` (CUDA-only). Newer commits crash on macOS.

## Key quirks

- **`brusher --export-every` must differ from `--total-train-iters`.** Brush iterates 0…total-1; exporting at `total` produces no file.
- **Video frames → use `--matcher sequential`.** Exhaustive on 500+ frames hangs for hours on MPS.
- **`colmapper` always passes `ImageReader.single_camera=1`.** No need to set it.
- **Mask format:** Brush gets `masks/<stem>.png`, COLMAP gets `colmap/masks/<fullname.ext>.png` (double extension).
- **Resumable:** Rerun any stage without `--force` on existing outputs (`sammer` needs `--force` to re-mask).

## Running

Commands run from the scene directory (folder with `images/`):

```bash
sammer --prompt "fountain" --dilation 8 --feather 12
colmapper --matcher sequential
brusher
```

## Code layout

```
src/
├── gausser/ui.py           shared helpers (IMAGE_EXTENSIONS, console, Rich table)
├── sammer/{cli,engine}.py
├── colmapper/{cli,pipeline,runner}.py
└── brusher/{cli,runner}.py
```

No tests, no CI, no linter config, no type checking. All `__init__.py` files are empty.

## What to check before editing

- **Read `AGENTS.md` from /Users/narqulie/AGENTS.md** for workstation-level conventions (Python version, pyenv, OpenCode config).
- Dependencies are minimal: `rich`, `numpy`, `Pillow`, `torch` (no `sam3` in `pyproject.toml` — it's installed separately).
