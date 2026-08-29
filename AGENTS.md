# gausser — agent instructions

## What this is

Scene-owned stages that turn photos into 3D Gaussian Splat assets:
`sammer` → `colmapper` → `brusher`, with optional `sharpframes`, experimental
unmasked `msplatter`, and explicit `lodder`. The independently runnable stage
CLIs are the primary product. `gausser` is a normal-screen Rich/questionary
launcher that delegates to them and provides three explicit fail-fast chains;
it is not a pipeline authority or persistent TUI. Native Gausser Studio is
optional and secondary; the former Textual TUI was retired.

## Entry points

| Command | Entry | What it does |
|---------|-------|-------------|
| `sammer` | `sammer.cli:main` | SAM3 text-prompted masking → `masks/` + `colmap/masks/` |
| `colmapper` | `colmapper.cli:main` | COLMAP SfM → `sparse/0/` |
| `brusher` | `brusher.cli:main` | Brush 3DGS training → `runs/brush/<run-id>/exports/` |
| `sharpframes` | `sharpframes.cli:main` | Optional selection into `images/` |
| `msplatter` | `msplatter.cli:main` | Experimental unmasked Msplat → `runs/msplat/<run-id>/` |
| `lodder` | `lodder.cli:main` | Streamed PlayCanvas LOD → `runs/lod/<run-id>/` |
| `gausser` | `gausser.launcher:main` | Interactive scene overview and stage launcher |
| `gausser-engine` | `gausser.engine:main` | JSON-lines service for optional Studio only |

All commands are registered in `pyproject.toml` via `[project.scripts]`.
`src/` layout, setuptools build. Torch is in the `sammer` extra; SAM3 itself is
installed manually. Msplat is optional through the exact `msplat==1.1.4` extra.

## Critical quirks

- **`PYTORCH_ENABLE_MPS_FALLBACK=1`** — Without this, SAM3 silently fails on every
  MPS image (0/527 hits). sammer/cli.py sets it before lazily importing Torch.
  The `~/.cargo/bin/sammer` wrapper also sets it. If running sammer outside
  the wrapper, the env var must be set.

- **Stage subprocess PATH** — Pyenv prepends its selected interpreter's `bin`
  directory inside Python processes, ahead of the user's shell PATH. The shared
  `gausser.process.command_environment()` removes only that synthetic prefix so
  launcher/controller children resolve the same `~/.cargo/bin` SAM/Colmap/Brush
  wrappers as direct shell commands. Launcher/Studio preflight validates those
  spawned wrappers; bare direct CLIs validate their current interpreter/process
  and only external dependencies. Preserve explicit `env["PATH"]` overrides.

- **SAM3 pin** — Must use `@d7a33bb3a6a023db88a9` commit. Newer commits import
  `triton` (CUDA-only) at module load time and crash on macOS. Install the
  `sammer` extra: it pins the compatible runtime stack to NumPy 1.x,
  OpenCV 4.10–4.11, setuptools below 81 (SAM3 imports `pkg_resources`), Torch
  2.10, and torchvision 0.25.

- **Msplat runtime and resume contract** — `msplatter` requires exactly Msplat
  1.1.4. A stable `--run-id` plus identical runtime/input/sparse/config identity
  resumes one staging attempt. `GaussianTrainer.step()` checkpoints into two
  atomically recorded generations every `--checkpoint-every` steps (default
  1000); bounded CPU format/hash validation occurs before upstream load. The
  final run includes `checkpoint-receipt.json`. Optimizer state/iteration resume
  exactly, but upstream does not serialize camera-shuffle RNG/order.

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
gausser  # optional guided command palette

# Bare direct commands open their stage-specific guided configurator:
sammer
colmapper
brusher

# Any explicit argument selects deterministic, non-interactive operation:
sammer --prompt "fountain" --dilation 8 --feather 12
colmapper --matcher sequential
brusher .
```

Zero-argument stage invocation is interactive only when stdin and stdout are
TTYs; otherwise it fails before inspection or mutation. Explicit arguments,
including `--help` and utility flags, must never prompt. Shared prompt and typed
option behavior lives in `gausser.interactive`; cancellation before execution
must not acquire the scene lock or create `.gausser/` metadata. Readiness exposes
one `required_action` (`none`, `force`, or `existing_policy`); `gausser-engine`
revalidates it under the scene lock before recording or launching a run.

The launcher must resolve options through `gausser.stage_options`, preview and
execute through `SceneController`, and refresh readiness between chain stages.
It must never call stage `main()` functions, acquire a chain-wide scene lock,
write manifests, or duplicate publication logic.

Human-facing CLI chrome is centralized in `gausser.ui`: the launcher and all six
stage CLIs use the Synthwave ’84 palette, shared headers, status text, summary
cards, and progress components. Preserve textual glyph/labels so state is not
color-only, let Rich honor redirected output and `NO_COLOR`, and render raw
subprocess lines as literal `Text` rather than markup. Never apply presentation
styling to `gausser-engine` stdout; it is a JSON-lines protocol.

`sammer` skips a complete matching mask set unless `--force`; `colmapper`
requires explicit `--existing resume|replace` for owned artifacts. Brush always
creates a fresh run; Msplat may resume its matching unpublished staging attempt
but publishes a fresh immutable run. Use `--score-threshold 0.3` if SAM hit rate
is low.

## Code layout

```
src/
├── gausser/launcher.py     lightweight all-stage command palette
├── gausser/interactive.py  shared bare-stage/launcher prompt configurator
├── gausser/controller.py   shared command construction and process execution
├── gausser/stage_options.py  authoritative interactive option catalog
├── gausser/ui.py           shared Rich and image-discovery helpers
├── sammer/{cli,engine}.py  SAM3 masking
├── colmapper/{cli,pipeline,runner}.py  COLMAP subprocess runner + output parsing
└── brusher/{cli,runner}.py  Brush subprocess runner + output parsing
```

Every mutating stage CLI acquires `.gausser/scene.lock` and publishes through
`gausser.publish.publish_stage`. Replacing images, masks, or reconstruction
transactionally archives incompatible dependents under `.gausser/invalidated/`;
manifest failure rolls canonical artifacts back and preserves staging.

Pytest coverage lives in `tests/`; use `pytest -q`. Python changes must also pass
`ruff check src tests monitor_and_run.py`, `mypy src`,
`./scripts/check-python310.sh` (the Docker-backed minimum-version gate), and
`PYTHON=.venv/bin/python ./scripts/build-package.sh` (clean wheel/sdist build,
content inspection, isolated installation, all-entry-point help smoke, and
non-TTY contract checks). There is no CI. Source changes take effect
immediately after `pip install -e .`.

## Native Gausser Studio

`macos/GausserStudio/` is the SwiftUI macOS client. It must remain a client of
`gausser-engine`; Python owns scene paths, manifests, command construction, and
process cancellation. Run native checks from that directory:

```bash
swift test
swift build -c release
```

The MetalSplatter viewer supports PLY/SPZ/`.splat` and requires macOS 15+.
Its Compose workspace keeps origin, orbit pivot, named focus points, and portable
camera views in artifact-local coordinates. Python atomically persists complete,
revision-checked, SHA-256-bound profiles under `presentations/` while Swift owns
camera interaction and projects RGB origin axes plus focus labels over the
splat; immutable artifact bytes are never transformed or rewritten.
`macos/package-app.sh` creates an ad-hoc-signed local developer app using the
active Python environment's `gausser-engine`; it is not a distributable bundle.

## External deps (must be on PATH)

- `colmap` — `brew install colmap`
- `brush` — Rust workspace, `cargo build --release`
- `sam3` — `pip install git+https://github.com/facebookresearch/sam3.git@d7a33bb3a6a023db88a9`
- `sharp-frames` plus `ffmpeg`/`ffprobe` — optional Sharp Frames stage
- `msplat==1.1.4` — optional extra, Apple Silicon/macOS 14+ only

SAM3 also needs `huggingface-cli login`. Msplat does not consume SAM/Brush
masks; `msplatter` is explicitly unmasked and requires binary COLMAP output.

## Wrapper scripts

`~/.cargo/bin/{sammer,colmapper,brusher}` are thin wrappers that activate the
Automask-GS venv before running. They take priority over pyenv shims because
`~/.cargo/bin` is earlier on PATH.

## What to check before editing

- Read `/Users/narqulie/AGENTS.md` for workstation conventions (pyenv, Python version).
- `gausser/ui.py` owns shared Rich presentation, the Synthwave ’84 palette, and
  image-discovery helpers used by the stage CLIs. Add visual regressions to
  `tests/test_ui.py` and retain literal raw subprocess output.
- `pip install -e ~/Developer/gausser` to apply core changes; use
  `pip install -e '~/Developer/gausser[sammer,test,quality]'` for Sammer and checks.
