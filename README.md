# Gausser

Gausser is a CLI-first collection of scene-owned tools for turning photographs
into 3D Gaussian Splat assets. Each stage is an independent command. Run only
the stages you need, inspect their files directly, and rerun them explicitly.

```text
sharpframes (optional) → sammer → colmapper → brusher
                                      └──────→ msplatter (experimental, unmasked)
training export ─────────────────────────────→ lodder (optional)
```

The direct CLIs are the primary product. `gausser` is a lightweight interactive
launcher for those same commands: it shows scene readiness, lets you choose a
step, previews the exact command, and returns to the normal terminal after each
run. The native Gausser Studio app is an optional secondary client. The former
persistent Textual TUI has been retired.

## Install

Gausser requires Python 3.10 or newer. From this repository:

```bash
python -m pip install -e .
```

Install development checks and Sammer's heavyweight Torch runtime when needed:

```bash
python -m pip install -e '.[test,quality,sammer]'
```

Msplat is optional and Apple-Silicon-only. Install the checkpoint-compatible
runtime pinned by Gausser when using `msplatter`:

```bash
python -m pip install -e '.[msplat]'
```

SAM3 itself is installed separately because its distribution and model access
are upstream-controlled. On macOS, use the known-compatible commit:

```bash
python -m pip install \
  'git+https://github.com/facebookresearch/sam3.git@d7a33bb3a6a023db88a9'
hf auth login
```

External executables are stage-specific:

| Stage | Required executable |
|---|---|
| `sharpframes` | `sharp-frames`, plus `ffmpeg` and `ffprobe` |
| `sammer` | SAM3 Python package and authenticated model access |
| `colmapper` | `colmap` |
| `brusher` | `brush` |
| `msplatter` | optional `msplat==1.1.4` Python package |
| `lodder` | `splat-transform` |

## Ordinary workflow

A scene is a directory whose canonical inputs live in `images/`:

```bash
mkdir -p my-scene/images
cp /path/to/photos/* my-scene/images/
cd my-scene
```

Then run a stage without arguments for guided configuration:

```bash
sammer
colmapper
brusher
```

Each bare stage command inspects the current scene, validates its current
process plus external dependencies, starts from the authoritative recommended
preset, asks for required values, previews the resolved shell command, and
confirms before mutation. It does not require its own launcher executable to be
rediscovered on `PATH`. Existing images or masks require explicit transactional
replacement; existing complete or partial COLMAP artifacts require resume or
replacement selection. Sharp Frames sources and explicit COLMAP mask or
vocabulary-tree paths must exist before a command can reach confirmation.

Any supplied argument keeps the established deterministic, non-interactive CLI
contract:

```bash
sammer --prompt "fountain" --dilation 8 --feather 12
colmapper --matcher sequential
brusher .
```

Bare commands prompt only when stdin and stdout are attached to a real terminal.
In redirected or automated execution they fail clearly instead of hanging or
starting expensive default work. Automation must therefore pass explicit
arguments (a scene path such as `brusher .` is sufficient where defaults are
intended). `--help` and other explicit utility flags never prompt. Back, EOF, or
Ctrl-C before launch leaves the scene unchanged.

Use `--score-threshold 0.3` when SAM's default threshold misses too many images.
Video frames generally need `colmapper --matcher sequential`; exhaustive
matching scales quadratically and is unsuitable for hundreds of adjacent
frames.

### Guided launcher

Run `gausser` inside an existing scene, or pass its path:

```bash
gausser /path/to/scene
```

The launcher renders a compact Rich overview and offers all six stage commands.
Unavailable stages remain visible with their readiness reason. Launcher and
Studio preflight validate the wrapper command they will actually spawn, while a
direct CLI validates its already-running interpreter and external dependencies.
The launcher shares the same extracted configurator and Python-owned option
catalog used by bare stage commands and Studio; required values are requested,
the shell-quoted command is shown, and execution still goes through the real
stage CLI.

The launcher and six human-facing stage CLIs share a Synthwave ’84 terminal
presentation: neon cyan/magenta/violet headers, readable state cards, consistent
progress bars, and explicit success/warning/error glyphs. Color supplements the
text instead of carrying state by itself. Rich automatically omits ANSI color in
redirected output and respects `NO_COLOR`; child-process log payloads are
rendered literally rather than interpreted as Rich markup. `gausser-engine` remains deliberately
unstyled because its stdout is a machine-readable JSON-lines protocol.

Three explicit, fail-fast sequences are available:

- **Masked Brush:** `sammer → colmapper → brusher`
- **Unmasked Msplat:** `colmapper → msplatter`
- **Brush to streamed LOD:** `brusher → lodder`

Every stage is configured and confirmed just before it runs. Scene preflight
uses one explicit required action (`none`, `force`, or `existing_policy`), and
the engine revalidates that action and runtime readiness after acquiring the
scene lock. The launcher does not hold a chain-wide lock, create pipeline state,
or infer success from files; each CLI retains its normal validation, scene lock,
publication, and manifest behavior. Back, EOF, or Ctrl-C at a launcher prompt
performs no mutation. Ctrl-C
after a stage starts cancels its process group: canonical published artifacts are
protected, while incomplete `.gausser/staging/` output may remain for diagnosis.
Automation should continue to use the direct commands.

Each command accepts an explicit scene when run elsewhere:

```bash
sammer --scene /path/to/scene --prompt "statue"
colmapper --scene /path/to/scene --matcher sequential
brusher /path/to/scene
```

### Optional Sharp Frames input

`sharpframes` transactionally replaces `images/` with a selected frame set:

```bash
sharpframes /path/to/video.mp4 --scene /path/to/scene
```

Use `--force` only when deliberately replacing a published image generation.
The command invokes upstream Sharp Frames with `--force-overwrite`.

### Experimental Msplat training

Msplat cannot consume SAM/Brush masks. It is never a silent substitute for
masked Brush training and requires explicit acknowledgement. An unattended
30,000-step run uses a stable run identity and explicit checkpoint cadence:

```bash
msplatter /absolute/path/to/scene \
  --run-id statue-001 \
  --iterations 30000 \
  --num-downscales 0 \
  --downscale-factor 4 \
  --checkpoint-every 1000 \
  --allow-unmasked
```

Rerun that exact command after a signal or failure. Gausser resumes only when
the run identity, Msplat 1.1.4 runtime, image and binary sparse-model hashes,
and complete resolved configuration are unchanged. A mismatch is an error; it
never silently restarts or overwrites the staging attempt.

Checkpoints rotate between two generations. Each save goes to a temporary file,
is closed and synced, receives a bounded CPU structural validation and SHA-256,
and is promoted before the atomic resume receipt advances. If the newest slot
is truncated or its hash differs, the prior slot is used. Checkpoint log lines
include duration and size because densified checkpoints can be several GB.
Successful runs publish `checkpoint.msplat` and an immutable
`checkpoint-receipt.json` containing the complete checkpoint history.

Msplat requires binary `sparse/0` output and does not expose a maximum-splats
control upstream. Its checkpoint restores model and optimizer state at the exact
iteration, but upstream does not serialize camera-shuffle RNG/order. Resumed
training is therefore optimizer-complete, not bit-for-bit trajectory identical.

### Optional streamed LOD

`lodder` accepts an explicit scene-owned PLY or SOG export:

```bash
lodder runs/brush/<run-id>/exports/<export>.ply
```

It validates the complete PlayCanvas streamed artifact before publishing it and
records the actual `splat-transform` version.

## Scene contract

```text
scene/
├── images/                       # flat canonical source image generation
├── masks/                        # Brush masks: <stem>.png
├── colmap/masks/                 # COLMAP masks: <name>.<ext>.png
├── database.db                   # COLMAP feature database
├── sparse/0/                     # selected COLMAP sparse model
├── runs/
│   ├── brush/<run-id>/
│   ├── msplat/<run-id>/
│   └── lod/<run-id>/
└── .gausser/
    ├── manifest.json             # stage provenance and current lineage
    ├── scene.lock                # cross-process mutation lease
    ├── staging/                  # incomplete or not-yet-published work
    └── invalidated/              # retired upstream generations/dependents
```

All mutating CLIs acquire the same scene lock. An engine-launched CLI adopts the
engine's inherited lock, so Studio and direct commands cannot concurrently
modify one scene.

Stage publication is transactional: validated files and their manifest
transition commit together. If publication metadata fails, canonical artifacts
roll back and completed staging remains available for diagnosis. Replacing an
upstream generation archives incompatible dependents rather than presenting
stale masks, reconstruction, or training runs as current:

- replacing `images/` retires masks, COLMAP state, and all runs;
- replacing masks retires COLMAP state and all runs;
- replacing reconstruction retires all training and LOD runs.

Failed external computation never replaces the last successful generation.

## Stage notes

### Sammer

- `PYTORCH_ENABLE_MPS_FALLBACK=1` must be set before Torch loads on Apple
  Silicon. The CLI sets it automatically.
- Brush masks use `masks/<image-stem>.png`.
- COLMAP masks use `colmap/masks/<image-name>.<ext>.png`.
- `--force` replaces both complete mask sets transactionally.

### Colmapper

- macOS defaults to CPU because Homebrew COLMAP has no CUDA implementation.
- Existing `database.db` or `sparse/` requires an explicit
  `--existing resume|replace` policy.
- If COLMAP emits multiple models, the model with the most registered images is
  normalized to `sparse/0/`.
- `ImageReader.single_camera=1` is currently part of the pipeline contract.

### Brusher

- `--export-every` must be less than `--total-train-iters`; Brush iterates
  `0..N-1`.
- Each invocation trains in staging and publishes a fresh
  `runs/brush/<run-id>/` only after at least one PLY exists.
- `--export-path`, when used, must name
  `runs/brush/<run-id>/exports` inside the scene.
- Brush progress can remain visually stale between upstream refinement or
  evaluation events even while training is active.

## Optional native Studio

`macos/GausserStudio/` is a SwiftUI client of the `gausser-engine` JSON-lines
service. Python remains authoritative for scene paths, manifests, typed stage
options, processes, cancellation, publication, and mutable presentation-profile
persistence. The app is optional and secondary to the stage CLIs.

The native **Compose** workspace opens scene-owned PLY, SPZ, and `.splat`
artifacts without modifying them. It can move and save an orbit pivot, set a
non-destructive presentation origin, record named focus points, and save named
portable camera views plus a home view. The viewer projects the saved origin as
RGB artifact-local axes and labels saved focus points directly over the splat.
Profiles live under `presentations/`,
are bound to the artifact path and SHA-256, and use atomic revision-checked
engine writes.

```bash
cd macos/GausserStudio
swift test
swift build -c release
```

`macos/package-app.sh` creates an ad-hoc-signed local developer app using the
active Python environment. It is not a distributable bundle.

## Legacy monitor helper

`monitor_and_run.py` remains a narrow helper for waiting on a Metashape COLMAP
export and then invoking one trainer. It delegates to the stage CLIs and is not
a second workflow authority.

## Development checks

```bash
pytest -q
ruff check src tests monitor_and_run.py
mypy src
./scripts/check-python310.sh  # Docker-backed minimum-version and package gate
PYTHON=.venv/bin/python ./scripts/build-package.sh  # build, install, and smoke wheel + sdist

cd macos/GausserStudio
swift test
swift build -c release
```

Pytest, Ruff, mypy, the Python 3.10 container gate, clean package-content build,
and the Swift build/tests are required gates for changes to their respective
Python or native surfaces.
