# Gausser workflow contract

This document defines the scene-owned workflow used by every command. It is the
implementation contract for the pipeline overhaul.

## Scene layout

```text
scene/
├── images/                         # materialized training images
├── masks/                          # Brush masks: <image-stem>.png
├── colmap/
│   └── masks/                      # COLMAP masks: <image-name>.<ext>.png
├── database.db                     # COLMAP feature database
├── sparse/0/                       # selected COLMAP model
├── runs/
│   ├── brush/<run-id>/             # PLY checkpoints, logs, run manifest
│   └── msplat/<run-id>/            # PLY checkpoints, logs, run manifest
└── .gausser/manifest.json          # authoritative workflow state
```

`images/` can be user-supplied or materialized by the frame-selection stage.
It is flat: nested paths and same-stem files with different extensions are
rejected, because Brush mask names are stem-based. The source video is not
copied by default; the manifest records its absolute path and content
fingerprint. Users who need portable scenes may copy source media themselves.

New commands write this layout, preserving the root-level `database.db` and
`sparse/` locations used by the current executable and supported by both
trainers. The README's contradictory `colmap/database.db` and
`colmap/sparse/` description will be corrected. Legacy shared `exports/` is
not silently mixed with run-scoped training output: an explicit migration
command will inspect it, adopt it as historical output, or refuse an
ambiguous scene.

## Path ownership

Every public command takes a scene root (defaulting only when omitted) and
constructs paths through one frozen `ScenePaths` value: `root`, `images`,
`brush_masks`, `colmap_masks`, `database`, `sparse`, `runs`, `manifest`, and
`staging`. External processes always receive absolute paths and an explicit
`cwd=scene_root`. No artifact path is computed from the caller's current
working directory after scene resolution.

## Manifest

`manifest.json` is versioned and atomically rewritten. It records:

- the scene format version;
- the source kind (`images` or `video`), source path, and SHA-256 fingerprint;
- materialized image inventory (relative path, SHA-256, dimensions);
- each stage's state, input fingerprint, normalized parameters, tool version,
  started/finished times, outputs, and bounded diagnostic tail; and
- trainer runs, identified by an immutable run id and backend.

A stage is exactly one of `pending`, `running`, `succeeded`, `failed`, or
`stale`. A succeeded stage is reusable only when its input fingerprint and
normalized parameters match. A changed upstream input marks dependents stale.

## Publication and failure policy

Stages write under `.gausser/staging/<stage>-<run-id>/` and atomically publish
only complete outputs. The former successful output remains available after a
failed rerun. A stage error is non-zero by default; optional best-effort modes
must be named explicitly and record skipped inputs in the manifest.

A process runner returns a structured result: argv, cwd, exit status, elapsed
time, combined output tail, and executable version. Ctrl-C terminates the
child process group, records `failed`, and leaves staging output for diagnosis.

## Trainers

Training is a backend of one scene workflow, not a parallel pipeline. Brush is
the stable masked backend. Msplat is opt-in and experimental: upstream Msplat
has no mask-input API, so `msplatter` always trains **unmasked** and requires
an explicit `--allow-unmasked` acknowledgement. It accepts only binary
COLMAP `sparse/0` models because that is the upstream loader's supported
COLMAP format. It is not offered as a transparent substitute for Brush.

Each training request creates a fresh `runs/<backend>/<run-id>/`; checkpoints
and final PLY output are never written to a shared CWD or overwritten in
place. Msplat records its output PLY, checkpoint, and optional evaluation
metrics in the scene manifest.

## Compatibility

The public `sammer`, `colmapper`, and `brusher` command names remain during
the migration, but they delegate to the shared scene model. Documentation
moves to the canonical layout. An explicit migration command, not hidden
fallbacks, handles older scenes and may adopt only complete, validated
root-level COLMAP artifacts.
