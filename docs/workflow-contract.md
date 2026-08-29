# Gausser workflow contract

This document defines the scene and mutation invariants shared by the direct
stage CLIs and the optional Studio engine.

## Product surface

The independently runnable `sharpframes`, `sammer`, `colmapper`, `brusher`,
`msplatter`, and `lodder` commands are the primary interface. `gausser` is an
ephemeral normal-screen command palette over those CLIs, not a second workflow
implementation. `gausser-engine` is infrastructure for the optional native
Studio; direct CLIs and the launcher do not require or route through it. The
retired persistent Textual TUI is not a supported surface.

A direct stage command invoked with zero arguments in a real terminal uses the
shared `gausser.interactive` configurator: it reads preflight state, presents the
Python-owned option catalog, previews the command, and returns resolved
arguments to that stage's own parser and execution path. Direct preflight checks
the current interpreter/process and external dependencies; it does not require
the already-running entry point to be rediscovered on `PATH`. Launcher and
Studio preflight instead validate the wrapper command they will spawn. Any
supplied argument is deterministic and non-interactive. A bare non-TTY
invocation fails before scene inspection or mutation. Cancellation before
execution acquires no lock and creates no scene metadata.

The launcher uses the same configurator to preview `SceneController` commands
and execute stages sequentially. Its three chains—masked Brush, unmasked Msplat,
and Brush-to-LOD—are explicit, user-confirmed, and fail-fast. Each stage
refreshes readiness and independently acquires the scene lock; the launcher
writes no pipeline or manifest record and holds no chain-wide mutation lease.

## Canonical scene layout

```text
scene/
├── images/                         # flat canonical source image generation
├── masks/                          # Brush masks: <image-stem>.png
├── colmap/masks/                   # COLMAP masks: <image-name>.<ext>.png
├── database.db                     # COLMAP features
├── sparse/0/                       # selected sparse model
├── runs/<backend>/<run-id>/        # immutable published trainer/LOD runs
├── presentations/                 # mutable per-artifact composition profiles
└── .gausser/
    ├── manifest.json               # current provenance and engine run state
    ├── scene.lock                  # cross-process mutation lease
    ├── staging/                    # incomplete or unpublished computation
    └── invalidated/                # retired generations and their dependents
```

`images/` is flat. Nested images, AppleDouble sidecars, and same-stem files with
different extensions are rejected because Brush mask names are stem-based.
Readiness counts only supported real images, so `.DS_Store` and AppleDouble-only
directories do not become completed image generations. Every command resolves
one frozen `ScenePaths` from the owning scene and passes absolute paths plus
`cwd=scene.root` to external tools. Typed Sharp Frames sources and explicit
COLMAP mask/vocabulary-tree paths are normalized relative to the scene and must
exist before command confirmation.

## Cross-process ownership

Every mutating stage CLI acquires `.gausser/scene.lock`. The optional engine
acquires the same lock before launching a stage and passes its descriptor to the
child. The child adopts that descriptor rather than attempting a second lock.
Closing the parent descriptor does not explicitly unlock an inherited open-file
description; ownership remains until the last holder exits.

A competing process fails before validating or changing canonical stage state.
Read-only inspection need not acquire the mutation lease. Scene readiness
carries one required action (`none`, `force`, or `existing_policy`) rather than
independent booleans. The engine validates runtime readiness before reservation,
then re-inspects scene state and enforces that action while holding the mutation
lock, before it records or launches a run.

## Generations and invalidation

Three upstream generation boundaries determine compatibility:

| Generation replaced | Canonical replacement | Retired dependents |
|---|---|---|
| `images` | `images/` | masks, COLMAP state, all runs |
| `masks` | `masks/`, `colmap/masks/` | COLMAP state, all runs |
| `reconstruction` | `database.db`, `sparse/` | all training/LOD runs |

Retired artifacts move intact beneath `.gausser/invalidated/<stage>-<id>/`.
The manifest transition removes superseded current-stage records and records the
archive path. Historical files are therefore preserved without appearing to
belong to the new lineage.

Brush, Msplat, SuperSplat reimport, and LOD create fresh run directories and do
not replace an upstream generation. Their latest provenance record may advance
while older immutable runs remain scene-owned history.

## Scene composition profiles

Studio may create one versioned `presentations/<artifact-key>.json` document for
each selected scene-owned PLY, SPZ, or `.splat` artifact. The document binds to
the artifact's relative path and SHA-256 and stores only artifact-local
presentation metadata: a display origin, optional default orbit pivot, named
focus points, portable camera poses, and an optional home view. It never rewrites
or transforms the immutable splat file.

Presentation writes use `.gausser/scene.lock`, bounded structural validation,
atomic replacement, and a content-derived revision. Studio submits the revision
it read; a stale update fails rather than overwriting another editor's changes.
A changed artifact hash also fails closed. Presentation profiles are mutable
scene annotations rather than pipeline-stage outputs, so they are not published
through the manifest transaction and cannot imply successful training.

## Publication transaction

External computation writes only beneath `.gausser/staging/<stage>-<id>/`.
Before publication, the stage validates every required output. One shared
publication primitive then:

1. verifies every destination belongs to the stage's declared generation;
2. retires the previous canonical generation and incompatible dependents;
3. moves staged outputs into their canonical destinations;
4. atomically transitions `.gausser/manifest.json` while preserving its exact
   previous bytes; and
5. finalizes the archive and removes rollback scratch space.

Any ordinary failure or user interrupt in steps 2–4 restores all original
canonical paths and the original manifest bytes before propagating the error.
Completed staged output is moved back into staging for diagnosis or recovery.
Failed external computation never enters the publication transaction, so the
previous successful generation remains untouched. Abrupt termination that
cannot run Python cleanup, such as `SIGKILL` or power loss, remains outside this
in-process rollback guarantee.

## Manifest

`manifest.json` is a versioned, atomically rewritten provenance document. Stage
records contain the normalized parameters, input fingerprints, outputs, tool
identity/version when available, and invalidation archive when one was created.
The optional engine additionally records active and completed run lifecycle
state. Filesystem artifacts remain authoritative; the manifest explains their
current lineage and Studio activity rather than hiding them behind an opaque
pipeline database.

## Trainer contracts

Brush is the stable masked trainer. Each invocation publishes a new
`runs/brush/<run-id>/` after validating at least one PLY export. Explicit export
paths must resolve to `runs/brush/<run-id>/exports` inside the scene.

Msplat is experimental and unmasked because upstream exposes no mask input. It
requires `--allow-unmasked`, the exact Msplat 1.1.4 runtime, a binary `sparse/0`
model, and an explicit image downscale when the surrounding monitor helper
invokes it. `--run-id` names one staging attempt; repeating the identical command
resumes it. The runtime version, run identity, image hashes, all three binary
sparse-model hashes, and complete resolved configuration must match the atomic
resume receipt. Any mismatch fails before staging is overwritten.

Training advances with `GaussianTrainer.step()`. At `--checkpoint-every`
boundaries (1,000 by default), the completed optimizer state is written to a
temporary file, closed and fsynced, structurally validated on CPU as the bounded
18-tensor v1 format, promoted into the alternate A/B slot, hashed, and finally
recorded by atomic receipt replacement. Resume hash-validates and structurally
validates the newest slot before calling upstream `load_checkpoint`, falling
back to the prior generation when necessary. Signals are deferred until the
current step finishes and a checkpoint attempt completes; engine cancellation
grants Msplat five minutes before escalation so multi-GB checkpoints can finish.
The previous valid generation remains recoverable after an incomplete write or
forced process death.

Publication occurs only after the configured terminal iteration (30,000 for the
full-quality preset), a nonempty PLY, and a validated terminal checkpoint. The
new immutable `runs/msplat/<run-id>/` contains the PLY, materialized dataset,
`checkpoint.msplat`, `checkpoint-receipt.json` with checkpoint iteration/splat
count/size/duration/hash history, and optional metrics. Earlier successful runs
remain untouched. Upstream checkpoints preserve model and optimizer state but
not camera-shuffle RNG/order, so resume is exact at the optimizer iteration but
not bit-deterministic relative to an uninterrupted trajectory.

LOD is explicit and never automatic after training. It accepts a scene-owned PLY
or SOG, validates the complete streamed tree, records the actual
`splat-transform` version, and publishes a fresh `runs/lod/<run-id>/`.
