# Textual TUI contract

`gausser` will be an optional Textual control surface launched in a scene folder. It is an adapter over the existing scene paths, manifest, stage CLIs, and shared process runner; widgets never build or own subprocesses.

## First release

- Detect the current folder as a scene; display images, masks, COLMAP model, and Brush runs.
- Preflight SAM3, COLMAP, Brush, experimental Msplat, and input-artifact readiness without changing disk state.
- Run SAM3, COLMAP, Brush, or guarded experimental Msplat individually, or execute the valid ordered pipeline.
- Show stage state, current command, live process output, retained failure tail, and explicit cancellation.
- Offer only explicit mutations: initialize a scene, replace existing outputs, or start a run after confirmation.

Sharp Frames remains outside the first UI release. Msplat has no native training-mask semantics, so the UI labels it unmasked and requires a second explicit confirmation before launching it; it cannot substitute for Brush in masked scenes.

## Engine boundary

The UI calls a `SceneController` interface with preflight and stage-request methods. The controller records state in `.gausser/manifest.json` and delegates command execution to `gausser.process.run_command`. A headless CLI uses the same controller. The UI uses Textual workers only to await controller operations and receives updates as messages; it never updates widgets from worker threads.

## States

Each stage is `unavailable`, `ready`, `running`, `completed`, `failed`, or `stale`. Dependency checks determine readiness: SAM3 needs images; COLMAP needs images (and optionally masks); Brush needs a valid COLMAP model; Msplat needs images, a valid binary COLMAP model, its Python package, and explicit unmasked acknowledgement; pipeline controls stop after a failure. Existing artifacts require an explicit policy rather than automatic deletion.
