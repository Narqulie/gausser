# Gausser Studio stage workspaces

**Status:** approved 2026-08-26. The native Studio will replace the current
mixed dashboard controls with complete, tool-specific workspaces. Delivery
order is **shared settings foundation → Sources & Sharp Frames → COLMAP → SAM3
→ Brush → Msplat → LOD/publish**. The local full-pipeline bundle resumes once
the workflow contract is stable.

## Design constraints

- Python is authoritative for stage fields, defaults, validation, presets,
  command construction, availability, and execution. SwiftUI renders engine
  data and submits typed option objects; it must not rebuild command lines.
- Headless CLIs remain permanent and unchanged. The protocol may retain legacy
  `arguments` during migration, but Studio uses only typed options once a stage
  migrates.
- A setting has one representation: a versioned engine stage specification.
  A scene's selected configuration belongs in `.gausser/manifest.json`; every
  run records the fully resolved specification version, preset, and values.
- Recommended presets must be derived from upstream defaults, documented
  operating constraints, or reproducible benchmark evidence. The UI must not
  present invented values as safe defaults.
- A complete workspace includes its settings, basic/advanced information
  hierarchy, validation, unavailable-state explanation, command preview,
  confirmation, lifecycle/log/cancel presentation, artifact/result summary,
  tests, and documentation. A lone new control is not a deliverable.

## Engine protocol foundation

Add an engine `stage_catalog` request and stage-specific options to `start`:

```json
{"id":"catalog","method":"stage_catalog","params":{"scene":"/absolute/scene"}}
{"id":"run","method":"start","params":{"scene":"/absolute/scene","stage":"colmap","options":{"preset":"video-sequential","matcher":"sequential","existing":"error"}}}
```

The catalog supplies `id`, title, description, prerequisites, warnings,
profiles, and ordered field groups. A field has a stable key, label, help,
type, default, range/enum constraints, unit, and advanced flag. The engine
validates the option object, resolves the selected preset and defaults, emits a
non-secret command preview, then creates `StageRequest` itself. The manifest
records that resolved object with the command result.

## Workspace contract

Every stage view shares a shell with:

1. scene/artifact context and prerequisite status;
2. **Recommended** preset followed by named alternatives;
3. primary configuration fields and an Advanced disclosure;
4. validation, resource/safety warnings, and command preview;
5. explicit confirmation before launch;
6. engine-owned live logs, progress when structurally available, cancellation,
   and durable result links.

## Live execution telemetry and previews

Logs are evidence, not a UI protocol. Each stage adapter must convert its
known upstream output into structured engine events; SwiftUI does not parse
terminal text.

```json
{"event":"progress","run_id":"run-…","phase":"mapping","current":425,"total":527,"unit":"registered images","rate":3.2,"eta_seconds":32,"observed_at":"…"}
{"event":"metric","run_id":"run-…","name":"splat_count","value":1500000,"observed_at":"…"}
{"event":"artifact","run_id":"run-…","kind":"checkpoint","path":"runs/brush/…/export_5000.ply","step":5000}
{"event":"snapshot","run_id":"run-…","kind":"colmap_cameras","camera_count":425,"point_count":12000,"payload":"…"}
```

The engine records monotonic start time, phase start time, last upstream
signal, and optional source-provided rate/ETA. The app shows elapsed time
continuously, clearly labels the age of the latest upstream signal, and shows
an ETA only when a stage reports enough evidence. It must not manufacture a
smoothly advancing ETA from a stale Brush line.

- **COLMAP:** emit feature, matching, mapper registration, and point-count
  progress from the existing stage parser. A background snapshotter may publish
  a downsampled camera-frusta/sparse-point snapshot only after it can safely
  validate a complete binary/text reconstruction revision. Partial model files
  are ignored, never parsed optimistically.
- **Brush and Msplat:** emit observed iteration, loss/evaluation metrics,
  splat count, and rate when upstream actually writes them. The native viewer
  may load only a completed, scene-owned exported checkpoint; it must never
  attach to trainer memory or imply every in-progress Gaussian is renderable.
  Checkpoint publication produces an artifact event and optional viewer refresh.
- **Native presentation:** a run view contains elapsed/phase timers, a progress
  bar where totals exist, metric history, last-update freshness, and a compact
  camera/splat preview. All preview state is explicitly labeled as the last
  validated snapshot/checkpoint.

## Approved delivery slices

### 1. Sources & Sharp Frames

Provide photo copy/link ingest and a video or video-directory source chooser.
Expose upstream Sharp Frames fields: frames per second, target frame count,
minimum buffer, output format, selection method, batch controls, outlier
controls, and output width. The default upstream profile remains explicit;
Gausser profiles are added only after a source/selection benchmark. Video
directories are the supported multi-clip input model and must record source
inventory/provenance.

### 2. COLMAP

Provide a capture-aware selector, then matcher, GPU policy, existing-artifact
policy, mask source, output model location, and vocabulary-tree selection when
applicable. Profiles begin with video-derived sequential matching and a small
photo-set exhaustive mode. Vocab-tree remains unavailable until its required
file is selected. The workspace must warn before expensive exhaustive matching
on large image sets.

### Later slices

- **SAM3:** prompt, score threshold, instance policy, device, checkpoint,
  dilation, feathering, and destructive rerun acknowledgement.
- **Brush:** iterations, checkpoint/export cadence, evaluation/refinement,
  resolution, validation split, and memory-safe maximum-splat profiles.
- **Msplat:** explicitly unmasked acknowledgement, iterations, downscale
  factor/passes, and evaluation policy; the factor-4 local safety profile is
  evidence-backed for the tested 425-camera scene.
- **LOD/publish:** exact input artifact, chunk sizing, extent, retained-level
  recipe, streamed-output validation, and publishing/handoff.

## Acceptance gate for each slice

1. Python option parsing/validation is TDD-covered.
2. The native view is protocol-tested for rendering, defaults, validation,
   confirmation, and lifecycle recovery.
3. A fake-engine integration test proves the exact resolved command/options.
4. A real smoke run uses only the documented preset/tool behavior.
5. The stage specification, manifest snapshot, and user documentation agree.
