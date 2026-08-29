# Gausser Studio completion plan

**Approved:** 2026-08-26

## Brainstorm and design decisions

The Studio must be a project application rather than a dashboard with hidden
command strings. The correct completion unit is a complete *stage workspace*,
not an additional control.

1. **Authoritative settings.** Python owns a versioned typed specification,
preset selection, cross-field validation, availability, warning generation,
and deterministic command construction. SwiftUI receives only catalog data,
uses it to render a workspace, invokes non-mutating resolution for validation
and preview, and starts the exact typed option object. Every completed stage
records the specification version, preset, values, and command in the manifest.

2. **One workspace per lifecycle step.** Sources, COLMAP, SAM3, Brush, Msplat,
and LOD/publish have separate navigation destinations. Each includes source or
artifact context, recommended and alternative presets, primary settings,
advanced settings, validation/warnings, an expanded command preview, explicit
confirmation, active-run state, result summary, and recovery actions.

3. **No invented defaults.** Presets are based only on upstream defaults or
observed project constraints: sequential COLMAP for video-derived imagery;
exhaustive COLMAP for small independent photo sets; macOS CPU COLMAP;
Brush capped below the observed Metal density-growth stall; and explicit
factor-four, unmasked Msplat for the proven 425-camera local-memory case.
Unknown configurations remain clearly labelled custom.

4. **A real event protocol.** Logs remain diagnostic evidence, but the engine
adds structured `progress`, `metric`, `artifact`, and validated `snapshot`
events. The Python adapters parse only known upstream lines and model revisions.
The engine owns elapsed time and freshness timestamps. SwiftUI never fabricates
an ETA or presents an in-memory trainer state as a renderable scene.

5. **Safe previews.** COLMAP camera/point previews are emitted only after a
complete model revision validates. Brush/Msplat previews use only completed,
scene-owned checkpoint artifacts. Preview failure is non-fatal to a training
run and is visibly distinct from the live process.

6. **Compatibility transition.** Existing headless commands remain stable.
Engine `arguments` remain only for unmigrated clients until every Studio stage
uses `options`; no SwiftUI command assembly is introduced.

7. **Release correctness.** Each stage migration carries option tests,
manifest/provenance tests, fake-engine protocol tests, a documented real smoke
run, a native workspace test, and a focused adversarial review. The local
full-pipeline dependency bundle follows only after these workflow contracts
settle.

## Delivery task list

### Foundation

- [ ] Extend the stage-catalog schema with groups, visibility/required rules,
  warnings, output context, and preset metadata; create a reusable SwiftUI
  stage-workspace shell.
- [ ] Upgrade engine events and run state with monotonic elapsed/freshness
  metadata, structured progress/metric/artifact/snapshot events, and client
  models/tests.
- [ ] Add shared result-summary and run-recovery surfaces, retaining logs as
  diagnostics rather than state.

### Complete workspaces

- [x] Sources & Sharp Frames — typed upstream options, transactional manifest
  settings, validation/preview, and source selection.
- [ ] COLMAP — capture-aware profiles, matcher/GPU/mask/artifact/vocabulary
  policies, safe large-exhaustive warning, and reconstruction telemetry.
- [ ] SAM3 — prompt, threshold, instance/device/checkpoint, dilation/feather,
  destructive rerun policy, mask summary, and transactional outputs.
- [ ] Brush — iterations, export/evaluation/refinement/resolution/validation,
  observed Metal-safe splat profiles, checkpoint metrics/artifacts, and stall
  freshness presentation.
- [ ] Msplat — explicit unmasked acknowledgement, iterations, downscale,
  evaluation policy, memory safety profile, checkpoint metrics/artifacts.
- [ ] LOD/publish — exact scene-owned input, level/chunk recipe, validation,
  output summary, and publishing/handoff surface.

### Native visualization and quality gates

- [ ] Render validated COLMAP camera/point snapshots and completed
  checkpoint previews in the native app with explicit snapshot provenance.
- [ ] Migrate dashboard controls into destinations and remove superseded
  settings strings only after the corresponding workspace is complete.
- [ ] Add end-to-end fake-engine and real-smoke coverage, documentation,
  release build checks, adversarial review, and only then resume the full
  local dependency bundle.
