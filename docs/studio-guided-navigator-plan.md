# Gausser Studio guided navigator

## Decision

Studio is a guided, revisitable project navigator—not a dashboard that launches configuration sheets. Its sidebar expresses the normal project order, while the detail region presents exactly one workspace at a time. The user selected this model on 2026-08-26.

```
Overview → Sources → SAM3 masks → COLMAP reconstruction → Training → Publish / LOD
                                      └─ Brush or Msplat (one explicit choice)
```

## Interaction contract

- A sidebar step always opens a full-screen workspace. It is never blocked merely because its execution prerequisites are incomplete.
- Each workspace owns its settings, readiness explanation, command preview, live run state, logs, outputs, and a clear next-step action. Engine validation still gates Run.
- Sources contains both photo ingest and the Sharp Frames branch. Selecting video source moves into Sharp Frames settings without a sheet.
- Training is a selection screen. Brush remains the standard masked route; Msplat is labelled unmasked and memory-sensitive. The two choices are mutually exclusive for a given training action.
- Previous steps are revisitable. There is no irreversible wizard lock; the scene manifest remains authoritative.
- Artifact browser and camera/splat preview live in the Overview/Publish surfaces, not as competing dashboard widgets.

## Implementation

1. Add a stable Swift `SceneStep` model and render it in the existing primary sidebar when a scene is active.
2. Replace the dashboard grid and stage sheets with a route-driven `SceneNavigator` detail surface.
3. Reuse `StageWorkspace` as an embedded page: it retains its scrollable settings and fixed action footer, but receives a deterministic back action instead of modal dismissal.
4. Preserve all existing engine requests, manifest/provenance behavior, cancellation, telemetry, source selection, and SuperSplat/LOD actions. This is presentation routing, not a second workflow engine.
5. Cover step ordering, route availability, training branch selection, and the no-prerequisite configuration invariant with Swift tests.
