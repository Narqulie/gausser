# Gausser Studio native-app contract

## Purpose

Gausser Studio is a macOS-native SwiftUI project workspace for creating and
running Gausser scenes. It remains secondary to the direct stage CLIs after feature
parity, but never replaces the headless CLIs. The application is a presentation
client; Python remains the authoritative owner of scene state and external
processes.

## V1 boundary

V1 is a locally built, non-sandboxed developer application for this
workstation. It opens or creates scenes, imports a photo directory by an
explicit copy/link policy, and selects video or image-directory input for Sharp
Frames. It requires macOS 15 because the pinned MetalSplatter dependency does.
Bundling/notarizing Python is a later delivery concern, not a reason to
duplicate the engine in Swift.

The first native vertical slice is:

1. choose an existing Gausser scene;
2. inspect its readiness and artifacts;
3. start or cancel one stage through the Python engine;
4. render streamed output and the durable result;
5. select a PLY, SPZ, or `.splat` artifact for a MetalSplatter proof of
   integration.

MetalSplatter is the built-in native viewer. It does not currently consume SOG
or PlayCanvas LOD trees; the engine may later create an explicit, scene-owned
viewer derivative. Its Compose workspace keeps camera position, target, up
vector, and field of view in explicit Swift model state rather than renderer
internals. Users can move and save an orbit pivot, define a non-destructive
presentation origin, add named focus points, and save or restore named camera
views and a home view. The Compose viewer projects the origin as RGB axes and
labels focus points in the same artifact-local coordinate space.

SuperSplat remains an explicit external editor handoff. Reimporting an edited
export creates a fresh `runs/supersplat/<run-id>/` artifact with source and
content-hash provenance; it never replaces the trainer's source export.

## Ownership

| Concern | Owner |
| --- | --- |
| Scene layout, manifests, artifact validation, staging/publication | Python Gausser engine |
| Command construction, preflight, child-process groups, cancellation | Python Gausser engine |
| SwiftUI interaction, camera navigation, and folder bookmarks | Gausser Studio |
| Presentation-profile validation, revisions, and atomic persistence | Python Gausser engine |
| Native splat rendering | MetalSplatter hosted by Gausser Studio |

The Swift app does not invoke stage CLIs directly and must not infer completion
from terminal text. It receives structured engine events instead.

## Engine transport

One app session owns one `gausser-engine` Python child process. Communication
is UTF-8 JSON Lines on stdin/stdout; every line is exactly one request or
response/event. `stderr` is reserved for engine diagnostics and must not carry
protocol data.

Requests:

```json
{"id":"req-1","method":"initialize","params":{"scene":"/absolute/empty-folder"}}
{"id":"req-2","method":"preflight","params":{"scene":"/absolute/scene"}}
{"id":"req-3","method":"artifacts","params":{"scene":"/absolute/scene"}}
{"id":"req-4","method":"reconstruction","params":{"scene":"/absolute/scene"}}
{"id":"req-5","method":"dataset_images","params":{"scene":"/absolute/scene","offset":0,"limit":200}}
{"id":"req-6","method":"ingest_photos","params":{"scene":"/absolute/scene","source":"/absolute/photos","policy":"copy"}}
{"id":"req-7","method":"mutate_dataset","params":{"scene":"/absolute/scene","additions":["/absolute/new.jpg"],"removals":["old.jpg"],"policy":"copy"}}
{"id":"req-8","method":"reimport_supersplat","params":{"scene":"/absolute/scene","source":"runs/brush/run/exports/scene.ply","edited":"/absolute/edited.ply"}}
{"id":"req-9","method":"resolve_stage","params":{"scene":"/absolute/scene","stage":"sharpframes","options":{"source":"/absolute/clips","fps":2}}}
{"id":"req-10","method":"start","params":{"scene":"/absolute/scene","stage":"sharpframes","options":{"source":"/absolute/clips","fps":2}}}
{"id":"req-11","method":"start","params":{"scene":"/absolute/scene","stage":"colmap","options":{"matcher":"sequential","existing":"resume"}}}
{"id":"req-12","method":"cancel","params":{"run_id":"run-123"}}
{"id":"req-13","method":"status","params":{}}
{"id":"req-14","method":"presentation","params":{"scene":"/absolute/scene","artifact":"runs/brush/run/exports/scene.ply"}}
{"id":"req-15","method":"update_presentation","params":{"scene":"/absolute/scene","presentation":{"version":1,"artifact":{"relative_path":"runs/brush/run/exports/scene.ply","sha256":"..."},"revision":null,"frame":{"origin":{"x":0,"y":0,"z":0}},"default_pivot":null,"focus_points":[],"views":[],"home_view_id":null}}}
```

`reconstruction` returns a fingerprint-stable, structurally validated completed `sparse/0` model's camera centres plus deterministic bounded samples of cameras (750 maximum) and coloured `points3D` geometry (4,000 maximum), alongside the full counts. This is view-only preview data—not a generated PLY artifact—and is intentionally not a claim of live COLMAP mapping: COLMAP writes stable geometry and poses when the mapper publishes its model.

`dataset_images` returns a page of authoritative direct `images/` children (`relative_path`, name, byte size, and an optional direct `masks/<stem>.png` SAM-mask path), the complete count, an optional `next_offset`, and a generation `revision`. Later pages send that revision; the engine rejects a changed dataset rather than allowing the client to mix generations. It does not scan source folders or transmit image bytes. The native client validates returned paths as direct children before resolving them through its already-authorized scene URL and produces bounded thumbnails locally.

`mutate_dataset` atomically adds copied or linked photo files and/or removes canonical file names. It validates the replacement dataset first, then archives the prior masks, COLMAP artifacts, and training/LOD runs under `.gausser/invalidated/` because they no longer match the source image set. Manifest recording occurs inside the publication transaction; failure restores every original path.

`presentation` returns the selected artifact's profile or an unsaved default.
Coordinates remain artifact-local. `update_presentation` validates bounded
vectors, unique identifiers, camera geometry, artifact path and SHA-256, then
uses the profile revision as optimistic concurrency control while atomically
replacing `presentations/<artifact-key>.json` under the scene lock. Presentation
metadata is mutable annotation, not a training artifact or manifest stage.

Synchronous responses use `{ "id": ..., "result": ... }` or
`{ "id": ..., "error": { "code": ..., "message": ... } }`.

`start` returns a generated run id immediately. It then emits ordered events:

```json
{"event":"started","run":{"id":"run-123","scene":"/absolute/scene","stage":"colmap"}}
{"event":"log","run_id":"run-123","line":"..."}
{"event":"finished","run":{"id":"run-123","state":"succeeded","result":{...}}}
```

Only one active run is permitted per canonical scene root. An engine shutdown
requests cancellation of every active run and waits for the existing process
runner's process-tree cleanup before exiting. Scene manifests—not app-local
state—remain the durable record following an app restart.

## Native scene access

The app uses `NSOpenPanel` and stores security-scoped bookmarks for recents,
including removable external volumes. V1 is not sandboxed. A sandboxed,
distributable build must establish a separate security-scoped access strategy
for the Python child process; app-process access alone is not evidence that a
child receives the same entitlement.

## Stage model

The engine presents `sharpframes`, `sammer`, `colmap`, `brush`, `msplat`, and
`lod` as first-class stages through one Python-owned typed option catalog.

`stage_catalog` returns versioned field definitions, defaults, presets,
constraints, and help for all six stages. `resolve_stage` validates a typed
`options` object non-mutatively and returns its fully expanded command for
review. Studio renders the catalog, resolves settings, then submits the same
typed options to `start`; the engine validates and resolves them into the
headless command. The resolved specification version, preset, and values are
passed to the transactional stage writer and retained in provenance where the
stage records typed settings. Raw argument start requests are rejected so no
client can bypass typed validation or safety acknowledgements.

## Acceptance constraints

- no duplicate command construction or subprocess ownership in Swift;
- no concurrent active run for one scene;
- every cancellation goes through `gausser.process.run_command`;
- malformed requests cannot terminate the engine;
- stdout remains valid JSON Lines under concurrent log events;
- stage mutations remain transactional and manifest-backed; presentation annotations use their own atomic revision contract;
- app restart can reconstruct completed artifacts from the scene alone;
- V1 packaging is developer-local and requires an installed `gausser-engine`;
  signing, notarization, and an embedded Python runtime are separate release work.
