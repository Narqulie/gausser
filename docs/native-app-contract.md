# Gausser Studio native-app contract

## Purpose

Gausser Studio is a macOS-native SwiftUI project workspace for creating and
running Gausser scenes. It replaces the optional Textual interface after feature
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
viewer derivative. SuperSplat remains an explicit external editor handoff.

## Ownership

| Concern | Owner |
| --- | --- |
| Scene layout, manifests, artifact validation, staging/publication | Python Gausser engine |
| Command construction, preflight, child-process groups, cancellation | Python Gausser engine |
| SwiftUI state, folder bookmarks, presentation, user interaction | Gausser Studio |
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
{"id":"req-4","method":"ingest_photos","params":{"scene":"/absolute/scene","source":"/absolute/photos","policy":"copy"}}
{"id":"req-5","method":"start","params":{"scene":"/absolute/scene","stage":"colmap","arguments":["--matcher","sequential"]}}
{"id":"req-6","method":"cancel","params":{"run_id":"run-123"}}
{"id":"req-7","method":"status","params":{}}
```

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
`lod` as first-class stages. Sharp Frames must join `SceneController` before
the native client is built; otherwise the controller no longer represents all
workflow stages.

Protocol arguments are parsed and validated as JSON arrays of strings. The
next protocol revision should replace generic arguments with stage-specific
option objects; it must retain the ability to run exactly the commands exposed
by the headless CLI.

## Acceptance constraints

- no duplicate command construction or subprocess ownership in Swift;
- no concurrent active run for one scene;
- every cancellation goes through `gausser.process.run_command`;
- malformed requests cannot terminate the engine;
- stdout remains valid JSON Lines under concurrent log events;
- all scene mutations remain transactional and manifest-backed;
- app restart can reconstruct completed artifacts from the scene alone;
- V1 packaging is developer-local and requires an installed `gausser-engine`;
  signing, notarization, and an embedded Python runtime are separate release work.
