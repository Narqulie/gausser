import Foundation
import Testing
@testable import GausserStudio

@Test @MainActor
func sceneViewModelSubscribesOnceAfterSwiftUIPersistsItsState() {
    let engine = EngineClient()
    let model = SceneViewModel(scene: URL(fileURLWithPath: "/tmp/scene-a"), engine: engine)

    #expect(engine.eventSubscriberCount == 0)
    model.beginObservingEvents()
    model.beginObservingEvents()
    #expect(engine.eventSubscriberCount == 1)

    model.stopObservingEvents()
    #expect(engine.eventSubscriberCount == 0)
}

@Test @MainActor
func sceneViewModelIgnoresEventsForAnotherScene() {
    let engine = EngineClient()
    let selected = URL(fileURLWithPath: "/tmp/scene-a")
    let model = SceneViewModel(scene: selected, engine: engine)
    let otherRun = EngineRun(id: "run-b", scene: "/tmp/scene-b", stage: "colmap", state: nil)

    model.receive(EngineMessage(id: nil, event: "started", result: nil, error: nil, run: otherRun, runID: nil, line: nil))

    #expect(model.activeRun == nil)
}

@Test @MainActor
func sceneViewModelStoresStructuredTelemetryForItsActiveRun() {
    let engine = EngineClient(engineExecutable: "/usr/bin/false")
    let model = SceneViewModel(scene: URL(fileURLWithPath: "/tmp/scene-a"), engine: engine)
    let run = EngineRun(id: "run-a", scene: "/tmp/scene-a", stage: "brush", state: nil)
    model.receive(EngineMessage(id: nil, event: "started", result: nil, error: nil, run: run, runID: nil, line: nil))

    model.receive(EngineMessage(
        id: nil, event: "metric", result: nil, error: nil, run: nil, runID: "run-a", line: nil,
        elapsedSeconds: 12.5, observedElapsedSeconds: 12.5, phase: "training", step: 1200,
        values: ["splat_count": .number(38291)]
    ))

    #expect(model.telemetry.phase == "training")
    #expect(model.telemetry.step == 1200)
    #expect(model.telemetry.values["splat_count"] == .number(38291))
}

@Test @MainActor
func sceneConfigurationRemainsAvailableUntilAStageStarts() {
    let engine = EngineClient()
    let model = SceneViewModel(scene: URL(fileURLWithPath: "/tmp/scene-a"), engine: engine)

    #expect(model.canConfigureStages)
    let run = EngineRun(id: "run-a", scene: "/tmp/scene-a", stage: "colmap", state: nil)
    model.receive(EngineMessage(id: nil, event: "started", result: nil, error: nil, run: run, runID: nil, line: nil))
    #expect(!model.canConfigureStages)
}

@Test @MainActor
func sceneViewModelAcceptsEventsForItsCanonicalScene() {
    let engine = EngineClient()
    let selected = URL(fileURLWithPath: "/tmp/scene-a")
    let model = SceneViewModel(scene: selected, engine: engine)
    let sameRun = EngineRun(id: "run-a", scene: "/tmp/scene-a", stage: "colmap", state: nil)

    model.receive(EngineMessage(id: nil, event: "started", result: nil, error: nil, run: sameRun, runID: nil, line: nil))

    #expect(model.activeRun?.id == "run-a")
}
