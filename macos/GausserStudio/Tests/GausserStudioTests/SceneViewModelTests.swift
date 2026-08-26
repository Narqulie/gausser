import Foundation
import Testing
@testable import GausserStudio

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
func sceneViewModelAcceptsEventsForItsCanonicalScene() {
    let engine = EngineClient()
    let selected = URL(fileURLWithPath: "/tmp/scene-a")
    let model = SceneViewModel(scene: selected, engine: engine)
    let sameRun = EngineRun(id: "run-a", scene: "/tmp/scene-a", stage: "colmap", state: nil)

    model.receive(EngineMessage(id: nil, event: "started", result: nil, error: nil, run: sameRun, runID: nil, line: nil))

    #expect(model.activeRun?.id == "run-a")
}
