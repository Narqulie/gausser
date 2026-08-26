import Foundation
import Testing
@testable import GausserStudio

@Test
func guidedPipelineUsesTheApprovedRevisitableOrder() {
    #expect(SceneStep.allCases == [.overview, .sources, .sammer, .colmap, .training, .publish])
}

@Test
func guidedPipelineKeepsTrainerChoiceInsideOneTrainingStep() {
    #expect(!SceneStep.allCases.contains { $0.rawValue == "brush" })
    #expect(!SceneStep.allCases.contains { $0.rawValue == "msplat" })
    #expect(SceneStep.training.title == "Training")
}

@Test
func sidebarRequestsAreDistinctEvenForTheSameStep() {
    let first = SceneNavigationRequest(step: .training)
    let second = SceneNavigationRequest(step: .training)

    #expect(first != second)
    #expect(first.step == second.step)
}

@Test @MainActor
func settingsDraftIsPreservedBeforeReviewRun() {
    let state = GuidedNavigationState()
    let options: [String: JSONValue] = ["matcher": .string("sequential")]

    state.recordDraft(stage: "colmap", options: options)

    #expect(state.seededValues(for: "colmap") == options)
    #expect(state.pendingRun == nil)
}

@Test @MainActor
func returningFromRunReviewPreservesTheValidatedSettingsDraft() {
    let state = GuidedNavigationState()
    let options: [String: JSONValue] = ["matcher": .string("sequential")]
    state.recordRunReview(stage: "colmap", options: options, resolution: resolution(stage: "colmap", values: options))

    state.returnedToSettings()

    #expect(state.pendingRun == nil)
    #expect(state.seededValues(for: "colmap") == options)
}

@Test @MainActor
func reselectingTheCurrentSidebarStepReturnsToItsRootScreen() {
    let state = GuidedNavigationState()
    state.selectedTrainer = .brush
    state.recordRunReview(stage: "brush", options: [:], resolution: resolution(stage: "brush", values: [:]))

    state.moved(to: .training)

    #expect(state.selectedTrainer == nil)
    #expect(state.pendingRun == nil)
}

@Test @MainActor
func selectingLODSourceInsidePublishKeepsThePublishRouteOpen() {
    let state = GuidedNavigationState()
    state.moved(to: .publish)

    state.lodSource = "runs/brush/export.ply"

    #expect(state.lodSource == "runs/brush/export.ply")
}

@Test @MainActor
func leavingASidebarStepClearsItsNestedScreenState() {
    let state = GuidedNavigationState()
    state.selectedTrainer = .brush
    state.sharpFramesSource = URL(fileURLWithPath: "/tmp/capture.mov")
    state.isShowingSharpFramesSettings = true
    state.lodSource = "runs/brush/export.ply"
    state.recordRunReview(stage: "brush", options: [:], resolution: resolution(stage: "brush", values: [:]))

    state.moved(to: .overview)

    #expect(state.selectedTrainer == nil)
    #expect(state.sharpFramesSource == nil)
    #expect(!state.isShowingSharpFramesSettings)
    #expect(state.lodSource == nil)
    #expect(state.pendingRun == nil)
}

private func resolution(stage: String, values: [String: JSONValue]) -> StageResolution {
    StageResolution(
        stage: stage,
        specificationVersion: 1,
        preset: "recommended",
        values: values,
        arguments: [],
        command: [stage]
    )
}
