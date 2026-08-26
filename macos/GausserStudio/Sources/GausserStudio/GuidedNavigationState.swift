import Foundation
import Observation

/// A selection request deliberately carries fresh identity. Selecting the
/// current sidebar item is a navigation action, not a no-op: it returns its
/// screen to the step root and clears nested presentation routes.
struct SceneNavigationRequest: Equatable {
    let step: SceneStep?
    private let requestID = UUID()

    init(step: SceneStep?) {
        self.step = step
    }
}

/// Ephemeral presentation state for one scene navigator. It deliberately owns
/// no workflow truth; durable outputs and resolved runs remain engine/manifest
/// owned. It does retain editable drafts while the user moves between the
/// separate settings and run-review screens.
@MainActor
@Observable
final class GuidedNavigationState {
    var selectedTrainer: Trainer?
    var sharpFramesSource: URL?
    var isShowingSharpFramesSettings = false
    var lodSource: String?
    var pendingRun: PendingStageRun?

    private var drafts: [String: [String: JSONValue]] = [:]

    func recordDraft(stage: String, options: [String: JSONValue]) {
        drafts[stage] = options
    }

    func recordRunReview(stage: String, options: [String: JSONValue], resolution: StageResolution) {
        recordDraft(stage: stage, options: options)
        pendingRun = PendingStageRun(stage: stage, options: options, resolution: resolution)
    }

    func seededValues(for stage: String, requiredValues: [String: JSONValue] = [:]) -> [String: JSONValue] {
        var values = drafts[stage] ?? [:]
        // A selected source is not a draft choice; it is the explicit artifact
        // currently being configured and must take precedence.
        values.merge(requiredValues) { _, required in required }
        return values
    }

    func returnedToSettings() {
        pendingRun = nil
    }

    /// Every sidebar selection—also a re-selection of the current item—returns
    /// to the owning step's root screen. Drafts remain available, but no nested
    /// configuration or pending launch can leak into a later visit.
    func moved(to _: SceneStep?) {
        clearNestedRoutes()
    }

    private func clearNestedRoutes() {
        selectedTrainer = nil
        isShowingSharpFramesSettings = false
        sharpFramesSource = nil
        lodSource = nil
        pendingRun = nil
    }

}
