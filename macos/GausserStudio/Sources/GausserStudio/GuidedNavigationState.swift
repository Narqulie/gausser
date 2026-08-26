import Foundation
import Observation

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

    func recordRunReview(stage: String, options: [String: JSONValue], resolution: StageResolution) {
        drafts[stage] = options
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

    /// Sidebar navigation exits nested pages rather than allowing stale trainer,
    /// Sharp Frames, LOD, or run-review state to reappear in another visit.
    func moved(to step: SceneStep?) {
        guard let step else {
            clearNestedRoutes()
            return
        }
        if step != .training { selectedTrainer = nil }
        if step != .sources {
            isShowingSharpFramesSettings = false
            sharpFramesSource = nil
        }
        if step != .publish { lodSource = nil }
        if let pendingRun, Self.owner(of: pendingRun.stage) != step {
            self.pendingRun = nil
        }
    }

    private func clearNestedRoutes() {
        selectedTrainer = nil
        isShowingSharpFramesSettings = false
        sharpFramesSource = nil
        lodSource = nil
        pendingRun = nil
    }

    private static func owner(of stage: String) -> SceneStep {
        switch stage {
        case "sharpframes": .sources
        case "sammer": .sammer
        case "colmap": .colmap
        case "brush", "msplat": .training
        default: .publish
        }
    }
}
