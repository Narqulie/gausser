import SwiftUI

/// The explicit boundary between editing engine-owned settings and launching a
/// mutation. A StageWorkspace may only arrive here with a resolved snapshot.
struct StageRunReviewScreen: View {
    let definition: StageDefinition?
    let pendingRun: PendingStageRun
    let isStarting: Bool
    let onBackToSettings: () -> Void
    let onRun: () -> Void

    var body: some View {
        StudioScrollPage {
            VStack(alignment: .leading, spacing: 24) {
                StepPageHeader(
                    step: step,
                    title: "Review the run before launch",
                    message: "These settings are an engine-validated immutable snapshot. Starting the stage records this exact configuration in the scene manifest."
                )
                StudioContentCard("Ready to run", systemImage: "checkmark.seal.fill") {
                    LabeledContent("Stage", value: definition?.title ?? studioStageTitle(pendingRun.stage))
                    LabeledContent("Preset", value: pendingRun.resolution.preset)
                    LabeledContent("Specification", value: "v\(pendingRun.resolution.specificationVersion)")
                }
                StudioContentCard("Resolved command", systemImage: "terminal") {
                    Text("The Python engine—not the client—constructed and validated this command.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                    Text(pendingRun.resolution.command.joined(separator: "\n"))
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.quaternary, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                }
                StudioContentCard("Run safety", systemImage: "lock.shield") {
                    Text("The engine holds the scene mutation lock, streams lifecycle/log events, and owns cancellation and cleanup. You can return to settings without launching anything.")
                        .foregroundStyle(.secondary)
                }
            }
        }
        .safeAreaInset(edge: .bottom, spacing: 0) { actionBar }
        .navigationTitle("Run \(definition?.title ?? studioStageTitle(pendingRun.stage))")
    }

    private var actionBar: some View {
        StudioActionBar {
            ViewThatFits(in: .horizontal) {
                HStack {
                    backButton
                    Spacer()
                    runControl
                }
                VStack(alignment: .leading, spacing: 12) {
                    backButton
                    runControl.frame(maxWidth: .infinity, alignment: .trailing)
                }
            }
            .font(.callout)
        }
    }

    private var backButton: some View {
        Button("Back to Settings", action: onBackToSettings)
            .buttonStyle(.bordered)
            .disabled(isStarting)
    }

    @ViewBuilder
    private var runControl: some View {
        if isStarting {
            Label("Starting…", systemImage: "clock")
                .foregroundStyle(.secondary)
        } else {
            StudioGlassActions {
                Button("Run \(definition?.title ?? studioStageTitle(pendingRun.stage))", action: onRun)
                    .studioGlassButton()
            }
        }
    }

    private var step: SceneStep {
        switch pendingRun.stage {
        case "sharpframes": .sources
        case "sammer": .sammer
        case "colmap": .colmap
        case "brush", "msplat": .training
        default: .publish
        }
    }
}
