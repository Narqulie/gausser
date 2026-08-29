import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// The stable, revisitable pages in a scene's normal project flow.
enum SceneStep: String, CaseIterable, Hashable, Identifiable {
    case overview
    case sources
    case sammer
    case colmap
    case training
    case compose
    case publish

    var id: String { rawValue }

    var title: String {
        switch self {
        case .overview: "Overview"
        case .sources: "Sources"
        case .sammer: "SAM3 Masks"
        case .colmap: "COLMAP"
        case .training: "Training"
        case .compose: "Compose"
        case .publish: "Publish & LOD"
        }
    }

    var symbol: String {
        switch self {
        case .overview: "rectangle.grid.1x2"
        case .sources: "photo.on.rectangle.angled"
        case .sammer: "wand.and.stars"
        case .colmap: "camera.aperture"
        case .training: "sparkles"
        case .compose: "scope"
        case .publish: "square.and.arrow.up"
        }
    }
}

enum Trainer: String {
    case brush
    case msplat
}

struct PendingStageRun {
    let stage: String
    let options: [String: JSONValue]
    let resolution: StageResolution
}

/// The scene detail surface. It routes presentation only: Python still owns
/// preflight, option validation, launches, cancellation, and scene state.
struct SceneNavigator: View {
    let scene: URL
    let engine: EngineClient
    @Binding var step: SceneStep?
    let navigationRequest: SceneNavigationRequest

    @State private var model: SceneViewModel
    @State private var navigation = GuidedNavigationState()
    @State private var requestedStage: String?
    @State private var requestedOptions: [String: JSONValue]?
    @State private var ingestPolicy = "copy"
    @State private var viewer = SplatViewerModel()
    @State private var composition: SceneCompositionModel
    @State private var reimportSource: ArtifactSummary?
    @State private var isChoosingEditedExport = false

    init(
        scene: URL,
        engine: EngineClient,
        step: Binding<SceneStep?>,
        navigationRequest: SceneNavigationRequest
    ) {
        self.scene = scene
        self.engine = engine
        _step = step
        self.navigationRequest = navigationRequest
        _model = State(initialValue: SceneViewModel(scene: scene, engine: engine))
        _composition = State(initialValue: SceneCompositionModel(scene: scene, engine: engine))
    }

    var body: some View {
        VStack(spacing: 0) {
            if let activeRun = model.activeRun {
                PipelineRunStrip(model: model, run: activeRun)
                Divider()
            } else if let completedRun = model.latestCompletedRun {
                PipelineCompletionStrip(model: model, run: completedRun)
                Divider()
            }
            Group {
                switch step ?? .overview {
                case .overview:
                    overview
                case .sources:
                    sources
                case .sammer:
                    configurationPage(stage: "sammer", onBack: { step = .sources }, next: .colmap)
                case .colmap:
                    configurationPage(stage: "colmap", onBack: { step = .sammer }, next: .training)
                case .training:
                    training
                case .compose:
                    SceneCompositionWorkspace(
                        scene: scene,
                        artifacts: model.artifacts,
                        composition: composition,
                        viewer: viewer,
                        onReveal: { NSWorkspace.shared.activateFileViewerSelecting([$0]) },
                        onNavigate: { step = $0 }
                    )
                case .publish:
                    publish
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .navigationTitle((step ?? .overview).title)
        .task {
            model.beginObservingEvents()
            model.refresh()
        }
        .onDisappear { model.stopObservingEvents() }
        .onChange(of: navigationRequest) { _, request in
            navigation.moved(to: request.step)
        }
        .toolbar {
            Button("Refresh", systemImage: "arrow.clockwise") { model.refresh() }
                .disabled(model.isRefreshing || !model.canConfigureStages)
        }
        .confirmationDialog(
            requestedStageRequiresReplacement
                ? "Replace existing \(studioStageTitle(requestedStage ?? "stage")) output?"
                : "Run \(studioStageTitle(requestedStage ?? "stage"))?",
            isPresented: Binding(get: { requestedStage != nil }, set: { if !$0 { clearRequestedStage() } })
        ) {
            if requestedStageRequiresReplacement {
                Button("Replace and Run", role: .destructive) { startRequestedStage() }
            } else {
                Button("Run") { startRequestedStage() }
            }
            Button("Cancel", role: .cancel) { clearRequestedStage() }
        } message: {
            Text(requestedStageRequiresReplacement
                ? "The reviewed command includes force=true. Python will replace this generation transactionally and invalidate incompatible downstream artifacts."
                : "Gausser will launch this scene-owned stage. Its logs and cancellation remain managed by the Python engine.")
        }
        .fileImporter(
            isPresented: $isChoosingEditedExport,
            allowedContentTypes: [.data],
            allowsMultipleSelection: false
        ) { result in
            guard case .success(let urls) = result, let edited = urls.first, let source = reimportSource else { return }
            reimportSource = nil
            model.reimportSuperSplat(sourceRelativePath: source.relativePath, edited: edited)
        }
        .alert("Stage error", isPresented: Binding(get: { model.errorMessage != nil }, set: { if !$0 { model.errorMessage = nil } })) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(model.errorMessage ?? "")
        }
    }

    private var overview: some View {
        StudioScrollPage(maxWidth: StudioLayout.wideContentWidth) {
            VStack(alignment: .leading, spacing: 24) {
                SceneHero(scene: scene, stages: model.stages)
                StudioContentCard("Project flow", systemImage: "point.3.connected.trianglepath.dotted") {
                    Text("Move through each workspace in order, or revisit a completed step. Settings are never blocked; the engine checks prerequisites when you validate or run.")
                        .foregroundStyle(.secondary)
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 190), spacing: 12)], spacing: 12) {
                        ForEach(SceneStep.allCases.filter { $0 != .overview }) { pipelineStepCard($0) }
                    }
                }
                ReconstructionPreview(
                    cameras: model.reconstructionCameras,
                    points: model.reconstructionPoints,
                    cameraCount: model.reconstructionCameraCount,
                    pointCount: model.reconstructionPointCount,
                    focus: model.reconstructionFocus
                )
                artifactsSummary
                viewerSection
            }
        }
    }

    private func pipelineStepCard(_ item: SceneStep) -> some View {
        let readiness = readiness(for: item)
        return Button {
            navigation.selectedTrainer = nil
            step = item
        } label: {
            VStack(alignment: .leading, spacing: 10) {
                Image(systemName: item.symbol)
                    .font(.title3)
                    .foregroundStyle(.tint)
                Text(item.title)
                    .font(.headline)
                Text(readiness?.detail ?? "Open workspace")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                Spacer(minLength: 0)
                Text(readiness?.state.capitalized ?? "Configure")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(readiness?.canRun == true ? .green : .secondary)
            }
            .frame(maxWidth: .infinity, minHeight: 132, alignment: .leading)
            .padding(14)
            .studioCardSurface(cornerRadius: 16)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Open \(item.title) workspace")
    }

    @ViewBuilder
    private var sources: some View {
        if navigation.isShowingSharpFramesSettings, let source = navigation.sharpFramesSource {
            configurationPage(stage: "sharpframes", seededValues: ["source": .string(source.path)], onBack: {
                navigation.isShowingSharpFramesSettings = false
                navigation.sharpFramesSource = nil
            }, next: .sammer)
        } else {
            StudioScrollPage {
                VStack(alignment: .leading, spacing: 24) {
                    StepPageHeader(
                        step: .sources,
                        title: "Bring your capture into the project",
                        message: "Photo folders are copied or linked into the scene. Video clips pass through Sharp Frames before reconstruction."
                    )
                    StudioContentCard("Photos", systemImage: "photo.stack") {
                        Picker("Import policy", selection: $ingestPolicy) {
                            Text("Copy files into the project").tag("copy")
                            Text("Link files in place").tag("link")
                        }
                        .pickerStyle(.menu)
                        Text("Copy is portable; link preserves the source files in place. Both choices are recorded in the scene manifest.")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                        Button("Choose Photo Folder…", systemImage: "folder.badge.plus") {
                            guard let source = SourcePicker.choosePhotoDirectory() else { return }
                            model.ingestPhotos(from: source, policy: ingestPolicy)
                        }
                        .disabled(model.isIngesting || !model.canConfigureStages)
                        .studioGlassButton()
                    }
                    StudioContentCard("Video / Sharp Frames", systemImage: "film") {
                        Text("Choose multiple clips from one directory, or choose the directory itself. Sharp Frames will build one flat image set transactionally.")
                            .foregroundStyle(.secondary)
                        Button("Choose Video Source…", systemImage: "film.stack") {
                            do {
                                guard let source = try SourcePicker.chooseSharpFramesSource() else { return }
                                navigation.sharpFramesSource = source
                                navigation.isShowingSharpFramesSettings = true
                            } catch {
                                model.errorMessage = error.localizedDescription
                            }
                        }
                        .disabled(!model.canConfigureStages)
                        .studioGlassButton()
                    }
                    DatasetImageBrowser(
                        scene: scene,
                        images: model.datasetImages,
                        revision: model.datasetRevision ?? "unversioned",
                        imageCount: model.datasetImageCount,
                        hasMore: model.nextDatasetImageOffset != nil,
                        isLoading: model.isLoadingDatasetImages,
                        isMutating: !model.canConfigureStages,
                        onLoadMore: model.loadMoreDatasetImages,
                        onAdd: { model.mutateDataset(additions: $0, removals: []) },
                        onRemove: { model.mutateDataset(additions: [], removals: $0) }
                    )
                    StepNavigation(next: .sammer) { step = $0 }
                }
            }
        }
    }

    @ViewBuilder
    private var training: some View {
        if let trainer = navigation.selectedTrainer {
            configurationPage(stage: trainer.rawValue, onBack: { navigation.selectedTrainer = nil }, next: .compose)
        } else {
            StudioScrollPage {
                VStack(alignment: .leading, spacing: 24) {
                    StepPageHeader(
                        step: .training,
                        title: "Choose one training route",
                        message: "Brush is the standard masked trainer. Msplat is an experimental unmasked alternative with an explicit image downscale requirement."
                    )
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 280), spacing: 16)], spacing: 16) {
                        TrainerChoiceCard(
                            title: "Brush",
                            symbol: "paintbrush.pointed.fill",
                            badge: "Recommended",
                            description: "Uses the scene masks and supports an Apple-Silicon-safe splat ceiling.",
                            action: { navigation.selectedTrainer = .brush }
                        )
                        TrainerChoiceCard(
                            title: "Msplat",
                            symbol: "sparkles",
                            badge: "Unmasked",
                            description: "Does not use SAM3 masks. Local runs require an explicit downscale factor to control memory use.",
                            action: { navigation.selectedTrainer = .msplat }
                        )
                    }
                    StepNavigation(previous: .colmap, next: .compose) { step = $0 }
                }
            }
        }
    }

    @ViewBuilder
    private var publish: some View {
        if let lodSource = navigation.lodSource {
            configurationPage(stage: "lod", seededValues: ["source": .string(lodSource)], onBack: {
                self.navigation.lodSource = nil
            })
        } else {
            StudioScrollPage {
                VStack(alignment: .leading, spacing: 24) {
                    StepPageHeader(
                        step: .publish,
                        title: "Inspect, view, and publish outputs",
                        message: "Scene-owned exports are safe to reveal for an external editor, view, reimport, or convert into streamed LOD only when explicitly selected."
                    )
                    artifacts
                    viewerSection
                    StudioContentCard("Streamed LOD", systemImage: "square.3.layers.3d") {
                        Text("Select a PLY or SOG export above and choose Create Streamed LOD from its More menu. The LOD workspace records exact source and converter provenance.")
                            .foregroundStyle(.secondary)
                    }
                    StepNavigation(previous: .compose) { step = $0 }
                }
            }
        }
    }

    @ViewBuilder
    private func configurationPage(
        stage: String,
        seededValues: [String: JSONValue] = [:],
        onBack: @escaping () -> Void = {},
        next: SceneStep? = nil
    ) -> some View {
        if let pendingRun = navigation.pendingRun, pendingRun.stage == stage {
            StageRunReviewScreen(
                definition: model.stageCatalog[stage],
                pendingRun: pendingRun,
                isStarting: model.isStartingStage,
                launchBlockReason: model.stageLaunchBlockReason(
                    stage: stage,
                    options: pendingRun.options
                ),
                onBackToSettings: { navigation.returnedToSettings() },
                onRun: {
                    requestedStage = pendingRun.stage
                    requestedOptions = pendingRun.options
                }
            )
        } else if let definition = model.stageCatalog[stage] {
            let lockedValues: [String: JSONValue] = model.stageRequiredAction(for: stage) == .force
                ? ["force": .bool(true)]
                : [:]
            StageWorkspace(
                definition: definition,
                seededValues: navigation.seededValues(for: stage, requiredValues: seededValues),
                lockedValues: lockedValues,
                onResolve: { options, completion in model.resolve(stage: stage, options: options, completion: completion) },
                onReviewRun: { options, resolution in
                    navigation.recordRunReview(stage: stage, options: options, resolution: resolution)
                },
                onDraftChange: { options in
                    navigation.recordDraft(stage: stage, options: options)
                },
                onClose: onBack,
                nextTitle: next?.title,
                onContinue: next.map { destination in { step = destination } }
            )
        } else {
            ContentUnavailableView(
                "Loading \(studioStageTitle(stage)) settings",
                systemImage: "slider.horizontal.3",
                description: Text("The engine is fetching the authoritative workspace definition."))
                .task(id: stage) {
                    model.stageDefinition(named: stage) { result in
                        if case .failure(let error) = result {
                            model.errorMessage = error.localizedDescription
                        }
                    }
                }
        }
    }

    private var requestedStageRequiresReplacement: Bool {
        guard let requestedStage else { return false }
        return model.stageRequiredAction(for: requestedStage) == .force
    }

    private func clearRequestedStage() {
        requestedStage = nil
        requestedOptions = nil
    }

    private func startRequestedStage() {
        guard let stage = requestedStage, let options = requestedOptions else { return }
        if let reason = model.stageLaunchBlockReason(stage: stage, options: options) {
            clearRequestedStage()
            navigation.returnedToSettings()
            model.errorMessage = "The scene changed after this command was reviewed. \(reason)"
            return
        }
        clearRequestedStage()
        model.start(stage: stage, options: options)
    }

    private func readiness(for item: SceneStep) -> StageReadiness? {
        let stages: [String]
        switch item {
        case .overview: stages = []
        case .sources: stages = ["images", "sharpframes"]
        case .sammer: stages = ["sammer"]
        case .colmap: stages = ["colmap"]
        case .training: stages = ["brush", "msplat"]
        case .compose: stages = []
        case .publish: stages = ["lod"]
        }
        return stages.compactMap { candidate in model.stages.first(where: { $0.stage == candidate }) }.first
    }

    private var artifactsSummary: some View {
        StudioContentCard("Exports", systemImage: "cube.box") {
            if model.artifacts.isEmpty {
                Text("Completed exports appear here after a trainer publishes them.")
                    .foregroundStyle(.secondary)
            } else {
                Text("\(model.artifacts.count) scene-owned artifact\(model.artifacts.count == 1 ? "" : "s") available for composition and publication.")
                    .foregroundStyle(.secondary)
                Button("Open Compose") { step = .compose }
                    .buttonStyle(.bordered)
            }
        }
    }

    private var artifacts: some View {
        StudioContentCard("Scene-owned artifacts", systemImage: "cube.box") {
            if model.artifacts.isEmpty {
                Text("No scene-owned exports yet.")
                    .foregroundStyle(.secondary)
            } else {
                ForEach(model.artifacts) { artifact in
                    VStack(alignment: .leading, spacing: 10) {
                        Text(artifact.relativePath)
                            .font(.system(.subheadline, design: .monospaced))
                            .lineLimit(2)
                            .textSelection(.enabled)
                        Text("\(artifact.kind) · \(artifact.format.uppercased()) · \(ByteCountFormatter.string(fromByteCount: Int64(artifact.sizeBytes), countStyle: .file))")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        HStack {
                            Button("Reveal") {
                                NSWorkspace.shared.activateFileViewerSelecting([artifactURL(artifact)])
                            }
                            if supportsNativeViewer(artifact) {
                                Button("View") { viewer.load(artifactURL(artifact)) }
                            }
                            if supportsEditorHandoff(artifact) {
                                Menu("More") {
                                    Button("Reveal for SuperSplat") { revealForSuperSplat(artifact) }
                                    Button("Reimport Edited Export…") {
                                        reimportSource = artifact
                                        isChoosingEditedExport = true
                                    }
                                    .disabled(!model.canConfigureStages || model.isReimporting)
                                    if artifact.format == "ply" || artifact.format == "sog" {
                                        Divider()
                                        Button("Create Streamed LOD…") {
                                            configurationPageForLOD(artifact)
                                        }
                                        .disabled(!model.canConfigureStages || !model.toolAvailable(for: "lod"))
                                    }
                                }
                            }
                            Spacer()
                        }
                    }
                    .padding(14)
                    .studioCardSurface(cornerRadius: 12)
                }
            }
        }
    }

    // Publish owns LOD configuration. This state is represented by a selected
    // source so the workspace remains full-screen rather than a nested sheet.

    private func configurationPageForLOD(_ artifact: ArtifactSummary) {
        navigation.lodSource = artifact.relativePath
    }

    private func artifactURL(_ artifact: ArtifactSummary) -> URL { scene.appending(path: artifact.relativePath) }
    private func supportsNativeViewer(_ artifact: ArtifactSummary) -> Bool { ["ply", "spz", "splat"].contains(artifact.format) }
    private func supportsEditorHandoff(_ artifact: ArtifactSummary) -> Bool { ["ply", "sog", "spz", "splat"].contains(artifact.format) }

    private func revealForSuperSplat(_ summary: ArtifactSummary) {
        do {
            let artifact = try SceneExportArtifact(scene: scene, artifact: artifactURL(summary))
            SuperSplatHandoff.revealInFinder(artifact)
        } catch { model.errorMessage = error.localizedDescription }
    }

    @ViewBuilder
    private var viewerSection: some View {
        switch viewer.state {
        case .idle: EmptyView()
        case .loading(let url):
            StudioContentCard("Native viewer", systemImage: "eye") {
                HStack { ProgressView(); Text("Loading \(url.lastPathComponent)…") }
            }
        case .loaded(let url, let splatCount):
            StudioContentCard("Native viewer · \(url.lastPathComponent) · \(splatCount.formatted()) splats", systemImage: "eye") {
                SplatViewerHost(model: viewer).frame(minHeight: 440)
                Text("Drag to orbit · pinch to zoom").font(.caption).foregroundStyle(.secondary)
            }
        case .failed(_, let message):
            StudioContentCard("Native viewer", systemImage: "exclamationmark.triangle") {
                Text(message).foregroundStyle(.red)
            }
        }
    }
}

private struct SceneHero: View {
    let scene: URL
    let stages: [StageReadiness]

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top, spacing: 14) {
                Image(systemName: "cube.transparent")
                    .font(.title2)
                    .foregroundStyle(.tint)
                    .frame(width: 52, height: 52)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                VStack(alignment: .leading, spacing: 7) {
                    Text(scene.lastPathComponent).font(.largeTitle.weight(.semibold))
                    Text(scene.path).font(.callout.monospaced()).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle).textSelection(.enabled)
                    Button("Reveal Source Folder", systemImage: "folder") {
                        NSWorkspace.shared.activateFileViewerSelecting([scene.appendingPathComponent("images", isDirectory: true)])
                    }
                    .buttonStyle(.borderless)
                    .accessibilityHint("Reveals this project’s published images folder in Finder")
                }
            }
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(stages.filter { $0.stage != "images" }) { stage in
                        StudioStatusBadge(title: studioStageTitle(stage.stage), state: stage.state)
                    }
                }
                .padding(.vertical, 2)
            }
            .scrollBounceBehavior(.basedOnSize)
        }
    }
}

struct StepPageHeader: View {
    let step: SceneStep
    let title: String
    let message: String

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            Image(systemName: step.symbol)
                .font(.title2)
                .foregroundStyle(.tint)
                .frame(width: 52, height: 52)
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            VStack(alignment: .leading, spacing: 6) {
                Text(step.title.uppercased()).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Text(title).font(.largeTitle.weight(.semibold))
                Text(message).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

private struct StepNavigation: View {
    var previous: SceneStep? = nil
    var next: SceneStep? = nil
    let navigate: (SceneStep) -> Void

    var body: some View {
        HStack {
            if let previous {
                Button("Back to \(previous.title)") { navigate(previous) }
                    .buttonStyle(.bordered)
            }
            Spacer()
            if let next {
                Button("Continue to \(next.title)") { navigate(next) }
                    .studioGlassButton()
            }
        }
        .padding(.top, 4)
    }
}

private struct TrainerChoiceCard: View {
    let title: String
    let symbol: String
    let badge: String
    let description: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Image(systemName: symbol).font(.title2).foregroundStyle(.tint)
                    Spacer()
                    Text(badge).font(.caption.weight(.semibold)).padding(.horizontal, 8).padding(.vertical, 4).background(.quaternary, in: Capsule())
                }
                Text(title).font(.title3.weight(.semibold))
                Text(description).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 0)
                Text("Configure \(title) →").font(.callout.weight(.semibold))
            }
            .frame(maxWidth: .infinity, minHeight: 220, alignment: .leading)
            .padding(StudioLayout.cardInset)
            .studioCardSurface()
        }
        .buttonStyle(.plain)
    }
}

private struct PipelineCompletionStrip: View {
    let model: SceneViewModel
    let run: EngineRun
    @State private var showsLogs = true

    private var state: String { run.state?.capitalized ?? "Finished" }
    private var succeeded: Bool { run.state == "succeeded" }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Image(systemName: succeeded ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                    .foregroundStyle(succeeded ? .green : .orange)
                VStack(alignment: .leading, spacing: 2) {
                    Text("\(studioStageTitle(run.stage)) \(state.lowercased())")
                        .font(.callout.weight(.semibold))
                    Text(succeeded ? "The run output remains available below." : "The run ended before producing its expected output. Review the diagnostic tail below.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            if !model.recentLogs.isEmpty {
                DisclosureGroup("Run log (\(model.logs.count) lines)", isExpanded: $showsLogs) {
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 4) {
                            ForEach(Array(model.recentLogs.enumerated()), id: \.offset) { _, line in
                                Text(line)
                                    .font(.system(.caption, design: .monospaced))
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .textSelection(.enabled)
                            }
                        }
                    }
                    .frame(maxHeight: 180)
                    .padding(.top, 6)
                }
                .font(.callout.weight(.medium))
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(succeeded ? Color.green.opacity(0.08) : Color.orange.opacity(0.1))
    }
}

private struct PipelineRunStrip: View {
    let model: SceneViewModel
    let run: EngineRun
    @State private var showsLogs = true

    var body: some View {
        VStack(spacing: 10) {
            HStack(spacing: 12) {
                ProgressView().controlSize(.small)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Running \(studioStageTitle(run.stage))").font(.callout.weight(.semibold))
                    Text(model.telemetry.phase?.capitalized ?? run.state?.capitalized ?? "Working")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                if let current = model.telemetry.current, let total = model.telemetry.total, total > 0 {
                    ProgressView(value: Double(current), total: Double(total)).frame(width: 140)
                    Text("\(current)/\(total)").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
                Button("Cancel", role: .destructive) { model.cancel() }.buttonStyle(.bordered)
            }
            if !model.recentLogs.isEmpty {
                Divider()
                DisclosureGroup("Live logs (\(model.logs.count))", isExpanded: $showsLogs) {
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 4) {
                            ForEach(Array(model.recentLogs.enumerated()), id: \.offset) { _, line in
                                Text(line)
                                    .font(.system(.caption, design: .monospaced))
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .textSelection(.enabled)
                            }
                        }
                    }
                    .frame(maxHeight: 180)
                    .padding(.top, 6)
                }
                .font(.callout.weight(.medium))
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
        .background(.bar)
    }
}
