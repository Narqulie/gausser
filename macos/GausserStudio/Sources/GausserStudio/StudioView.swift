import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct StudioView: View {
    @State private var session = ProjectSession()
    @State private var recents = RecentSceneStore()
    @State private var engine = EngineClient()
    @State private var isChoosingScene = false
    @State private var isChoosingNewProject = false
    @State private var errorMessage: String?

    var body: some View {
        NavigationSplitView {
            List {
                Section("Projects") {
                    if let scene = session.scene {
                        Label(scene.lastPathComponent, systemImage: "cube.transparent")
                    }
                }
                Section("Recent") {
                    if recents.scenes.isEmpty {
                        Text("No recent scenes")
                            .foregroundStyle(.secondary)
                    }
                    ForEach(recents.scenes) { recent in
                        Button {
                            open(recent)
                        } label: {
                            Label(recent.displayName, systemImage: "folder")
                        }
                        .contextMenu {
                            Button("Remove from Recents", role: .destructive) {
                                recents.remove(recent)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Gausser Studio")
            .toolbar {
                Button("New Project", systemImage: "plus") {
                    isChoosingNewProject = true
                }
                Button("Open Scene", systemImage: "folder.badge.plus") {
                    isChoosingScene = true
                }
            }
        } detail: {
            if let scene = session.scene {
                SceneDashboard(scene: scene, engine: engine)
                    .id(scene.standardizedFileURL.path)
            } else {
                ContentUnavailableView(
                    "Open a Gausser scene",
                    systemImage: "sparkles",
                    description: Text("Choose an existing project folder to inspect its images, reconstruction, and exports.")
                )
            }
        }
        .fileImporter(
            isPresented: $isChoosingScene,
            allowedContentTypes: [.folder],
            allowsMultipleSelection: false
        ) { result in
            if case .success(let urls) = result, let url = urls.first {
                open(url)
            }
        }
        .fileImporter(
            isPresented: $isChoosingNewProject,
            allowedContentTypes: [.folder],
            allowsMultipleSelection: false
        ) { result in
            if case .success(let urls) = result, let url = urls.first {
                initializeProject(at: url)
            }
        }
        .alert("Unable to open scene", isPresented: Binding(
            get: { errorMessage != nil },
            set: { if !$0 { errorMessage = nil } }
        )) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(errorMessage ?? "")
        }
    }

    private func initializeProject(at url: URL) {
        session.select(url)
        engine.initialize(scene: url) { result in
            switch result {
            case .success:
                do {
                    try recents.add(url)
                } catch {
                    errorMessage = error.localizedDescription
                }
            case .failure(let error):
                session.close()
                errorMessage = error.localizedDescription
            }
        }
    }

    private func open(_ url: URL) {
        do {
            try recents.add(url)
            session.select(url)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func open(_ recent: RecentScene) {
        do {
            session.select(try recents.resolve(recent))
        } catch {
            recents.remove(recent)
            errorMessage = "This recent scene is no longer accessible."
        }
    }
}

private struct SceneDashboard: View {
    let scene: URL
    let engine: EngineClient

    @State private var model: SceneViewModel
    @State private var requestedStage: StageReadiness?
    @State private var requestedArguments: [String] = []
    @State private var requestedOptions: [String: JSONValue]?
    @State private var sharpFramesSource: URL?
    @State private var presentsSharpFramesWorkspace = false
    @State private var workspaceStage: String?
    @State private var presentsStageWorkspace = false
    @State private var ingestPolicy = "copy"
    @State private var reimportSource: ArtifactSummary?
    @State private var isChoosingEditedExport = false
    @State private var viewer = SplatViewerModel()

    init(scene: URL, engine: EngineClient) {
        self.scene = scene
        self.engine = engine
        _model = State(initialValue: SceneViewModel(scene: scene, engine: engine))
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                header
                sourceImport
                settings
                pipeline
                artifacts
                viewerSection
                if let activeRun = model.activeRun {
                    activeRunView(activeRun)
                }
                if !model.logs.isEmpty {
                    logs
                }
            }
            .padding(28)
        }
        .navigationTitle(scene.lastPathComponent)
        .task { model.refresh() }
        .onDisappear { model.stopObservingEvents() }
        .sheet(isPresented: $presentsSharpFramesWorkspace) {
            if let source = sharpFramesSource, let definition = model.stageCatalog["sharpframes"] {
                StageWorkspace(
                    definition: definition,
                    seededValues: ["source": .string(source.path)],
                    onResolve: { options, completion in model.resolve(stage: "sharpframes", options: options, completion: completion) }
                ) { options in
                    requestedOptions = options
                    requestedStage = StageReadiness(stage: "sharpframes", state: "ready", detail: source.path)
                }
            } else {
                ContentUnavailableView("Sharp Frames settings unavailable", systemImage: "exclamationmark.triangle")
            }
        }
        .sheet(isPresented: $presentsStageWorkspace) {
            if let stage = workspaceStage, let definition = model.stageCatalog[stage] {
                StageWorkspace(
                    definition: definition,
                    onResolve: { options, completion in model.resolve(stage: stage, options: options, completion: completion) }
                ) { options in
                    requestedOptions = options
                    requestedStage = StageReadiness(stage: stage, state: "ready", detail: "Validated settings")
                }
            } else {
                ContentUnavailableView("Stage settings unavailable", systemImage: "exclamationmark.triangle")
            }
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
        .toolbar {
            Button("Refresh", systemImage: "arrow.clockwise") { model.refresh() }
                .disabled(model.isRefreshing || model.activeRun != nil)
        }
        .confirmationDialog(
            "Run \(requestedStage?.stage ?? "stage")?",
            isPresented: Binding(get: { requestedStage != nil }, set: { if !$0 { requestedStage = nil } })
        ) {
            Button("Run", role: .destructive) { startRequestedStage() }
        } message: {
            Text("Gausser will launch this scene-owned stage. Its logs and cancellation remain managed by the Python engine.")
        }
        .alert("Stage error", isPresented: Binding(get: { model.errorMessage != nil }, set: { if !$0 { model.errorMessage = nil } })) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(model.errorMessage ?? "")
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(scene.lastPathComponent)
                .font(.largeTitle.bold())
            Text(scene.path)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
    }

    private var sourceImport: some View {
        GroupBox("Sources") {
            HStack {
                Text("Import a photo folder, or select one video or a folder of videos for Sharp Frames.")
                    .foregroundStyle(.secondary)
                Spacer()
                Picker("Import policy", selection: $ingestPolicy) {
                    Text("Copy files").tag("copy")
                    Text("Link files").tag("link")
                }
                .labelsHidden()
                Button("Import Photo Folder…", systemImage: "photo.on.rectangle") {
                    guard let source = SourcePicker.choosePhotoDirectory() else { return }
                    model.ingestPhotos(from: source, policy: ingestPolicy)
                }
                .disabled(model.isIngesting || model.activeRun != nil)
                Button("Import Video / Sharp Frames…", systemImage: "film") {
                    guard let source = SourcePicker.chooseSharpFramesSource() else { return }
                    guard model.stageCatalog["sharpframes"] != nil else {
                        model.errorMessage = "Sharp Frames settings are still loading. Refresh and try again."
                        return
                    }
                    sharpFramesSource = source
                    presentsSharpFramesWorkspace = true
                }
                .disabled(model.activeRun != nil || !model.toolAvailable(for: "sharpframes") || !model.stageReady(for: "sharpframes"))
            }
            .padding(.top, 4)
        }
    }

    private var settings: some View {
        GroupBox("Run settings") {
            Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 10) {
            }
            .padding(.top, 4)
        }
    }

    private var pipeline: some View {
        GroupBox("Pipeline") {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 230), spacing: 14)], spacing: 14) {
                ForEach(model.stages) { stage in
                    stageCard(stage)
                }
            }
            .padding(.top, 4)
        }
    }

    private func stageCard(_ stage: StageReadiness) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(stage.stage.capitalized)
                    .font(.headline)
                Spacer()
                Text(stage.state)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(stage.canRun ? .green : .secondary)
            }
            Text(stage.detail)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
            Button(model.stageCatalog[stage.stage] == nil ? "Run \(stage.stage.capitalized)" : "Configure \(stage.stage.capitalized)") {
                if model.stageCatalog[stage.stage] != nil {
                    workspaceStage = stage.stage
                    presentsStageWorkspace = true
                } else {
                    requestedArguments = []
                    requestedOptions = nil
                    requestedStage = stage
                }
            }
            .disabled(!canRequest(stage))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
    }

    private var artifacts: some View {
        GroupBox("Artifacts") {
            if model.artifacts.isEmpty {
                Text("No scene-owned exports yet.")
                    .foregroundStyle(.secondary)
                    .padding(.top, 4)
            } else {
                ForEach(model.artifacts) { artifact in
                    HStack {
                        VStack(alignment: .leading) {
                            Text(artifact.relativePath)
                                .font(.system(.subheadline, design: .monospaced))
                            Text("\(artifact.kind) · \(artifact.format.uppercased()) · \(ByteCountFormatter.string(fromByteCount: Int64(artifact.sizeBytes), countStyle: .file))")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button("Reveal") {
                            NSWorkspace.shared.activateFileViewerSelecting([artifactURL(artifact)])
                        }
                        if supportsNativeViewer(artifact) {
                            Button("View") {
                                viewer.load(artifactURL(artifact))
                            }
                        }
                        if supportsEditorHandoff(artifact) {
                            Button("SuperSplat") {
                                handoffToSuperSplat(artifact)
                            }
                            Button("Reimport Edited") {
                                reimportSource = artifact
                                isChoosingEditedExport = true
                            }
                            .disabled(model.activeRun != nil || model.isReimporting)
                        }
                        if artifact.format == "ply" || artifact.format == "sog" {
                            Button("Create LOD") {
                                requestedArguments = [artifact.relativePath]
                                requestedOptions = nil
                                requestedStage = StageReadiness(stage: "lod", state: "ready", detail: artifact.relativePath)
                            }
                            .disabled(model.activeRun != nil || !model.toolAvailable(for: "lod"))
                        }
                    }
                    .padding(.vertical, 4)
                }
            }
        }
    }

    @ViewBuilder
    private var viewerSection: some View {
        switch viewer.state {
        case .idle:
            EmptyView()
        case .loading(let url):
            GroupBox("Native viewer") {
                HStack { ProgressView(); Text("Loading \(url.lastPathComponent)…") }
                    .padding(.top, 4)
            }
        case .loaded(let url, let splatCount):
            GroupBox("Native viewer · \(url.lastPathComponent) · \(splatCount.formatted()) splats") {
                VStack(alignment: .leading) {
                    SplatViewerHost(model: viewer)
                        .frame(minHeight: 440)
                    Text("Drag to orbit · pinch to zoom")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        case .failed(_, let message):
            GroupBox("Native viewer") {
                Text(message).foregroundStyle(.red)
            }
        }
    }

    private func artifactURL(_ artifact: ArtifactSummary) -> URL {
        scene.appending(path: artifact.relativePath)
    }

    private func supportsNativeViewer(_ artifact: ArtifactSummary) -> Bool {
        ["ply", "spz", "splat"].contains(artifact.format)
    }

    private func supportsEditorHandoff(_ artifact: ArtifactSummary) -> Bool {
        ["ply", "sog", "spz", "splat"].contains(artifact.format)
    }

    private func handoffToSuperSplat(_ summary: ArtifactSummary) {
        do {
            let artifact = try SceneExportArtifact(scene: scene, artifact: artifactURL(summary))
            Task {
                do {
                    try await SuperSplatHandoff.perform(SuperSplatHandoff.primaryAction(for: artifact))
                } catch {
                    model.errorMessage = error.localizedDescription
                }
            }
        } catch {
            model.errorMessage = error.localizedDescription
        }
    }

    private func activeRunView(_ run: EngineRun) -> some View {
        TimelineView(.periodic(from: .now, by: 1)) { context in
            let elapsed = model.displayedElapsedSeconds(now: context.date)
            let freshness = model.freshnessAgeSeconds(now: context.date)
            GroupBox("Running \(run.stage)") {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        ProgressView()
                        Text("\(run.stage.capitalized) is \(run.state ?? "running").")
                        Spacer()
                        Text("Elapsed \(formattedDuration(elapsed))")
                            .monospacedDigit()
                        Button("Cancel", role: .destructive) { model.cancel() }
                    }
                    if let phase = model.telemetry.phase {
                        Text(phase.capitalized)
                            .font(.subheadline.weight(.medium))
                    }
                    if let current = model.telemetry.current, let total = model.telemetry.total, total > 0 {
                        ProgressView(value: Double(current), total: Double(total)) {
                            Text("\(current) / \(total) \(model.telemetry.unit ?? "items")")
                        }
                    }
                    if let step = model.telemetry.step {
                        Text("Step \(step)")
                            .font(.subheadline)
                    }
                    if !model.telemetry.values.isEmpty {
                        HStack(spacing: 12) {
                            ForEach(model.telemetry.values.keys.sorted(), id: \.self) { key in
                                Text("\(key.replacingOccurrences(of: "_", with: " ").capitalized): \(model.telemetry.values[key]!.displayText)")
                            }
                        }
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    }
                    if let freshness {
                        Text("Last upstream update \(formattedDuration(freshness)) ago")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(.top, 4)
            }
        }
    }

    private func formattedDuration(_ seconds: Double) -> String {
        let value = max(0, Int(seconds.rounded(.down)))
        return String(format: "%02d:%02d:%02d", value / 3600, (value / 60) % 60, value % 60)
    }

    private var logs: some View {
        GroupBox("Live logs") {
            Text(model.logs.joined(separator: "\n"))
                .font(.system(.caption, design: .monospaced))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 4)
        }
    }

    private func canRequest(_ stage: StageReadiness) -> Bool {
        guard stage.stage != "images", stage.stage != "sharpframes", stage.stage != "lod" else { return false }
        return stage.canRun && model.toolAvailable(for: stage.stage) && model.activeRun == nil
    }

    private func startRequestedStage() {
        guard let stage = requestedStage else { return }
        let arguments = requestedArguments
        let options = requestedOptions
        requestedStage = nil
        requestedArguments = []
        requestedOptions = nil
        switch stage.stage {
        case "sharpframes":
            guard let options else {
                model.errorMessage = "Choose a Sharp Frames source and settings before launching."
                return
            }
            model.start(stage: stage.stage, options: options)
        case "sammer":
            guard let options else {
                model.errorMessage = "Validate SAM3 settings before launching."
                return
            }
            model.start(stage: stage.stage, options: options)
        case "colmap":
            guard let options else {
                model.errorMessage = "Validate COLMAP settings before launching."
                return
            }
            model.start(stage: stage.stage, options: options)
        case "brush":
            guard let options else {
                model.errorMessage = "Validate Brush settings before launching."
                return
            }
            model.start(stage: stage.stage, options: options)
        case "msplat":
            guard let options else {
                model.errorMessage = "Acknowledge and validate Msplat's unmasked settings before launching."
                return
            }
            model.start(stage: stage.stage, options: options)
        case "lod":
            model.start(stage: stage.stage, arguments: arguments)
        default:
            model.errorMessage = "This stage requires a source or artifact selected in its dedicated workspace."
        }
    }
}
