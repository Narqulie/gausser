import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct StudioView: View {
    @State private var session = ProjectSession()
    @State private var recents = RecentSceneStore()
    @State private var engine = EngineClient()
    @State private var isChoosingScene = false
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
                    guard let destination = SourcePicker.chooseNewProjectDirectory() else { return }
                    initializeProject(at: destination)
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
        .navigationSplitViewStyle(.balanced)
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
        engine.initialize(scene: url) { result in
            switch result {
            case .success:
                session.select(url)
                do {
                    try recents.add(url)
                } catch {
                    errorMessage = error.localizedDescription
                }
            case .failure(let error):
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
    @State private var requestedOptions: [String: JSONValue]?
    @State private var sharpFramesSource: URL?
    @State private var sharpFramesDefinition: StageDefinition?
    @State private var presentsSharpFramesWorkspace = false
    @State private var workspaceStage: String?
    @State private var workspaceDefinition: StageDefinition?
    @State private var workspaceSeededValues: [String: JSONValue] = [:]
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
            LazyVStack(alignment: .leading, spacing: 24) {
                header
                sourceImport
                pipeline
                ReconstructionPreview(cameras: model.reconstructionCameras)
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
        .task {
            model.beginObservingEvents()
            model.refresh()
        }
        .onDisappear { model.stopObservingEvents() }
        .sheet(isPresented: $presentsSharpFramesWorkspace) {
            if let source = sharpFramesSource, let definition = sharpFramesDefinition {
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
            if let stage = workspaceStage, let definition = workspaceDefinition {
                StageWorkspace(
                    definition: definition,
                    seededValues: workspaceSeededValues,
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
                .disabled(model.isRefreshing || !model.canConfigureStages)
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
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top, spacing: 14) {
                Image(systemName: "cube.transparent")
                    .font(.title2)
                    .foregroundStyle(.tint)
                    .frame(width: 52, height: 52)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                VStack(alignment: .leading, spacing: 5) {
                    Text(scene.lastPathComponent)
                        .font(.largeTitle.weight(.semibold))
                    Text(scene.path)
                        .font(.callout.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .textSelection(.enabled)
                }
            }
            if !model.stages.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    StudioGlassActions {
                        HStack(spacing: 8) {
                            ForEach(model.stages.filter { $0.stage != "images" }) { stage in
                                StudioStatusBadge(title: studioStageTitle(stage.stage), state: stage.state)
                            }
                        }
                        .padding(.vertical, 2)
                    }
                }
                .scrollBounceBehavior(.basedOnSize)
            }
        }
        .padding(.bottom, 4)
    }

    private var sourceImport: some View {
        StudioContentCard("Sources", systemImage: "photo.on.rectangle.angled") {
            Text("Bring in a photo folder, select several clips from one folder, or select the containing video folder for Sharp Frames.")
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 12) {
                Picker("Import policy", selection: $ingestPolicy) {
                    Text("Copy files").tag("copy")
                    Text("Link files").tag("link")
                }
                .pickerStyle(.menu)
                Spacer()
                StudioGlassActions {
                    Button("Import Photos…", systemImage: "photo.on.rectangle") {
                        guard let source = SourcePicker.choosePhotoDirectory() else { return }
                        model.ingestPhotos(from: source, policy: ingestPolicy)
                    }
                    .disabled(model.isIngesting || !model.canConfigureStages)
                    .studioGlassButton()
                    Button("Select Video Source…", systemImage: "film") {
                        do {
                            guard let source = try SourcePicker.chooseSharpFramesSource() else { return }
                            model.stageDefinition(named: "sharpframes") { result in
                                switch result {
                                case .success(let definition):
                                    sharpFramesSource = source
                                    sharpFramesDefinition = definition
                                    presentsSharpFramesWorkspace = true
                                case .failure(let error):
                                    model.errorMessage = error.localizedDescription
                                }
                            }
                        } catch {
                            model.errorMessage = error.localizedDescription
                        }
                    }
                    .disabled(!model.canConfigureStages)
                    .studioGlassButton()
                }
            }
        }
    }

    private var pipeline: some View {
        StudioContentCard("Pipeline", systemImage: "point.3.connected.trianglepath.dotted") {
            Text("Inspect and configure every tool at any time. The engine validates prerequisites only when you validate or run it.")
                .font(.callout)
                .foregroundStyle(.secondary)
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 270), spacing: 14)], spacing: 14) {
                ForEach(model.stages.filter { $0.stage != "images" && $0.stage != "sharpframes" }) { stage in
                    stageCard(stage)
                }
            }
        }
    }

    private func stageCard(_ stage: StageReadiness) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: studioStageSymbol(stage.stage))
                    .foregroundStyle(stage.canRun ? Color.accentColor : Color.secondary)
                    .frame(width: 28, height: 28)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                VStack(alignment: .leading, spacing: 3) {
                    Text(studioStageTitle(stage.stage))
                        .font(.headline)
                    Text(stage.state.capitalized)
                        .font(.caption.weight(.medium))
                        .foregroundStyle(stage.canRun ? .green : .secondary)
                }
                Spacer()
            }
            Text(stage.detail)
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            HStack {
                if !stage.canRun {
                    Label("Prerequisites pending", systemImage: "info.circle")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Configure…") {
                    model.stageDefinition(named: stage.stage) { result in
                        switch result {
                        case .success(let definition):
                            workspaceStage = stage.stage
                            workspaceDefinition = definition
                            workspaceSeededValues = [:]
                            presentsStageWorkspace = true
                        case .failure(let error):
                            model.errorMessage = error.localizedDescription
                        }
                    }
                }
                .disabled(!model.canConfigureStages)
                .buttonStyle(.borderedProminent)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(.background, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    private var artifacts: some View {
        StudioContentCard("Artifacts", systemImage: "cube.box") {
            if model.artifacts.isEmpty {
                Text("No scene-owned exports yet.")
                    .foregroundStyle(.secondary)
                    .padding(.top, 4)
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
                                Button("View") {
                                    viewer.load(artifactURL(artifact))
                                }
                            }
                            if supportsEditorHandoff(artifact) {
                                Menu("More") {
                                    Button("Open in SuperSplat") {
                                        handoffToSuperSplat(artifact)
                                    }
                                    Button("Reimport Edited Export…") {
                                        reimportSource = artifact
                                        isChoosingEditedExport = true
                                    }
                                    .disabled(model.activeRun != nil || model.isReimporting)
                                    if artifact.format == "ply" || artifact.format == "sog" {
                                        Divider()
                                        Button("Create Streamed LOD…") {
                                            model.stageDefinition(named: "lod") { result in
                                                switch result {
                                                case .success(let definition):
                                                    workspaceStage = "lod"
                                                    workspaceDefinition = definition
                                                    workspaceSeededValues = ["source": .string(artifact.relativePath)]
                                                    presentsStageWorkspace = true
                                                case .failure(let error):
                                                    model.errorMessage = error.localizedDescription
                                                }
                                            }
                                        }
                                        .disabled(model.activeRun != nil || !model.toolAvailable(for: "lod"))
                                    }
                                }
                            }
                            Spacer()
                        }
                    }
                    .padding(14)
                    .background(.background, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
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
            StudioContentCard("Running \(studioStageTitle(run.stage))", systemImage: "waveform.path.ecg") {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        ProgressView()
                        Text("\(studioStageTitle(run.stage)) is \(run.state ?? "running").")
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
        StudioContentCard("Live logs", systemImage: "text.alignleft") {
            ScrollView(.vertical) {
                Text(model.logs.joined(separator: "\n"))
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 260)
            .background(.quaternary, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        }
    }

    private func startRequestedStage() {
        guard let stage = requestedStage else { return }
        let options = requestedOptions
        requestedStage = nil
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
            guard let options else {
                model.errorMessage = "Validate streamed LOD settings before launching."
                return
            }
            model.start(stage: stage.stage, options: options)
        default:
            model.errorMessage = "This stage requires a source or artifact selected in its dedicated workspace."
        }
    }
}
