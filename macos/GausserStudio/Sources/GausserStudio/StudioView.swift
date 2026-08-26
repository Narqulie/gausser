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
    @State private var samPrompt = ""
    @State private var matcher = "sequential"
    @State private var maxSplats = "1500000"
    @State private var msplatDownscale = "4"

    init(scene: URL, engine: EngineClient) {
        self.scene = scene
        self.engine = engine
        _model = State(initialValue: SceneViewModel(scene: scene, engine: engine))
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                header
                settings
                pipeline
                artifacts
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

    private var settings: some View {
        GroupBox("Run settings") {
            Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 10) {
                GridRow {
                    Text("SAM3 prompt")
                    TextField("e.g. fountain", text: $samPrompt)
                        .textFieldStyle(.roundedBorder)
                }
                GridRow {
                    Text("COLMAP matcher")
                    Picker("Matcher", selection: $matcher) {
                        Text("Sequential").tag("sequential")
                        Text("Exhaustive").tag("exhaustive")
                    }
                    .labelsHidden()
                }
                GridRow {
                    Text("Brush maximum splats")
                    TextField("1500000", text: $maxSplats)
                        .textFieldStyle(.roundedBorder)
                }
                GridRow {
                    Text("Msplat downscale")
                    TextField("4", text: $msplatDownscale)
                        .textFieldStyle(.roundedBorder)
                }
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
            Button("Run \(stage.stage.capitalized)") {
                requestedArguments = []
                requestedStage = stage
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
                            NSWorkspace.shared.activateFileViewerSelecting([scene.appending(path: artifact.relativePath)])
                        }
                        if artifact.format == "ply" || artifact.format == "sog" {
                            Button("Create LOD") {
                                requestedArguments = [artifact.relativePath]
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

    private func activeRunView(_ run: EngineRun) -> some View {
        GroupBox("Running \(run.stage)") {
            HStack {
                ProgressView()
                Text("\(run.stage.capitalized) is \(run.state ?? "running").")
                Spacer()
                Button("Cancel", role: .destructive) { model.cancel() }
            }
            .padding(.top, 4)
        }
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
        requestedStage = nil
        requestedArguments = []
        switch stage.stage {
        case "sammer":
            guard !samPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                model.errorMessage = "Enter a SAM3 prompt before starting masking."
                return
            }
            model.start(stage: stage.stage, arguments: ["--prompt", samPrompt])
        case "colmap":
            model.start(stage: stage.stage, arguments: ["--matcher", matcher])
        case "brush":
            guard let value = Int(maxSplats), value > 0 else {
                model.errorMessage = "Brush maximum splats must be a positive integer."
                return
            }
            model.start(stage: stage.stage, arguments: ["--max-splats", maxSplats])
        case "msplat":
            guard let value = Int(msplatDownscale), value > 0 else {
                model.errorMessage = "Msplat downscale must be a positive integer."
                return
            }
            model.start(stage: stage.stage, arguments: ["--downscale-factor", msplatDownscale])
        case "lod":
            model.start(stage: stage.stage, arguments: arguments)
        default:
            model.errorMessage = "This stage requires a source or artifact selected in its dedicated workspace."
        }
    }
}
