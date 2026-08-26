import SwiftUI
import UniformTypeIdentifiers

/// The app-level shell: project access and recents live in the sidebar, while a
/// selected scene exposes its guided, revisitable pipeline in that same sidebar.
struct StudioView: View {
    @State private var session = ProjectSession()
    @State private var recents = RecentSceneStore()
    @State private var engine = EngineClient()
    @State private var isChoosingScene = false
    @State private var errorMessage: String?
    @State private var selectedStep: SceneStep? = .overview

    var body: some View {
        NavigationSplitView {
            List(selection: $selectedStep) {
                Section("Projects") {
                    if let scene = session.scene {
                        Label(scene.lastPathComponent, systemImage: "cube.transparent")
                            .foregroundStyle(.primary)
                    }
                }
                if session.scene != nil {
                    Section("Pipeline") {
                        ForEach(SceneStep.allCases) { step in
                            Label(step.title, systemImage: step.symbol)
                                .tag(Optional(step))
                        }
                    }
                }
                Section("Recent") {
                    if recents.scenes.isEmpty {
                        Text("No recent scenes").foregroundStyle(.secondary)
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
                SceneNavigator(scene: scene, engine: engine, step: $selectedStep)
                    .id(scene.standardizedFileURL.path)
            } else {
                ContentUnavailableView(
                    "Open a Gausser scene",
                    systemImage: "sparkles",
                    description: Text("Choose an existing project folder to move through sources, reconstruction, training, and publishing one step at a time.")
                )
            }
        }
        .fileImporter(
            isPresented: $isChoosingScene,
            allowedContentTypes: [.folder],
            allowsMultipleSelection: false
        ) { result in
            if case .success(let urls) = result, let url = urls.first { open(url) }
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
                selectedStep = .overview
                do { try recents.add(url) }
                catch { errorMessage = error.localizedDescription }
            case .failure(let error):
                errorMessage = error.localizedDescription
            }
        }
    }

    private func open(_ url: URL) {
        do {
            try recents.add(url)
            session.select(url)
            selectedStep = .overview
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func open(_ recent: RecentScene) {
        do {
            session.select(try recents.resolve(recent))
            selectedStep = .overview
        } catch {
            recents.remove(recent)
            errorMessage = "This recent scene is no longer accessible."
        }
    }
}
