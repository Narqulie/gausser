import SwiftUI
import UniformTypeIdentifiers

struct StudioView: View {
    @State private var session = ProjectSession()
    @State private var recents = RecentSceneStore()
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
                Button("Open Scene", systemImage: "folder.badge.plus") {
                    isChoosingScene = true
                }
            }
        } detail: {
            if let scene = session.scene {
                SceneDashboard(scene: scene)
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
        .alert("Unable to open scene", isPresented: Binding(
            get: { errorMessage != nil },
            set: { if !$0 { errorMessage = nil } }
        )) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(errorMessage ?? "")
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

    private let stages = [
        ("Sources", "photo.stack", "Waiting for engine preflight"),
        ("Sharp Frames", "film", "Optional video ingestion"),
        ("SAM3", "viewfinder", "Masking stage"),
        ("COLMAP", "camera.aperture", "Reconstruction stage"),
        ("Brush / Msplat", "sparkles", "Training stage"),
        ("LOD", "square.3.layers.3d", "Publish stage"),
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(scene.lastPathComponent)
                        .font(.largeTitle.bold())
                    Text(scene.path)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }

                GroupBox("Pipeline") {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 220), spacing: 14)], spacing: 14) {
                        ForEach(stages, id: \.0) { stage in
                            VStack(alignment: .leading, spacing: 12) {
                                Label(stage.0, systemImage: stage.1)
                                    .font(.headline)
                                Text(stage.2)
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                                HStack {
                                    Image(systemName: "circle.dotted")
                                        .foregroundStyle(.secondary)
                                    Text("Not inspected")
                                        .font(.caption)
                                }
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding()
                            .background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
                        }
                    }
                    .padding(.top, 4)
                }

                ContentUnavailableView(
                    "Engine not connected yet",
                    systemImage: "bolt.horizontal.circle",
                    description: Text("The next slice will populate this dashboard from the persistent Python engine.")
                )
                .frame(maxWidth: .infinity)
                .padding(.vertical, 24)
            }
            .padding(28)
        }
        .navigationTitle(scene.lastPathComponent)
    }
}
