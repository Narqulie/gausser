import AppKit
import Foundation

/// The file formats SuperSplat handoff can preserve without converting the selected export.
enum SplatExportFormat: String, CaseIterable, Codable, Sendable {
    case ply
    case sog
    case spz
    case splat

    init?(url: URL) {
        self.init(rawValue: url.pathExtension.lowercased())
    }
}

enum SceneExportArtifactError: LocalizedError, Equatable {
    case sceneIsNotDirectory(URL)
    case unsupportedFormat(URL)
    case artifactIsOutsideScene(URL)
    case artifactIsNotRegularFile(URL)

    var errorDescription: String? {
        switch self {
        case .sceneIsNotDirectory(let url):
            return "Scene root is not a directory: \(url.path)"
        case .unsupportedFormat(let url):
            return "Unsupported splat export format: \(url.pathExtension)"
        case .artifactIsOutsideScene(let url):
            return "Export artifact must belong to the selected scene: \(url.path)"
        case .artifactIsNotRegularFile(let url):
            return "Export artifact is not a regular file: \(url.path)"
        }
    }
}

/// A caller-selected, published splat export that is canonically contained by a scene's `runs/` directory.
struct SceneExportArtifact: Equatable, Sendable {
    let sceneURL: URL
    let url: URL
    let relativePath: String
    let format: SplatExportFormat

    init(scene: URL, artifact: URL, fileManager: FileManager = .default) throws {
        let canonicalScene = scene.standardizedFileURL.resolvingSymlinksInPath()
        let canonicalArtifact = artifact.standardizedFileURL.resolvingSymlinksInPath()
        var sceneIsDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: canonicalScene.path, isDirectory: &sceneIsDirectory), sceneIsDirectory.boolValue else {
            throw SceneExportArtifactError.sceneIsNotDirectory(canonicalScene)
        }
        guard let format = SplatExportFormat(url: canonicalArtifact) else {
            throw SceneExportArtifactError.unsupportedFormat(canonicalArtifact)
        }
        guard fileManager.fileExists(atPath: canonicalArtifact.path), Self.isRegularFile(canonicalArtifact) else {
            throw SceneExportArtifactError.artifactIsNotRegularFile(canonicalArtifact)
        }

        let relativePath: String
        do {
            relativePath = try canonicalArtifact.relativePath(from: canonicalScene)
        } catch {
            throw SceneExportArtifactError.artifactIsOutsideScene(canonicalArtifact)
        }
        guard relativePath.split(separator: "/").first == "runs" else {
            throw SceneExportArtifactError.artifactIsOutsideScene(canonicalArtifact)
        }

        sceneURL = canonicalScene
        url = canonicalArtifact
        self.relativePath = relativePath
        self.format = format
    }

    private static func isRegularFile(_ url: URL) -> Bool {
        (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
    }
}

/// The only stable external-editor configuration available locally: an explicitly configured app bundle.
///
/// No SuperSplat application bundle or command-line launch protocol is installed on this workstation.
/// Consequently Gausser Studio does not guess a browser URL or synthesize a shell command. In the
/// absence of a configured app, it reveals the selected export in Finder and can copy its canonical path.
struct SuperSplatEditorConfiguration: Equatable, Sendable {
    let applicationURL: URL?

    static let finderOnly = Self(applicationURL: nil)
}

enum SuperSplatHandoffAction: Equatable, Sendable {
    case openInConfiguredApplication(artifact: URL, application: URL)
    case revealInFinder(URL)
    case copyCanonicalPath(String)
}

enum SuperSplatHandoffError: LocalizedError {
    case copyPathFailed

    var errorDescription: String? {
        "Unable to copy the export path to the clipboard."
    }
}

/// Plans and performs a safe, explicit handoff. It never invokes a shell or changes the training export.
@MainActor
enum SuperSplatHandoff {
    static func primaryAction(
        for artifact: SceneExportArtifact,
        editor: SuperSplatEditorConfiguration = .finderOnly
    ) -> SuperSplatHandoffAction {
        if let applicationURL = editor.applicationURL {
            return .openInConfiguredApplication(artifact: artifact.url, application: applicationURL.standardizedFileURL.resolvingSymlinksInPath())
        }
        return .revealInFinder(artifact.url)
    }

    static func copyPathAction(for artifact: SceneExportArtifact) -> SuperSplatHandoffAction {
        .copyCanonicalPath(artifact.url.path)
    }

    static func perform(_ action: SuperSplatHandoffAction, workspace: NSWorkspace = .shared) async throws {
        switch action {
        case .openInConfiguredApplication(let artifact, let application):
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                workspace.open([artifact], withApplicationAt: application, configuration: .init()) { _, error in
                    if let error {
                        continuation.resume(throwing: error)
                    } else {
                        continuation.resume(returning: ())
                    }
                }
            }
        case .revealInFinder(let artifact):
            workspace.activateFileViewerSelecting([artifact])
        case .copyCanonicalPath(let path):
            let pasteboard = NSPasteboard.general
            pasteboard.clearContents()
            guard pasteboard.setString(path, forType: .string) else {
                throw SuperSplatHandoffError.copyPathFailed
            }
        }
    }
}

private extension URL {
    func relativePath(from ancestor: URL) throws -> String {
        let components = pathComponents
        let ancestorComponents = ancestor.pathComponents
        guard components.starts(with: ancestorComponents) else {
            throw CocoaError(.fileNoSuchFile)
        }
        return components.dropFirst(ancestorComponents.count).joined(separator: "/")
    }
}
