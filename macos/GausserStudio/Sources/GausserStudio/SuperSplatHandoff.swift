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

/// SuperSplat has no installed app bundle or stable local launch protocol here.
/// Reveal the canonical scene-owned export without guessing a browser URL,
/// synthesizing a shell command, or changing the trainer-owned file.
@MainActor
enum SuperSplatHandoff {
    static func revealInFinder(_ artifact: SceneExportArtifact, workspace: NSWorkspace = .shared) {
        workspace.activateFileViewerSelecting([artifact.url])
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
