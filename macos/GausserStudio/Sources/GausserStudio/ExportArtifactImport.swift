import Foundation

enum SuperSplatReimportError: LocalizedError, Equatable {
    case editedArtifactIsNotRegularFile(URL)
    case unsupportedEditedFormat(URL)
    case invalidImportID(String)
    case destinationAlreadyExists(URL)

    var errorDescription: String? {
        switch self {
        case .editedArtifactIsNotRegularFile(let url):
            return "Edited splat artifact is not a regular file: \(url.path)"
        case .unsupportedEditedFormat(let url):
            return "Unsupported edited splat format: \(url.pathExtension)"
        case .invalidImportID(let id):
            return "Invalid SuperSplat import ID: \(id)"
        case .destinationAlreadyExists(let url):
            return "SuperSplat import destination already exists: \(url.path)"
        }
    }
}

/// Durable provenance that an engine-side importer can write beside a newly published edited artifact.
struct SuperSplatArtifactProvenance: Codable, Equatable, Sendable {
    let sourceArtifact: String
    let editedArtifactFilename: String
    let editor: String

    init(sourceArtifact: String, editedArtifactFilename: String) {
        self.sourceArtifact = sourceArtifact
        self.editedArtifactFilename = editedArtifactFilename
        editor = "SuperSplat"
    }
}

/// A non-mutating request for the Python engine to import an edited SuperSplat export.
///
/// The destination is always a fresh `runs/supersplat/<id>/exports/` artifact. This model
/// deliberately does not copy files or update manifests: scene publication remains owned by
/// the engine and must be transactional. It nevertheless makes the no-overwrite invariant and
/// the source provenance explicit at the Swift/Python seam.
struct SuperSplatReimportPlan: Equatable, Sendable {
    let source: SceneExportArtifact
    let editedArtifactURL: URL
    let destinationURL: URL
    let provenance: SuperSplatArtifactProvenance

    init(
        source: SceneExportArtifact,
        editedArtifact: URL,
        importID: String = UUID().uuidString.lowercased(),
        fileManager: FileManager = .default
    ) throws {
        let canonicalEditedArtifact = editedArtifact.standardizedFileURL.resolvingSymlinksInPath()
        guard fileManager.fileExists(atPath: canonicalEditedArtifact.path), Self.isRegularFile(canonicalEditedArtifact) else {
            throw SuperSplatReimportError.editedArtifactIsNotRegularFile(canonicalEditedArtifact)
        }
        guard let format = SplatExportFormat(url: canonicalEditedArtifact) else {
            throw SuperSplatReimportError.unsupportedEditedFormat(canonicalEditedArtifact)
        }
        guard Self.isValidImportID(importID) else {
            throw SuperSplatReimportError.invalidImportID(importID)
        }

        let runDirectory = source.sceneURL
            .appending(path: "runs", directoryHint: .isDirectory)
            .appending(path: "supersplat", directoryHint: .isDirectory)
            .appending(path: importID, directoryHint: .isDirectory)
        guard !fileManager.fileExists(atPath: runDirectory.path) else {
            throw SuperSplatReimportError.destinationAlreadyExists(runDirectory)
        }
        let destination = runDirectory
            .appending(path: "exports", directoryHint: .isDirectory)
            .appending(path: "edited.\(format.rawValue)")

        self.source = source
        editedArtifactURL = canonicalEditedArtifact
        destinationURL = destination
        provenance = SuperSplatArtifactProvenance(
            sourceArtifact: source.relativePath,
            editedArtifactFilename: canonicalEditedArtifact.lastPathComponent
        )
    }

    private static func isValidImportID(_ value: String) -> Bool {
        !value.isEmpty && value.allSatisfy { $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" }
    }

    private static func isRegularFile(_ url: URL) -> Bool {
        (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
    }
}
