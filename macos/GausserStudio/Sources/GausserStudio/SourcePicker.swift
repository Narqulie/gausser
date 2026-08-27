import AppKit
import UniformTypeIdentifiers

@MainActor
enum SourcePicker {
    /// Select a folder whose supported images will be copied or linked into a scene.
    static func choosePhotoDirectory() -> URL? {
        let panel = NSOpenPanel()
        panel.title = "Choose Photo Folder"
        panel.message = "Choose a folder containing the photos for this scene."
        panel.prompt = "Choose Photos"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        return panel.runModal() == .OK ? panel.url : nil
    }

    /// Select one or several photos to copy into the published scene dataset.
    static func choosePhotoFiles() -> [URL] {
        let panel = NSOpenPanel()
        panel.title = "Add Photos to Dataset"
        panel.message = "Choose image files to copy into this scene. Changing the dataset archives derived masks, reconstruction, and training outputs."
        panel.prompt = "Add Photos"
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = [.image]
        return panel.runModal() == .OK ? panel.urls : []
    }

    /// Select an empty destination folder for a new Gausser project.
    static func chooseNewProjectDirectory() -> URL? {
        let panel = NSOpenPanel()
        panel.title = "Create Gausser Project"
        panel.message = "Choose an empty folder for the new project. You may create one here."
        panel.prompt = "Create Project"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        return panel.runModal() == .OK ? panel.url : nil
    }

    /// Select one or several sibling videos, or one source directory, for Sharp Frames.
    ///
    /// Sharp Frames receives one source path. Multiple video selections are therefore
    /// represented by their common parent directory, which preserves upstream's
    /// deterministic directory-input behaviour without copying user media.
    static func chooseSharpFramesSource() throws -> URL? {
        let panel = NSOpenPanel()
        panel.title = "Choose Videos or Source Folder"
        panel.message = "Choose one video, several videos from one folder, or one source folder."
        panel.prompt = "Choose Sources"
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = [.movie]
        guard panel.runModal() == .OK else { return nil }
        return try commonSharpFramesSource(for: panel.urls)
    }

    static func commonSharpFramesSource(for selections: [URL]) throws -> URL {
        guard let first = selections.first else {
            throw SourcePickerError.noSelection
        }
        guard selections.count > 1 else { return first }
        guard selections.allSatisfy({ !$0.hasDirectoryPath }) else {
            throw SourcePickerError.mixedSelections
        }
        let parent = first.deletingLastPathComponent().standardizedFileURL
        guard selections.allSatisfy({ $0.deletingLastPathComponent().standardizedFileURL == parent }) else {
            throw SourcePickerError.separateDirectories
        }
        return parent
    }
}

enum SourcePickerError: LocalizedError, Equatable {
    case noSelection
    case mixedSelections
    case separateDirectories

    var errorDescription: String? {
        switch self {
        case .noSelection:
            "No source was selected."
        case .mixedSelections:
            "Select either one source folder or video files from one folder."
        case .separateDirectories:
            "Selected videos must be in one folder. Choose their containing folder to include clips from multiple locations."
        }
    }
}
