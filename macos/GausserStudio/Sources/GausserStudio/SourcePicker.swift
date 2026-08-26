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

    /// Select a video or an image folder for the explicit Sharp Frames stage.
    static func chooseSharpFramesSource() -> URL? {
        let panel = NSOpenPanel()
        panel.title = "Choose Video or Image Folder"
        panel.message = "Choose a video to extract sharp frames, or a folder to select from existing images."
        panel.prompt = "Choose Source"
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.movie]
        return panel.runModal() == .OK ? panel.url : nil
    }
}
