import Foundation
import Testing
@testable import GausserStudio

@Test func sceneExportArtifactAcceptsAnExplicitCanonicalRunExport() throws {
    let fixture = try SceneFixture()
    defer { fixture.remove() }
    let export = try fixture.write("runs/brush/run-1/exports/training.PLY")

    let artifact = try SceneExportArtifact(scene: fixture.scene, artifact: export)

    #expect(artifact.format == .ply)
    #expect(artifact.relativePath == "runs/brush/run-1/exports/training.PLY")
    #expect(artifact.url == export.standardizedFileURL.resolvingSymlinksInPath())
}

@Test func sceneExportArtifactRejectsUnsupportedAndNonRunFiles() throws {
    let fixture = try SceneFixture()
    defer { fixture.remove() }
    let unsupported = try fixture.write("runs/brush/run-1/exports/notes.txt")
    let nonRunExport = try fixture.write("images/scene.ply")

    #expect(throws: SceneExportArtifactError.unsupportedFormat(unsupported)) {
        try SceneExportArtifact(scene: fixture.scene, artifact: unsupported)
    }
    #expect(throws: SceneExportArtifactError.artifactIsOutsideScene(nonRunExport)) {
        try SceneExportArtifact(scene: fixture.scene, artifact: nonRunExport)
    }
}

@Test func sceneExportArtifactRejectsASymlinkThatEscapesTheScene() throws {
    let fixture = try SceneFixture()
    defer { fixture.remove() }
    let outside = try fixture.writeOutside("outside.ply")
    let symlink = fixture.scene.appending(path: "runs/brush/run-1/exports/escaped.ply")
    try FileManager.default.createDirectory(at: symlink.deletingLastPathComponent(), withIntermediateDirectories: true)
    try FileManager.default.createSymbolicLink(at: symlink, withDestinationURL: outside)

    #expect(throws: SceneExportArtifactError.artifactIsOutsideScene(outside.standardizedFileURL.resolvingSymlinksInPath())) {
        try SceneExportArtifact(scene: fixture.scene, artifact: symlink)
    }
}

@Test @MainActor func handoffUsesConfiguredApplicationOrSafeFinderAndCopyPathActions() throws {
    let fixture = try SceneFixture()
    defer { fixture.remove() }
    let export = try fixture.write("runs/brush/run-1/exports/training.splat")
    let artifact = try SceneExportArtifact(scene: fixture.scene, artifact: export)
    let configuredApp = URL(fileURLWithPath: "/Applications/External Splat Editor.app")

    #expect(
        SuperSplatHandoff.primaryAction(for: artifact, editor: .init(applicationURL: configuredApp))
            == .openInConfiguredApplication(artifact: artifact.url, application: configuredApp.standardizedFileURL.resolvingSymlinksInPath())
    )
    #expect(SuperSplatHandoff.primaryAction(for: artifact) == .revealInFinder(artifact.url))
    #expect(SuperSplatHandoff.copyPathAction(for: artifact) == .copyCanonicalPath(artifact.url.path))
}

@Test func reimportPlanCreatesAFreshDestinationAndRetainsTrainingProvenance() throws {
    let fixture = try SceneFixture()
    defer { fixture.remove() }
    let trainingExport = try fixture.write("runs/brush/run-1/exports/training.ply")
    let editedExport = try fixture.writeOutside("edited.spz")
    let source = try SceneExportArtifact(scene: fixture.scene, artifact: trainingExport)

    let plan = try SuperSplatReimportPlan(source: source, editedArtifact: editedExport, importID: "edit-001")

    #expect(plan.destinationURL.path.hasSuffix("runs/supersplat/edit-001/exports/edited.spz"))
    #expect(plan.destinationURL != source.url)
    #expect(!FileManager.default.fileExists(atPath: plan.destinationURL.path))
    #expect(plan.provenance.sourceArtifact == "runs/brush/run-1/exports/training.ply")
    #expect(plan.provenance.editedArtifactFilename == "edited.spz")
    #expect(plan.provenance.editor == "SuperSplat")
}

private struct SceneFixture {
    let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString, directoryHint: .isDirectory)

    var scene: URL { root.appending(path: "scene", directoryHint: .isDirectory) }

    init() throws {
        try FileManager.default.createDirectory(at: scene, withIntermediateDirectories: true)
    }

    func write(_ relativePath: String) throws -> URL {
        let url = scene.appending(path: relativePath)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("splat".utf8).write(to: url)
        return url
    }

    func writeOutside(_ name: String) throws -> URL {
        let url = root.appending(path: name)
        try Data("splat".utf8).write(to: url)
        return url
    }

    func remove() {
        try? FileManager.default.removeItem(at: root)
    }
}
