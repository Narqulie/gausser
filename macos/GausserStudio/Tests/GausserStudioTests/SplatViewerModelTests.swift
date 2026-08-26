import Foundation
import simd
import SplatIO
import Testing
@testable import GausserStudio

@Test @MainActor func viewerAcceptsExplicitLocalSupportedArtifact() throws {
    let artifact = try makeTemporaryArtifact(extension: "PLY")
    defer { try? FileManager.default.removeItem(at: artifact.deletingLastPathComponent()) }

    let model = SplatViewerModel()
    model.load(artifact)

    #expect(model.state == .loading(artifact.standardizedFileURL))
}

@Test @MainActor func viewerRejectsUnsupportedSOGArtifactWithoutCreatingLoadRequest() throws {
    let artifact = try makeTemporaryArtifact(extension: "sog")
    defer { try? FileManager.default.removeItem(at: artifact.deletingLastPathComponent()) }

    let model = SplatViewerModel()
    model.load(artifact)

    guard case let .failed(url, message) = model.state else {
        Issue.record("Expected an unsupported SOG artifact to fail validation")
        return
    }
    #expect(url == artifact)
    #expect(message.contains("PLY, SPZ, and .splat"))
}

@Test @MainActor func viewerRejectsRemoteURLs() {
    let model = SplatViewerModel()
    model.load(URL(string: "https://example.com/scene.spz")!)

    #expect(model.state.errorMessage == "The native viewer can load only local files.")
}

@Test func MetalSplatterAutodetectReaderLoadsPLYSPZAndDotSplat() async throws {
    let directory = FileManager.default.temporaryDirectory
        .appending(path: UUID().uuidString, directoryHint: .isDirectory)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }

    let point = SplatPoint(
        position: SIMD3(0, 0, 0),
        color: .sphericalHarmonicFloat([SIMD3(0.5, 0.5, 0.5)]),
        opacity: .linearFloat(0.9),
        scale: .exponent(SIMD3(-3, -3, -3)),
        rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    )
    let urls = [
        try await writePLY(point, to: directory.appending(path: "scene.ply")),
        try await writeSPZ(point, to: directory.appending(path: "scene.spz")),
        try await writeDotSplat(point, to: directory.appending(path: "scene.splat")),
    ]

    for url in urls {
        let reader = try AutodetectSceneReader(url)
        let points = try await reader.readAll()
        #expect(points.count == 1)
    }
}

@Test @MainActor func viewerClearCancelsTheCurrentExplicitSelection() throws {
    let artifact = try makeTemporaryArtifact(extension: "splat")
    defer { try? FileManager.default.removeItem(at: artifact.deletingLastPathComponent()) }

    let model = SplatViewerModel()
    model.load(artifact)
    model.clear()

    #expect(model.state == .idle)
}

private func writePLY(_ point: SplatPoint, to url: URL) async throws -> URL {
    let writer = try SplatPLYSceneWriter(toFileAtPath: url.path)
    try await writer.start(sphericalHarmonicDegree: 0, pointCount: 1)
    try await writer.write([point])
    try await writer.close()
    return url
}

private func writeSPZ(_ point: SplatPoint, to url: URL) async throws -> URL {
    let writer = try SPZSceneWriter(toFileAtPath: url.path)
    try await writer.start(numPoints: 1)
    try await writer.write([point])
    try await writer.close()
    return url
}

private func writeDotSplat(_ point: SplatPoint, to url: URL) async throws -> URL {
    let writer = try DotSplatSceneWriter(toFileAtPath: url.path)
    try await writer.write([point])
    try await writer.close()
    return url
}

private func makeTemporaryArtifact(extension extensionName: String) throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appending(path: UUID().uuidString, directoryHint: .isDirectory)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let artifact = directory.appending(path: "scene.\(extensionName)")
    try Data().write(to: artifact)
    return artifact
}
