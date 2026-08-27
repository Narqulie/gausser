import Foundation
import Testing
@testable import GausserStudio

@Test func preflightRequestUsesStableJSONLinesEnvelope() throws {
    let request = EngineRequest(
        id: "request-1",
        method: "preflight",
        params: ["scene": .string("/Volumes/T7 Shield/scene")]
    )

    let encoded = try JSONEncoder().encode(request)
    let decoded = try JSONDecoder().decode(EngineRequest.self, from: encoded)

    #expect(decoded == request)
}

@Test func engineMessageDecodesStructuredRequestResult() throws {
    let data = Data("""
    {"id":"request-1","result":{"scene":"/tmp/scene"}}
    """.utf8)

    let message = try JSONDecoder().decode(EngineMessage.self, from: data)

    #expect(message.id == "request-1")
    #expect(message.result == .object(["scene": .string("/tmp/scene")]))
}

@Test func stageCatalogDecodesEngineOwnedSharpFramesDefaults() throws {
    let data = Data("""
    {"stages":{"sharpframes":{"version":1,"title":"Sharp Frames","description":"Select frames","preset":"upstream-defaults","presets":[],"fields":[{"key":"fps","label":"Frames per second","type":"integer","default":10,"minimum":1}]}}}
    """.utf8)

    let catalog = try JSONDecoder().decode(StageCatalogResponse.self, from: data)
    let sharpFrames = try #require(catalog.stages["sharpframes"])
    #expect(sharpFrames.fields[0].defaultValue == .number(10))
    #expect(sharpFrames.fields[0].minimum == 1)
}

@Test @MainActor func recentScenePersistsAndResolvesBookmark() throws {
    let suite = "GausserStudioTests.\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString, directoryHint: .isDirectory)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let store = RecentSceneStore(defaults: defaults)
    try store.add(root)
    let reloaded = RecentSceneStore(defaults: defaults)

    #expect(reloaded.scenes.count == 1)
    #expect(try reloaded.resolve(reloaded.scenes[0]).standardizedFileURL == root.standardizedFileURL)
}

@Test func stageCatalogDecodesEngineWarningsWithoutSyntheticIDs() throws {
    let data = Data("""
    {"stages":{"sharpframes":{"version":1,"title":"Sharp Frames","description":"Select","preset":"upstream-defaults","presets":[],"warnings":[{"level":"warning","message":"An upstream warning."}],"fields":[]}}}
    """.utf8)

    let catalog = try JSONDecoder().decode(StageCatalogResponse.self, from: data)

    #expect(catalog.stages["sharpframes"]?.warnings?.first?.id == "warning:An upstream warning.")
}

@Test func stageCatalogDecodesGroupsAndFieldVisibility() throws {
    let data = Data("""
    {"stages":{"sharpframes":{"version":1,"title":"Sharp Frames","description":"Select","preset":"upstream-defaults","groups":[{"id":"selection","title":"Selection","description":"Frame selection"}],"presets":[{"id":"upstream-defaults","title":"Defaults","description":"Upstream","recommended":true,"values":{}}],"fields":[{"key":"source","label":"Source","type":"path","default":null,"required":true,"group":"selection"},{"key":"vocab_tree","label":"Vocabulary tree","type":"path","default":null,"visible_when":{"key":"matcher","equals":"vocab_tree"}}]}}}
    """.utf8)

    let catalog = try JSONDecoder().decode(StageCatalogResponse.self, from: data)
    let stage = try #require(catalog.stages["sharpframes"])

    #expect(stage.groups?.first?.id == "selection")
    #expect(stage.presets.first?.recommended == true)
    #expect(stage.fields[1].visibleWhen?.key == "matcher")
    #expect(stage.fields[1].visibleWhen?.equals == .string("vocab_tree"))
}

@Test func datasetImageResponseDecodesEngineOwnedRelativePaths() throws {
    let data = Data("""
    {"images":[{"relative_path":"images/frame_0001.jpg","name":"frame_0001.jpg","size_bytes":1234,"mask_relative_path":"masks/frame_0001.png"}],"image_count":425,"next_offset":200,"revision":"dataset-v1"}
    """.utf8)

    let response = try JSONDecoder().decode(DatasetImageResponse.self, from: data)

    #expect(response.images == [DatasetImageSummary(relativePath: "images/frame_0001.jpg", name: "frame_0001.jpg", sizeBytes: 1234, maskRelativePath: "masks/frame_0001.png")])
    #expect(response.imageCount == 425)
    #expect(response.nextOffset == 200)
    #expect(response.revision == "dataset-v1")
}

@Test func reconstructionResponseDecodesSampledSparseModel() throws {
    let data = Data("""
    {"cameras":[{"name":"frame.jpg","x":1,"y":2,"z":3}],"points":[{"x":4,"y":5,"z":6,"red":10,"green":20,"blue":30}],"camera_count":7,"point_count":123}
    """.utf8)

    let reconstruction = try JSONDecoder().decode(ReconstructionResponse.self, from: data)

    #expect(reconstruction.cameras.count == 1)
    #expect(reconstruction.points == [ReconstructionPoint(x: 4, y: 5, z: 6, red: 10, green: 20, blue: 30)])
    #expect(reconstruction.cameraCount == 7)
    #expect(reconstruction.pointCount == 123)
}

@Test func engineMessageDecodesLogEvent() throws {
    let data = Data("""
    {"event":"log","run_id":"run-1","line":"iteration 5000"}
    """.utf8)

    let message = try JSONDecoder().decode(EngineMessage.self, from: data)

    #expect(message.event == "log")
    #expect(message.runID == "run-1")
    #expect(message.line == "iteration 5000")
}

@Test func engineMessageDecodesStructuredTelemetry() throws {
    let data = Data("""
    {"event":"metric","run_id":"run-1","elapsed_seconds":12.5,"observed_elapsed_seconds":12.5,"phase":"training","step":1200,"values":{"splat_count":38291,"psnr":27.8}}
    """.utf8)

    let message = try JSONDecoder().decode(EngineMessage.self, from: data)

    #expect(message.event == "metric")
    #expect(message.step == 1200)
    #expect(message.elapsedSeconds == 12.5)
    #expect(message.values?["splat_count"] == .number(38291))
}

@Test func engineMessageDecodesStructuredRunEvent() throws {
    let data = Data("""
    {"event":"finished","run":{"id":"run-1","scene":"/tmp/scene","stage":"colmap","state":"succeeded"}}
    """.utf8)

    let message = try JSONDecoder().decode(EngineMessage.self, from: data)

    #expect(message.event == "finished")
    #expect(message.run?.id == "run-1")
    #expect(message.run?.state == "succeeded")
}
