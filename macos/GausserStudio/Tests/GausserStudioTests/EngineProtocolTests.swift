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

@Test func engineMessageDecodesLogEvent() throws {
    let data = Data("""
    {"event":"log","run_id":"run-1","line":"iteration 5000"}
    """.utf8)

    let message = try JSONDecoder().decode(EngineMessage.self, from: data)

    #expect(message.event == "log")
    #expect(message.runID == "run-1")
    #expect(message.line == "iteration 5000")
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
