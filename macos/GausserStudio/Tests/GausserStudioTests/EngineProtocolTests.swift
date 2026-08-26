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

@Test func engineMessageDecodesStructuredRunEvent() throws {
    let data = Data("""
    {"event":"finished","run":{"id":"run-1","scene":"/tmp/scene","stage":"colmap","state":"succeeded"}}
    """.utf8)

    let message = try JSONDecoder().decode(EngineMessage.self, from: data)

    #expect(message.event == "finished")
    #expect(message.run?.id == "run-1")
    #expect(message.run?.state == "succeeded")
}
