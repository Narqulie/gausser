import Foundation
import Testing
@testable import GausserStudio

@Test @MainActor
func engineClientClearsFailedProcessAndCanRelaunch() async throws {
    let client = EngineClient(engineExecutable: "/usr/bin/false")
    var failures = 0

    client.status { result in
        if case .failure = result { failures += 1 }
    }

    for _ in 0..<50 where client.isConnected {
        try await Task.sleep(for: .milliseconds(10))
    }

    #expect(!client.isConnected)
    #expect(failures == 1)

    client.status { result in
        if case .failure = result { failures += 1 }
    }
    for _ in 0..<50 where client.isConnected {
        try await Task.sleep(for: .milliseconds(10))
    }
    #expect(!client.isConnected)
    #expect(failures == 2)
}
