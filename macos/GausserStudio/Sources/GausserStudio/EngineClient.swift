import Foundation
import Observation

struct EngineClientError: LocalizedError {
    let message: String

    var errorDescription: String? { message }
}

/// A single long-lived local `gausser-engine` process for one app session.
@MainActor
@Observable
final class EngineClient {
    private var process: Process?
    private var input: FileHandle?
    private var outputBuffer = Data()
    private var pending: [String: (Result<EngineMessage, Error>) -> Void] = [:]

    var isConnected = false
    var events: [EngineMessage] = []

    func initialize(scene: URL, completion: @escaping (Result<EngineMessage, Error>) -> Void) {
        send(method: "initialize", params: ["scene": .string(scene.path)], completion: completion)
    }

    func close() {
        guard let process else { return }
        process.terminate()
        self.process = nil
        input = nil
        isConnected = false
        let error = EngineClientError(message: "The Gausser engine stopped.")
        let waiting = pending
        pending.removeAll()
        waiting.values.forEach { $0(.failure(error)) }
    }

    private func send(
        method: String,
        params: [String: JSONValue],
        completion: @escaping (Result<EngineMessage, Error>) -> Void
    ) {
        do {
            try startIfNeeded()
            let request = EngineRequest(id: UUID().uuidString, method: method, params: params)
            var data = try JSONEncoder().encode(request)
            data.append(0x0A)
            pending[request.id] = completion
            try input?.write(contentsOf: data)
        } catch {
            completion(.failure(error))
        }
    }

    private func startIfNeeded() throws {
        guard process == nil else { return }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [ProcessInfo.processInfo.environment["GAUSSER_ENGINE"] ?? "gausser-engine"]

        let inputPipe = Pipe()
        let outputPipe = Pipe()
        process.standardInput = inputPipe
        process.standardOutput = outputPipe
        process.standardError = FileHandle.nullDevice

        outputPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            Task { @MainActor [weak self] in
                self?.consume(data)
            }
        }
        try process.run()
        self.process = process
        input = inputPipe.fileHandleForWriting
        isConnected = true
    }

    private func consume(_ data: Data) {
        guard !data.isEmpty else { return }
        outputBuffer.append(data)
        while let newline = outputBuffer.firstIndex(of: 0x0A) {
            let line = outputBuffer.prefix(upTo: newline)
            outputBuffer.removeSubrange(...newline)
            guard let message = try? JSONDecoder().decode(EngineMessage.self, from: line) else { continue }
            if let id = message.id, let completion = pending.removeValue(forKey: id) {
                if let error = message.error {
                    completion(.failure(EngineClientError(message: error.message)))
                } else {
                    completion(.success(message))
                }
            } else {
                events.append(message)
            }
        }
    }

}
