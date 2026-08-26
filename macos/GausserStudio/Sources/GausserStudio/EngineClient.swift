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
    var onEvent: ((EngineMessage) -> Void)?

    func initialize(scene: URL, completion: @escaping (Result<EngineMessage, Error>) -> Void) {
        send(method: "initialize", params: ["scene": .string(scene.path)], completion: completion)
    }

    func preflight(scene: URL, completion: @escaping (Result<EngineMessage, Error>) -> Void) {
        send(method: "preflight", params: ["scene": .string(scene.path)], completion: completion)
    }

    func artifacts(scene: URL, completion: @escaping (Result<EngineMessage, Error>) -> Void) {
        send(method: "artifacts", params: ["scene": .string(scene.path)], completion: completion)
    }

    func ingestPhotos(
        scene: URL,
        source: URL,
        policy: String,
        completion: @escaping (Result<EngineMessage, Error>) -> Void
    ) {
        send(
            method: "ingest_photos",
            params: ["scene": .string(scene.path), "source": .string(source.path), "policy": .string(policy)],
            completion: completion
        )
    }

    func start(
        scene: URL,
        stage: String,
        arguments: [String],
        completion: @escaping (Result<EngineMessage, Error>) -> Void
    ) {
        send(
            method: "start",
            params: [
                "scene": .string(scene.path),
                "stage": .string(stage),
                "arguments": .array(arguments.map(JSONValue.string)),
            ],
            completion: completion
        )
    }

    func cancel(runID: String, completion: @escaping (Result<EngineMessage, Error>) -> Void) {
        send(method: "cancel", params: ["run_id": .string(runID)], completion: completion)
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
        process.arguments = [engineExecutablePath]

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

    private var engineExecutablePath: String {
        if let configured = ProcessInfo.processInfo.environment["GAUSSER_ENGINE"], !configured.isEmpty {
            return configured
        }
        if let bundled = Bundle.main.url(forResource: "gausser-engine", withExtension: nil) {
            return bundled.path
        }
        return "gausser-engine"
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
                onEvent?(message)
            }
        }
    }

}
