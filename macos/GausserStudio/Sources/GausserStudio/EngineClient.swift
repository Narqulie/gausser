import Foundation
import Observation

struct EngineClientError: LocalizedError {
    let message: String
    let code: String?

    init(message: String, code: String? = nil) {
        self.message = message
        self.code = code
    }

    var errorDescription: String? { message }
}

/// A single long-lived local `gausser-engine` process for one app session.
@MainActor
@Observable
final class EngineClient {
    private let executableOverride: String?
    private var process: Process?
    private var input: FileHandle?
    private var outputBuffer = Data()
    private var pending: [String: (Result<EngineMessage, Error>) -> Void] = [:]

    var isConnected = false
    var events: [EngineMessage] = []
    private var eventSubscribers: [UUID: (EngineMessage) -> Void] = [:]

    /// Internal test seams for lifecycle and duplicate-request regression coverage.
    var eventSubscriberCount: Int { eventSubscribers.count }
    var pendingRequestCount: Int { pending.count }

    init(engineExecutable: String? = nil) {
        self.executableOverride = engineExecutable
    }

    func subscribeToEvents(_ subscriber: @escaping (EngineMessage) -> Void) -> UUID {
        let id = UUID()
        eventSubscribers[id] = subscriber
        return id
    }

    func unsubscribeFromEvents(_ id: UUID) {
        eventSubscribers.removeValue(forKey: id)
    }

    func initialize(scene: URL, completion: @escaping (Result<EngineMessage, Error>) -> Void) {
        send(method: "initialize", params: ["scene": .string(scene.path)], completion: completion)
    }

    func preflight(scene: URL, completion: @escaping (Result<EngineMessage, Error>) -> Void) {
        send(method: "preflight", params: ["scene": .string(scene.path)], completion: completion)
    }

    func stageCatalog(scene: URL, completion: @escaping (Result<EngineMessage, Error>) -> Void) {
        send(method: "stage_catalog", params: ["scene": .string(scene.path)], completion: completion)
    }

    func artifacts(scene: URL, completion: @escaping (Result<EngineMessage, Error>) -> Void) {
        send(method: "artifacts", params: ["scene": .string(scene.path)], completion: completion)
    }

    func reconstruction(scene: URL, completion: @escaping (Result<EngineMessage, Error>) -> Void) {
        send(method: "reconstruction", params: ["scene": .string(scene.path)], completion: completion)
    }

    func datasetImages(
        scene: URL,
        offset: Int,
        revision: String? = nil,
        limit: Int = 200,
        completion: @escaping (Result<EngineMessage, Error>) -> Void
    ) {
        var params: [String: JSONValue] = ["scene": .string(scene.path), "offset": .number(Double(offset)), "limit": .number(Double(limit))]
        if let revision { params["revision"] = .string(revision) }
        send(method: "dataset_images", params: params, completion: completion)
    }

    func status(completion: @escaping (Result<EngineMessage, Error>) -> Void) {
        send(method: "status", params: [:], completion: completion)
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

    func mutateDataset(
        scene: URL,
        additions: [URL],
        removals: [String],
        policy: String,
        revision: String,
        completion: @escaping (Result<EngineMessage, Error>) -> Void
    ) {
        send(
            method: "mutate_dataset",
            params: [
                "scene": .string(scene.path),
                "additions": .array(additions.map { .string($0.path) }),
                "removals": .array(removals.map(JSONValue.string)),
                "policy": .string(policy),
                "revision": .string(revision),
            ],
            completion: completion
        )
    }

    func reimportSuperSplat(
        scene: URL,
        sourceRelativePath: String,
        edited: URL,
        completion: @escaping (Result<EngineMessage, Error>) -> Void
    ) {
        send(
            method: "reimport_supersplat",
            params: [
                "scene": .string(scene.path),
                "source": .string(sourceRelativePath),
                "edited": .string(edited.path),
            ],
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

    func resolveStage(
        scene: URL,
        stage: String,
        options: [String: JSONValue],
        completion: @escaping (Result<EngineMessage, Error>) -> Void
    ) {
        send(
            method: "resolve_stage",
            params: [
                "scene": .string(scene.path),
                "stage": .string(stage),
                "options": .object(options),
            ],
            completion: completion
        )
    }

    func start(
        scene: URL,
        stage: String,
        options: [String: JSONValue],
        completion: @escaping (Result<EngineMessage, Error>) -> Void
    ) {
        send(
            method: "start",
            params: [
                "scene": .string(scene.path),
                "stage": .string(stage),
                "options": .object(options),
            ],
            completion: completion
        )
    }

    func cancel(runID: String, completion: @escaping (Result<EngineMessage, Error>) -> Void) {
        send(method: "cancel", params: ["run_id": .string(runID)], completion: completion)
    }

    func close() {
        // EOF asks the JSON-lines service to run its shutdown path, which cancels
        // and joins the scene-owned child before this process terminates.
        input?.closeFile()
        input = nil
    }

    private func send(
        method: String,
        params: [String: JSONValue],
        completion: @escaping (Result<EngineMessage, Error>) -> Void
    ) {
        var requestID: String?
        do {
            try startIfNeeded()
            let request = EngineRequest(id: UUID().uuidString, method: method, params: params)
            requestID = request.id
            var data = try JSONEncoder().encode(request)
            data.append(0x0A)
            guard let input else {
                throw EngineClientError(message: "The Gausser engine input is unavailable.")
            }
            pending[request.id] = completion
            try input.write(contentsOf: data)
        } catch {
            if let requestID, let waiting = pending.removeValue(forKey: requestID) {
                waiting(.failure(error))
            } else {
                completion(.failure(error))
            }
        }
    }

    private func startIfNeeded() throws {
        guard process == nil else { return }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [engineExecutablePath]
        process.environment = Self.launchEnvironment()

        let inputPipe = Pipe()
        let outputPipe = Pipe()
        process.standardInput = inputPipe
        process.standardOutput = outputPipe
        process.standardError = FileHandle.nullDevice
        process.terminationHandler = { [weak self] terminated in
            Task { @MainActor [weak self] in
                self?.engineDidTerminate(terminated)
            }
        }

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

    static func launchEnvironment(
        base: [String: String] = ProcessInfo.processInfo.environment,
        home: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> [String: String] {
        let standardToolPaths = [
            home.appending(path: ".cargo/bin").path,
            home.appending(path: ".pyenv/shims").path,
            "/opt/homebrew/bin",
            "/usr/local/bin",
        ]
        let existing = base["PATH"]?.split(separator: ":").map(String.init) ?? []
        var seen = Set<String>()
        let path = (standardToolPaths + existing).filter { seen.insert($0).inserted }.joined(separator: ":")
        var environment = base
        environment["PATH"] = path
        return environment
    }

    private var engineExecutablePath: String {
        if let executableOverride, !executableOverride.isEmpty {
            return executableOverride
        }
        if let configured = ProcessInfo.processInfo.environment["GAUSSER_ENGINE"], !configured.isEmpty {
            return configured
        }
        if let bundled = Bundle.main.url(forResource: "gausser-engine", withExtension: nil) {
            return bundled.path
        }
        return "gausser-engine"
    }

    private func engineDidTerminate(_ terminated: Process) {
        guard process === terminated else { return }
        input?.closeFile()
        input = nil
        process = nil
        isConnected = false
        outputBuffer.removeAll(keepingCapacity: false)
        let error = EngineClientError(message: "The Gausser engine stopped unexpectedly.")
        let waiting = pending
        pending.removeAll()
        waiting.values.forEach { $0(.failure(error)) }
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
                    completion(.failure(EngineClientError(message: error.message, code: error.code)))
                } else {
                    completion(.success(message))
                }
            } else {
                events.append(message)
                eventSubscribers.values.forEach { $0(message) }
            }
        }
    }

}
