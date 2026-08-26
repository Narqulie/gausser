import Foundation
import Observation

struct ArtifactSummary: Decodable, Identifiable, Equatable {
    let relativePath: String
    let kind: String
    let runID: String
    let format: String
    let sizeBytes: Int

    var id: String { relativePath }

    enum CodingKeys: String, CodingKey {
        case kind, format
        case relativePath = "relative_path"
        case runID = "run_id"
        case sizeBytes = "size_bytes"
    }
}

struct StageReadiness: Decodable, Identifiable, Equatable {
    let stage: String
    let state: String
    let detail: String

    var id: String { stage }
    var canRun: Bool { state == "ready" }
}

private struct ArtifactResponse: Decodable {
    let artifacts: [ArtifactSummary]
}

private struct PreflightResponse: Decodable {
    let scene: String
    let sceneReadiness: [StageReadiness]
    let toolReadiness: [StageReadiness]

    enum CodingKeys: String, CodingKey {
        case scene
        case sceneReadiness = "scene_readiness"
        case toolReadiness = "tool_readiness"
    }
}

/// Presentation state for one scene. Workflow ownership remains in gausser-engine.
@MainActor
@Observable
final class SceneViewModel {
    let scene: URL
    private let engine: EngineClient

    private(set) var sceneReadiness: [StageReadiness] = []
    private(set) var toolReadiness: [StageReadiness] = []
    private(set) var artifacts: [ArtifactSummary] = []
    private(set) var activeRun: EngineRun?
    private(set) var logs: [String] = []
    private(set) var isRefreshing = false
    var errorMessage: String?

    init(scene: URL, engine: EngineClient) {
        self.scene = scene
        self.engine = engine
        engine.onEvent = { [weak self] message in
            self?.receive(message)
        }
    }

    var stages: [StageReadiness] {
        let pipelineOrder = ["images", "sharpframes", "sammer", "colmap", "brush", "msplat", "lod"]
        return sceneReadiness.sorted {
            pipelineOrder.firstIndex(of: $0.stage, default: .max) < pipelineOrder.firstIndex(of: $1.stage, default: .max)
        }
    }

    func toolAvailable(for stage: String) -> Bool {
        toolReadiness.first(where: { $0.stage == stage })?.canRun ?? false
    }

    func refresh() {
        isRefreshing = true
        engine.preflight(scene: scene) { [weak self] result in
            guard let self else { return }
            self.isRefreshing = false
            switch result {
            case .success(let message):
                do {
                    let response = try self.decodePreflight(message)
                    self.sceneReadiness = response.sceneReadiness
                    self.toolReadiness = response.toolReadiness
                } catch {
                    self.errorMessage = error.localizedDescription
                }
            case .failure(let error):
                self.errorMessage = error.localizedDescription
            }
        }
        engine.artifacts(scene: scene) { [weak self] result in
            guard let self else { return }
            switch result {
            case .success(let message):
                do {
                    self.artifacts = try self.decodeArtifacts(message).artifacts
                } catch {
                    self.errorMessage = error.localizedDescription
                }
            case .failure(let error):
                self.errorMessage = error.localizedDescription
            }
        }
    }

    func start(stage: String, arguments: [String] = []) {
        guard activeRun == nil else {
            errorMessage = "Wait for the active stage to finish or cancel it first."
            return
        }
        logs.removeAll()
        engine.start(scene: scene, stage: stage, arguments: arguments) { [weak self] result in
            guard let self else { return }
            switch result {
            case .success(let message):
                guard let runID = message.result?.objectValue?["run_id"]?.stringValue else {
                    self.errorMessage = "The engine accepted a stage without returning a run identifier."
                    return
                }
                self.activeRun = EngineRun(id: runID, scene: self.scene.path, stage: stage, state: "queued")
            case .failure(let error):
                self.errorMessage = error.localizedDescription
            }
        }
    }

    func cancel() {
        guard let activeRun else { return }
        engine.cancel(runID: activeRun.id) { [weak self] result in
            if case .failure(let error) = result {
                self?.errorMessage = error.localizedDescription
            }
        }
    }

    private func receive(_ message: EngineMessage) {
        switch message.event {
        case "started":
            activeRun = message.run
        case "log":
            guard message.runID == activeRun?.id, let line = message.line else { return }
            logs.append(line)
            if logs.count > 300 { logs.removeFirst(logs.count - 300) }
        case "finished":
            guard message.run?.id == activeRun?.id else { return }
            activeRun = nil
            refresh()
        default:
            break
        }
    }

    private func decodePreflight(_ message: EngineMessage) throws -> PreflightResponse {
        try decodeResult(message, as: PreflightResponse.self)
    }

    private func decodeArtifacts(_ message: EngineMessage) throws -> ArtifactResponse {
        try decodeResult(message, as: ArtifactResponse.self)
    }

    private func decodeResult<Result: Decodable>(_ message: EngineMessage, as type: Result.Type) throws -> Result {
        guard let result = message.result else {
            throw EngineClientError(message: "The engine returned no result.")
        }
        return try JSONDecoder().decode(Result.self, from: JSONEncoder().encode(result))
    }
}

private extension Array where Element: Equatable {
    func firstIndex(of element: Element, default fallback: Int) -> Int {
        firstIndex(of: element) ?? fallback
    }
}

private extension JSONValue {
    var objectValue: [String: JSONValue]? {
        guard case .object(let value) = self else { return nil }
        return value
    }

    var stringValue: String? {
        guard case .string(let value) = self else { return nil }
        return value
    }
}
