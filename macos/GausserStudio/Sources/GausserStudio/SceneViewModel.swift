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

struct ReconstructionCamera: Decodable, Identifiable, Equatable {
    let name: String
    let x: Double
    let y: Double
    let z: Double

    var id: String { name }
}

private struct ReconstructionResponse: Decodable {
    let cameras: [ReconstructionCamera]
}

private struct StatusResponse: Decodable {
    let activeRuns: [EngineRun]

    enum CodingKeys: String, CodingKey {
        case activeRuns = "active_runs"
    }
}

struct StageTelemetryState: Equatable {
    var phase: String?
    var current: Int?
    var total: Int?
    var unit: String?
    var message: String?
    var step: Int?
    var values: [String: JSONValue] = [:]
    var elapsedSeconds: Double = 0
    var observedElapsedSeconds: Double?
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
    private var eventSubscription: UUID?

    private(set) var sceneReadiness: [StageReadiness] = []
    private(set) var toolReadiness: [StageReadiness] = []
    private(set) var artifacts: [ArtifactSummary] = []
    private(set) var reconstructionCameras: [ReconstructionCamera] = []
    private(set) var stageCatalog: [String: StageDefinition] = [:]
    private(set) var activeRun: EngineRun?
    private(set) var activeRunStartedAt: Date?
    private(set) var telemetry = StageTelemetryState()
    private(set) var logs: [String] = []
    private(set) var isRefreshing = false
    private(set) var isIngesting = false
    private(set) var isReimporting = false
    var errorMessage: String?

    init(scene: URL, engine: EngineClient) {
        self.scene = scene
        self.engine = engine
    }

    /// Starts the one engine subscription for this persisted scene state.
    /// SwiftUI may construct disposable view values repeatedly, so subscribing
    /// from `init` would create an invalidation loop while a scene is visible.
    func beginObservingEvents() {
        guard eventSubscription == nil else { return }
        eventSubscription = engine.subscribeToEvents { [weak self] message in
            self?.receive(message)
        }
    }

    func stopObservingEvents() {
        if let eventSubscription {
            engine.unsubscribeFromEvents(eventSubscription)
            self.eventSubscription = nil
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

    func stageReady(for stage: String) -> Bool {
        sceneReadiness.first(where: { $0.stage == stage })?.canRun ?? false
    }

    func refresh() {
        isRefreshing = true
        refreshStageCatalog { _ in }
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
        engine.reconstruction(scene: scene) { [weak self] result in
            guard let self else { return }
            switch result {
            case .success(let message):
                do {
                    self.reconstructionCameras = try self.decodeReconstruction(message).cameras
                } catch {
                    self.errorMessage = error.localizedDescription
                }
            case .failure(let error):
                self.errorMessage = error.localizedDescription
            }
        }
        engine.status { [weak self] result in
            guard let self else { return }
            switch result {
            case .success(let message):
                do {
                    let active = try self.decodeStatus(message).activeRuns.first(where: { self.belongsToScene($0) })
                    if let active {
                        self.activeRun = active
                        self.activeRunStartedAt = Date().addingTimeInterval(-(active.elapsedSeconds ?? 0))
                    } else if self.activeRun?.state != "queued" {
                        self.activeRun = nil
                        self.activeRunStartedAt = nil
                    }
                } catch {
                    self.errorMessage = error.localizedDescription
                }
            case .failure(let error):
                self.errorMessage = error.localizedDescription
            }
        }
    }

    /// Fetches the engine-owned catalog on demand. This protects settings sheets
    /// from an initial refresh racing a newly-created scene's presentation.
    func stageDefinition(named stage: String, completion: @escaping (Result<StageDefinition, Error>) -> Void) {
        if let definition = stageCatalog[stage] {
            completion(.success(definition))
            return
        }
        refreshStageCatalog { [weak self] result in
            switch result {
            case .success(let catalog):
                guard let definition = catalog[stage] else {
                    completion(.failure(EngineClientError(message: "The engine does not define settings for \(stage).")))
                    return
                }
                completion(.success(definition))
            case .failure(let error):
                self?.errorMessage = error.localizedDescription
                completion(.failure(error))
            }
        }
    }

    func ingestPhotos(from source: URL, policy: String) {
        guard !isIngesting, activeRun == nil else {
            errorMessage = "Wait for the active operation to finish first."
            return
        }
        isIngesting = true
        engine.ingestPhotos(scene: scene, source: source, policy: policy) { [weak self] result in
            guard let self else { return }
            self.isIngesting = false
            switch result {
            case .success:
                self.refresh()
            case .failure(let error):
                self.errorMessage = error.localizedDescription
            }
        }
    }

    func reimportSuperSplat(sourceRelativePath: String, edited: URL) {
        guard !isReimporting, activeRun == nil else {
            errorMessage = "Wait for the active operation to finish first."
            return
        }
        isReimporting = true
        engine.reimportSuperSplat(scene: scene, sourceRelativePath: sourceRelativePath, edited: edited) { [weak self] result in
            guard let self else { return }
            self.isReimporting = false
            switch result {
            case .success:
                self.refresh()
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
        telemetry = StageTelemetryState()
        engine.start(scene: scene, stage: stage, arguments: arguments) { [weak self] result in
            guard let self else { return }
            switch result {
            case .success(let message):
                guard let runID = message.result?.objectValue?["run_id"]?.stringValue else {
                    self.errorMessage = "The engine accepted a stage without returning a run identifier."
                    return
                }
                self.activeRun = EngineRun(id: runID, scene: self.scene.path, stage: stage, state: "queued")
                self.activeRunStartedAt = Date()
            case .failure(let error):
                self.errorMessage = error.localizedDescription
            }
        }
    }

    func resolve(stage: String, options: [String: JSONValue], completion: @escaping (Result<StageResolution, Error>) -> Void) {
        engine.resolveStage(scene: scene, stage: stage, options: options) { [weak self] result in
            guard let self else { return }
            switch result {
            case .success(let message):
                do {
                    completion(.success(try self.decodeStageResolution(message)))
                } catch {
                    completion(.failure(error))
                }
            case .failure(let error):
                completion(.failure(error))
            }
        }
    }

    func start(stage: String, options: [String: JSONValue]) {
        guard activeRun == nil else {
            errorMessage = "Wait for the active stage to finish or cancel it first."
            return
        }
        logs.removeAll()
        telemetry = StageTelemetryState()
        engine.start(scene: scene, stage: stage, options: options) { [weak self] result in
            self?.receiveStartResult(stage: stage, result: result)
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

    func receive(_ message: EngineMessage) {
        switch message.event {
        case "started":
            guard let run = message.run, belongsToScene(run) else { return }
            activeRun = run
            activeRunStartedAt = Date().addingTimeInterval(-(message.elapsedSeconds ?? 0))
            telemetry.elapsedSeconds = message.elapsedSeconds ?? 0
        case "log":
            guard message.runID == activeRun?.id, let line = message.line else { return }
            logs.append(line)
            if logs.count > 300 { logs.removeFirst(logs.count - 300) }
        case "progress", "metric":
            guard message.runID == activeRun?.id else { return }
            telemetry.phase = message.phase ?? telemetry.phase
            telemetry.current = message.current ?? telemetry.current
            telemetry.total = message.total ?? telemetry.total
            telemetry.unit = message.unit ?? telemetry.unit
            telemetry.message = message.message ?? telemetry.message
            telemetry.step = message.step ?? telemetry.step
            telemetry.elapsedSeconds = message.elapsedSeconds ?? telemetry.elapsedSeconds
            telemetry.observedElapsedSeconds = message.observedElapsedSeconds ?? telemetry.observedElapsedSeconds
            if let values = message.values { telemetry.values.merge(values) { _, newest in newest } }
        case "finished":
            guard let run = message.run, belongsToScene(run), run.id == activeRun?.id else { return }
            activeRun = nil
            activeRunStartedAt = nil
            refresh()
        default:
            break
        }
    }

    private func receiveStartResult(stage: String, result: Result<EngineMessage, Error>) {
        switch result {
        case .success(let message):
            guard let runID = message.result?.objectValue?["run_id"]?.stringValue else {
                errorMessage = "The engine accepted a stage without returning a run identifier."
                return
            }
            activeRun = EngineRun(id: runID, scene: scene.path, stage: stage, state: "queued")
            activeRunStartedAt = Date()
        case .failure(let error):
            errorMessage = error.localizedDescription
        }
    }

    func displayedElapsedSeconds(now: Date = .now) -> Double {
        guard let activeRunStartedAt else { return telemetry.elapsedSeconds }
        return max(telemetry.elapsedSeconds, now.timeIntervalSince(activeRunStartedAt))
    }

    func freshnessAgeSeconds(now: Date = .now) -> Double? {
        guard let observed = telemetry.observedElapsedSeconds else { return nil }
        return max(0, displayedElapsedSeconds(now: now) - observed)
    }

    private func belongsToScene(_ run: EngineRun) -> Bool {
        URL(fileURLWithPath: run.scene).standardizedFileURL.resolvingSymlinksInPath()
            == scene.standardizedFileURL.resolvingSymlinksInPath()
    }

    private func refreshStageCatalog(completion: @escaping (Result<[String: StageDefinition], Error>) -> Void) {
        engine.stageCatalog(scene: scene) { [weak self] result in
            guard let self else { return }
            switch result {
            case .success(let message):
                do {
                    let catalog = try self.decodeStageCatalog(message).stages
                    self.stageCatalog = catalog
                    completion(.success(catalog))
                } catch {
                    self.errorMessage = error.localizedDescription
                    completion(.failure(error))
                }
            case .failure(let error):
                self.errorMessage = error.localizedDescription
                completion(.failure(error))
            }
        }
    }

    private func decodePreflight(_ message: EngineMessage) throws -> PreflightResponse {
        try decodeResult(message, as: PreflightResponse.self)
    }

    private func decodeStageResolution(_ message: EngineMessage) throws -> StageResolution {
        try decodeResult(message, as: StageResolution.self)
    }

    private func decodeStageCatalog(_ message: EngineMessage) throws -> StageCatalogResponse {
        try decodeResult(message, as: StageCatalogResponse.self)
    }

    private func decodeArtifacts(_ message: EngineMessage) throws -> ArtifactResponse {
        try decodeResult(message, as: ArtifactResponse.self)
    }

    private func decodeReconstruction(_ message: EngineMessage) throws -> ReconstructionResponse {
        try decodeResult(message, as: ReconstructionResponse.self)
    }

    private func decodeStatus(_ message: EngineMessage) throws -> StatusResponse {
        try decodeResult(message, as: StatusResponse.self)
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
