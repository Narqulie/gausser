import Foundation
import Observation
import simd

struct SceneVector3: Codable, Equatable, Sendable {
    var x: Double
    var y: Double
    var z: Double

    init(x: Double, y: Double, z: Double) {
        self.x = x
        self.y = y
        self.z = z
    }

    init(_ value: SIMD3<Float>) {
        self.init(x: Double(value.x), y: Double(value.y), z: Double(value.z))
    }

    var simd: SIMD3<Float> {
        SIMD3(Float(x), Float(y), Float(z))
    }
}

struct PresentationArtifact: Codable, Equatable, Sendable {
    let relativePath: String
    let sha256: String

    enum CodingKeys: String, CodingKey {
        case sha256
        case relativePath = "relative_path"
    }
}

struct PresentationFrame: Codable, Equatable, Sendable {
    var origin: SceneVector3
}

struct PresentationFocusPoint: Codable, Equatable, Identifiable, Sendable {
    let id: String
    var name: String
    var position: SceneVector3
    var framingRadius: Double

    enum CodingKeys: String, CodingKey {
        case id, name, position
        case framingRadius = "framing_radius"
    }
}

struct PresentationView: Codable, Equatable, Identifiable, Sendable {
    let id: String
    var name: String
    var position: SceneVector3
    var target: SceneVector3
    var up: SceneVector3
    var verticalFOVDegrees: Double

    init(
        id: String,
        name: String,
        position: SceneVector3,
        target: SceneVector3,
        up: SceneVector3,
        verticalFOVDegrees: Double
    ) {
        self.id = id
        self.name = name
        self.position = position
        self.target = target
        self.up = up
        self.verticalFOVDegrees = verticalFOVDegrees
    }

    init(id: String, name: String, camera: SplatCameraPose) {
        self.init(
            id: id,
            name: name,
            position: SceneVector3(camera.position),
            target: SceneVector3(camera.target),
            up: SceneVector3(camera.up),
            verticalFOVDegrees: Double(camera.verticalFOVDegrees)
        )
    }

    var cameraPose: SplatCameraPose {
        SplatCameraPose(
            position: position.simd,
            target: target.simd,
            up: up.simd,
            verticalFOVDegrees: Float(verticalFOVDegrees)
        )
    }

    enum CodingKeys: String, CodingKey {
        case id, name, position, target, up
        case verticalFOVDegrees = "vertical_fov_degrees"
    }
}

struct ScenePresentation: Codable, Equatable, Sendable {
    let version: Int
    let artifact: PresentationArtifact
    var revision: String?
    var frame: PresentationFrame
    var defaultPivot: SceneVector3?
    var focusPoints: [PresentationFocusPoint]
    var views: [PresentationView]
    var homeViewID: String?

    enum CodingKeys: String, CodingKey {
        case version, artifact, revision, frame, views
        case defaultPivot = "default_pivot"
        case focusPoints = "focus_points"
        case homeViewID = "home_view_id"
    }
}

struct PresentationResponse: Decodable {
    let presentation: ScenePresentation
}

@MainActor
protocol PresentationEngineClient: AnyObject {
    func presentation(
        scene: URL,
        artifactRelativePath: String,
        completion: @escaping (Result<EngineMessage, Error>) -> Void
    )
    func updatePresentation(
        scene: URL,
        presentation: ScenePresentation,
        completion: @escaping (Result<EngineMessage, Error>) -> Void
    )
}

extension EngineClient: PresentationEngineClient {}

/// Editable presentation state for one explicitly selected immutable artifact.
/// The engine remains the filesystem and revision authority; camera navigation
/// itself stays local until the user invokes one of these semantic save actions.
@MainActor
@Observable
final class SceneCompositionModel {
    let scene: URL
    private let engine: any PresentationEngineClient
    private var selectionGeneration = 0

    private(set) var selectedArtifact: ArtifactSummary?
    private(set) var presentation: ScenePresentation?
    private(set) var isLoading = false
    private(set) var isSaving = false
    var errorMessage: String?

    init(scene: URL, engine: any PresentationEngineClient) {
        self.scene = scene
        self.engine = engine
    }

    var canEdit: Bool {
        presentation != nil && !isLoading && !isSaving
    }

    func select(_ artifact: ArtifactSummary, viewer: SplatViewerModel) {
        selectionGeneration += 1
        let generation = selectionGeneration
        selectedArtifact = artifact
        presentation = nil
        errorMessage = nil
        isLoading = true
        isSaving = false
        viewer.load(scene.appending(path: artifact.relativePath))
        engine.presentation(scene: scene, artifactRelativePath: artifact.relativePath) { [weak self, weak viewer] result in
            guard let self, generation == self.selectionGeneration else { return }
            self.isLoading = false
            switch result {
            case .success(let message):
                do {
                    let loaded = try Self.decode(message)
                    self.presentation = loaded
                    if let viewer { self.apply(loaded, to: viewer) }
                } catch {
                    self.errorMessage = error.localizedDescription
                }
            case .failure(let error):
                self.errorMessage = error.localizedDescription
            }
        }
    }

    func reload(viewer: SplatViewerModel) {
        guard let selectedArtifact else { return }
        select(selectedArtifact, viewer: viewer)
    }

    func clear(viewer: SplatViewerModel) {
        selectionGeneration += 1
        selectedArtifact = nil
        presentation = nil
        isLoading = false
        isSaving = false
        errorMessage = nil
        viewer.clear()
    }

    func setOriginFromPivot(viewer: SplatViewerModel) {
        mutate(viewer: viewer) { profile in
            profile.frame.origin = SceneVector3(viewer.camera.target)
        }
    }

    func resetOrigin(viewer: SplatViewerModel) {
        mutate(viewer: viewer) { profile in
            profile.frame.origin = SceneVector3(x: 0, y: 0, z: 0)
        }
    }

    func saveDefaultPivot(viewer: SplatViewerModel) {
        mutate(viewer: viewer) { profile in
            profile.defaultPivot = SceneVector3(viewer.camera.target)
        }
    }

    func restoreDefaultPivot(viewer: SplatViewerModel) {
        guard let pivot = presentation?.defaultPivot else { return }
        viewer.restorePivot(pivot.simd)
    }

    func addFocusPoint(named rawName: String, viewer: SplatViewerModel) {
        let name = rawName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else {
            errorMessage = "Focus point name cannot be empty."
            return
        }
        mutate(viewer: viewer) { profile in
            let distance = max(simd_distance(viewer.camera.position, viewer.camera.target), 0.001)
            profile.focusPoints.append(PresentationFocusPoint(
                id: UUID().uuidString.lowercased(),
                name: name,
                position: SceneVector3(viewer.camera.target),
                framingRadius: Double(distance * 0.08)
            ))
        }
    }

    func restoreFocusPoint(_ point: PresentationFocusPoint, viewer: SplatViewerModel) {
        viewer.restorePivot(point.position.simd)
    }

    func deleteFocusPoint(id: String, viewer: SplatViewerModel) {
        mutate(viewer: viewer) { profile in
            profile.focusPoints.removeAll { $0.id == id }
        }
    }

    func saveView(named rawName: String, viewer: SplatViewerModel) {
        let name = rawName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else {
            errorMessage = "Saved view name cannot be empty."
            return
        }
        mutate(viewer: viewer) { profile in
            profile.views.append(PresentationView(
                id: UUID().uuidString.lowercased(),
                name: name,
                camera: viewer.camera
            ))
        }
    }

    func restoreView(_ view: PresentationView, viewer: SplatViewerModel) {
        viewer.restoreCamera(view.cameraPose)
    }

    func setHomeView(id: String, viewer: SplatViewerModel) {
        mutate(viewer: viewer) { profile in
            profile.homeViewID = id
        }
    }

    func deleteView(id: String, viewer: SplatViewerModel) {
        mutate(viewer: viewer) { profile in
            profile.views.removeAll { $0.id == id }
            if profile.homeViewID == id { profile.homeViewID = nil }
        }
    }

    private func mutate(
        viewer: SplatViewerModel,
        change: (inout ScenePresentation) -> Void
    ) {
        guard !isSaving, var changed = presentation, let selectedArtifact else { return }
        let previous = changed
        let generation = selectionGeneration
        let artifactRelativePath = selectedArtifact.relativePath
        change(&changed)
        presentation = changed
        viewer.setPresentationOrigin(changed.frame.origin.simd)
        isSaving = true
        engine.updatePresentation(scene: scene, presentation: changed) { [weak self, weak viewer] result in
            guard let self,
                  generation == self.selectionGeneration,
                  artifactRelativePath == self.selectedArtifact?.relativePath
            else { return }
            self.isSaving = false
            switch result {
            case .success(let message):
                do {
                    let saved = try Self.decode(message)
                    self.presentation = saved
                    viewer?.setPresentationOrigin(saved.frame.origin.simd)
                } catch {
                    self.presentation = previous
                    viewer?.setPresentationOrigin(previous.frame.origin.simd)
                    self.errorMessage = error.localizedDescription
                }
            case .failure(let error):
                self.presentation = previous
                viewer?.setPresentationOrigin(previous.frame.origin.simd)
                self.errorMessage = error.localizedDescription
            }
        }
    }

    private func apply(_ profile: ScenePresentation, to viewer: SplatViewerModel) {
        viewer.setPresentationOrigin(profile.frame.origin.simd)
        if let homeViewID = profile.homeViewID,
           let home = profile.views.first(where: { $0.id == homeViewID }) {
            viewer.restoreCamera(home.cameraPose)
        } else if let pivot = profile.defaultPivot {
            viewer.restorePivot(pivot.simd)
        }
    }

    private static func decode(_ message: EngineMessage) throws -> ScenePresentation {
        guard let result = message.result else {
            throw EngineClientError(message: "The engine returned no presentation.")
        }
        let encoded = try JSONEncoder().encode(result)
        return try JSONDecoder().decode(PresentationResponse.self, from: encoded).presentation
    }
}
