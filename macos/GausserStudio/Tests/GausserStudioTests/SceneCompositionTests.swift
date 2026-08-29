import Foundation
import simd
import Testing
@testable import GausserStudio

@Test
func presentationProfileDecodesArtifactLocalComposition() throws {
    let data = Data(#"""
    {
      "version": 1,
      "artifact": {
        "relative_path": "runs/brush/run-a/exports/scene.ply",
        "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      },
      "revision": null,
      "frame": {"origin": {"x": 1, "y": 2, "z": 3}},
      "default_pivot": {"x": 4, "y": 5, "z": 6},
      "focus_points": [{
        "id": "signature",
        "name": "Artist signature",
        "position": {"x": 0.4, "y": 0.3, "z": -0.2},
        "framing_radius": 0.15
      }],
      "views": [{
        "id": "hero",
        "name": "Hero view",
        "position": {"x": 3.8, "y": 1.9, "z": 5.1},
        "target": {"x": 0, "y": 1.2, "z": 0},
        "up": {"x": 0, "y": 1, "z": 0},
        "vertical_fov_degrees": 50
      }],
      "home_view_id": "hero"
    }
    """#.utf8)

    let profile = try JSONDecoder().decode(ScenePresentation.self, from: data)

    #expect(profile.frame.origin == SceneVector3(x: 1, y: 2, z: 3))
    #expect(profile.defaultPivot == SceneVector3(x: 4, y: 5, z: 6))
    #expect(profile.focusPoints.first?.id == "signature")
    #expect(profile.views.first?.verticalFOVDegrees == 50)
    #expect(profile.homeViewID == "hero")
}

@Test @MainActor
func splatCameraStateOwnsOrbitZoomAndPivotOutsideTheRenderer() {
    let model = SplatViewerModel()
    model.restoreCamera(SplatCameraPose(
        position: SIMD3(0, 0, 5),
        target: .zero,
        up: SIMD3(0, 1, 0),
        verticalFOVDegrees: 60
    ))
    let initialDistance = simd_distance(model.camera.position, model.camera.target)

    model.orbit(horizontal: .pi / 2, vertical: 0)

    #expect(simd_distance(model.camera.position, model.camera.target).isApproximatelyEqual(to: initialDistance))
    #expect(model.camera.target == .zero)

    let orbitOffset = model.camera.position - model.camera.target
    model.pan(horizontal: 0.5, vertical: 0.25)
    #expect((model.camera.position - model.camera.target).isApproximatelyEqual(to: orbitOffset))
    #expect(model.camera.target != .zero)

    model.zoom(by: 0.5)
    #expect(simd_distance(model.camera.position, model.camera.target).isApproximatelyEqual(to: initialDistance * 0.5))
}

@Test @MainActor
func compositionModelPersistsSemanticEditsAndRestoresSavedNavigation() throws {
    let scene = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString, directoryHint: .isDirectory)
    let relativePath = "runs/brush/run-a/exports/scene.ply"
    let artifactURL = scene.appending(path: relativePath)
    try FileManager.default.createDirectory(at: artifactURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data("ply".utf8).write(to: artifactURL)
    defer { try? FileManager.default.removeItem(at: scene) }

    let profile = samplePresentation(relativePath: relativePath)
    let engine = PresentationEngineStub(presentation: profile)
    let composition = SceneCompositionModel(scene: scene, engine: engine)
    let viewer = SplatViewerModel()
    let artifact = ArtifactSummary(relativePath: relativePath, kind: "brush", runID: "run-a", format: "ply", sizeBytes: 3)

    composition.select(artifact, viewer: viewer)
    #expect(composition.presentation == profile)
    #expect(viewer.camera == profile.views[0].cameraPose)

    viewer.restorePivot(SIMD3(1, 2, 3))
    composition.setOriginFromPivot(viewer: viewer)
    #expect(engine.updateCount == 1)
    #expect(composition.presentation?.frame.origin == SceneVector3(x: 1, y: 2, z: 3))
    #expect(viewer.presentationOrigin == SIMD3(1, 2, 3))

    composition.addFocusPoint(named: "Plaque", viewer: viewer)
    #expect(engine.updateCount == 2)
    #expect(composition.presentation?.focusPoints.last?.name == "Plaque")
    #expect(composition.presentation?.focusPoints.last?.position == SceneVector3(x: 1, y: 2, z: 3))

    composition.saveView(named: "Detail", viewer: viewer)
    #expect(engine.updateCount == 3)
    #expect(composition.presentation?.views.last?.name == "Detail")
}

@Test @MainActor
func staleSaveCompletionCannotReplaceANewerArtifactSelection() throws {
    let scene = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString, directoryHint: .isDirectory)
    let firstPath = "runs/brush/run-a/exports/first.ply"
    let secondPath = "runs/brush/run-b/exports/second.ply"
    for relativePath in [firstPath, secondPath] {
        let url = scene.appending(path: relativePath)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("ply".utf8).write(to: url)
    }
    defer { try? FileManager.default.removeItem(at: scene) }

    let firstProfile = samplePresentation(relativePath: firstPath)
    let secondProfile = samplePresentation(relativePath: secondPath)
    let engine = DeferredPresentationEngineStub(profiles: [
        firstPath: firstProfile,
        secondPath: secondProfile,
    ])
    let composition = SceneCompositionModel(scene: scene, engine: engine)
    let viewer = SplatViewerModel()
    let first = ArtifactSummary(relativePath: firstPath, kind: "brush", runID: "run-a", format: "ply", sizeBytes: 3)
    let second = ArtifactSummary(relativePath: secondPath, kind: "brush", runID: "run-b", format: "ply", sizeBytes: 3)

    composition.select(first, viewer: viewer)
    composition.addFocusPoint(named: "First artifact point", viewer: viewer)
    #expect(composition.isSaving)

    composition.select(second, viewer: viewer)
    #expect(composition.presentation?.artifact.relativePath == secondPath)
    #expect(!composition.isSaving)

    engine.completePendingUpdate()
    #expect(composition.presentation?.artifact.relativePath == secondPath)
    #expect(composition.presentation?.focusPoints.isEmpty == true)
}

@Test
func cameraViewMatrixMapsTheEyeToOriginAndTargetToTheViewAxis() {
    let pose = SplatCameraPose(
        position: SIMD3(3, 2, 5),
        target: SIMD3(1, 1, 0),
        up: SIMD3(0, 1, 0),
        verticalFOVDegrees: 50
    )

    let eye = pose.viewMatrix * SIMD4(pose.position, 1)
    let target = pose.viewMatrix * SIMD4(pose.target, 1)

    #expect(SIMD3(eye.x, eye.y, eye.z).isApproximatelyEqual(to: .zero))
    #expect(abs(target.x) < 0.0001)
    #expect(abs(target.y) < 0.0001)
    #expect(target.z < 0)
}

@Test
func presentationOriginProjectsIntoTheViewerAsAVisibleGizmo() {
    let camera = SplatCameraPose(
        position: SIMD3(0, 0, 5),
        target: .zero,
        up: SIMD3(0, 1, 0),
        verticalFOVDegrees: 50
    )

    let projected = projectCompositionPoint(
        .zero,
        camera: camera,
        nearZ: 0.01,
        farZ: 100,
        size: CGSize(width: 800, height: 600)
    )

    #expect(projected != nil)
    #expect(abs(projected!.x - 400) < 0.001)
    #expect(abs(projected!.y - 300) < 0.001)
}

@Test @MainActor
func savedViewRoundTripsThePortableCameraPose() {
    let pose = SplatCameraPose(
        position: SIMD3(3.8, 1.9, 5.1),
        target: SIMD3(0, 1.2, 0),
        up: SIMD3(0, 1, 0),
        verticalFOVDegrees: 50
    )
    let view = PresentationView(id: "hero", name: "Hero", camera: pose)

    #expect(view.cameraPose.position.isApproximatelyEqual(to: pose.position))
    #expect(view.cameraPose.target.isApproximatelyEqual(to: pose.target))
    #expect(view.cameraPose.up.isApproximatelyEqual(to: pose.up))
    #expect(view.cameraPose.verticalFOVDegrees == pose.verticalFOVDegrees)
}

private func samplePresentation(relativePath: String) -> ScenePresentation {
    ScenePresentation(
        version: 1,
        artifact: PresentationArtifact(relativePath: relativePath, sha256: String(repeating: "a", count: 64)),
        revision: nil,
        frame: PresentationFrame(origin: SceneVector3(x: 0, y: 0, z: 0)),
        defaultPivot: nil,
        focusPoints: [],
        views: [PresentationView(
            id: "hero",
            name: "Hero",
            position: SceneVector3(x: 0, y: 0, z: 5),
            target: SceneVector3(x: 0, y: 0, z: 0),
            up: SceneVector3(x: 0, y: 1, z: 0),
            verticalFOVDegrees: 50
        )],
        homeViewID: "hero"
    )
}

@MainActor
private final class PresentationEngineStub: PresentationEngineClient {
    var presentationValue: ScenePresentation
    private(set) var updateCount = 0

    init(presentation: ScenePresentation) {
        presentationValue = presentation
    }

    func presentation(
        scene _: URL,
        artifactRelativePath _: String,
        completion: @escaping (Result<EngineMessage, Error>) -> Void
    ) {
        completion(.success(message(presentationValue)))
    }

    func updatePresentation(
        scene _: URL,
        presentation: ScenePresentation,
        completion: @escaping (Result<EngineMessage, Error>) -> Void
    ) {
        updateCount += 1
        var saved = presentation
        saved.revision = "revision-\(updateCount)"
        presentationValue = saved
        completion(.success(message(saved)))
    }

    private func message(_ presentation: ScenePresentation) -> EngineMessage {
        let data = try! JSONEncoder().encode(presentation)
        let value = try! JSONDecoder().decode(JSONValue.self, from: data)
        return EngineMessage(
            id: "test",
            event: nil,
            result: .object(["presentation": value]),
            error: nil,
            run: nil,
            runID: nil,
            line: nil
        )
    }
}

@MainActor
private final class DeferredPresentationEngineStub: PresentationEngineClient {
    let profiles: [String: ScenePresentation]
    private var pending: ((Result<EngineMessage, Error>) -> Void)?
    private var pendingPresentation: ScenePresentation?

    init(profiles: [String: ScenePresentation]) {
        self.profiles = profiles
    }

    func presentation(
        scene _: URL,
        artifactRelativePath: String,
        completion: @escaping (Result<EngineMessage, Error>) -> Void
    ) {
        completion(.success(message(profiles[artifactRelativePath]!)))
    }

    func updatePresentation(
        scene _: URL,
        presentation: ScenePresentation,
        completion: @escaping (Result<EngineMessage, Error>) -> Void
    ) {
        pendingPresentation = presentation
        pending = completion
    }

    func completePendingUpdate() {
        var saved = pendingPresentation!
        saved.revision = "completed-late"
        pending?(.success(message(saved)))
        pending = nil
        pendingPresentation = nil
    }

    private func message(_ presentation: ScenePresentation) -> EngineMessage {
        let data = try! JSONEncoder().encode(presentation)
        let value = try! JSONDecoder().decode(JSONValue.self, from: data)
        return EngineMessage(
            id: "test",
            event: nil,
            result: .object(["presentation": value]),
            error: nil,
            run: nil,
            runID: nil,
            line: nil
        )
    }
}

private extension Float {
    func isApproximatelyEqual(to other: Float, tolerance: Float = 0.0001) -> Bool {
        abs(self - other) <= tolerance
    }
}

private extension SIMD3<Float> {
    func isApproximatelyEqual(to other: SIMD3<Float>, tolerance: Float = 0.0001) -> Bool {
        simd_length(self - other) <= tolerance
    }
}
