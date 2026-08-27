import Testing
import simd
@testable import GausserStudio

@Test
func sparsePreviewOrbitsAboutTheEngineDerivedCameraFocus() {
    let focus = ReconstructionFocus(x: 10, y: 20, z: 30)
    let viewport = ReconstructionViewport(
        cameras: [ReconstructionCamera(name: "camera", x: 0, y: 0, z: 0)],
        points: [ReconstructionPoint(x: 20, y: 20, z: 30, red: 1, green: 2, blue: 3)],
        focus: focus
    )

    #expect(viewport.center == SIMD3(10, 20, 30))
}

@Test
func sparsePreviewOffersOpenCVFriendlyAndRawDisplayAxes() {
    let colmap = SIMD3<Double>(2, 3, 4)

    // This is a user-selectable display orientation: COLMAP world axes are arbitrary.
    #expect(SparsePreviewAxisConvention.opencvFriendly.transform(colmap) == SIMD3(2, -3, -4))
    #expect(SparsePreviewAxisConvention.rawCOLMAP.transform(colmap) == colmap)
}
