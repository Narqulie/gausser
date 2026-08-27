import Testing
import simd
@testable import GausserStudio

@Test
func sparsePreviewOffersOpenCVFriendlyAndRawDisplayAxes() {
    let colmap = SIMD3<Double>(2, 3, 4)

    // This is a user-selectable display orientation: COLMAP world axes are arbitrary.
    #expect(SparsePreviewAxisConvention.opencvFriendly.transform(colmap) == SIMD3(2, -3, -4))
    #expect(SparsePreviewAxisConvention.rawCOLMAP.transform(colmap) == colmap)
}
