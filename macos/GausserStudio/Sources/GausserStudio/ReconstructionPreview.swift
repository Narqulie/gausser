import SwiftUI
import simd

/// Interactive, view-only rendering of COLMAP's completed sparse/0 model.
///
/// COLMAP publishes binary/text sparse geometry rather than a PLY at this stage.
/// The engine supplies a bounded sample of its authoritative coloured points, so
/// the Studio can show the reconstruction without materialising a duplicate file.
struct ReconstructionPreview: View {
    let cameras: [ReconstructionCamera]
    let points: [ReconstructionPoint]
    let cameraCount: Int
    let pointCount: Int

    @State private var orbit = ReconstructionOrbit()
    @State private var previousDrag = CGSize.zero
    @State private var pinchStartZoom: Double?

    var body: some View {
        StudioContentCard("3D sparse reconstruction", systemImage: "cube.transparent") {
            if cameras.isEmpty {
                ContentUnavailableView(
                    "No completed camera model",
                    systemImage: "camera.metering.none",
                    description: Text("The rotatable sparse point cloud appears after COLMAP writes a valid sparse/0 reconstruction."))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 24)
            } else {
                VStack(alignment: .leading, spacing: 12) {
                    SparseModelCanvas(cameras: cameras, points: points, orbit: orbit)
                        .frame(minHeight: 340)
                        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                        .overlay {
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .stroke(.white.opacity(0.12), lineWidth: 1)
                        }
                        .gesture(orbitGesture)
                        .accessibilityLabel("Rotatable 3D sparse COLMAP reconstruction with \(points.count) preview points and \(cameras.count) cameras")
                        .accessibilityHint("Drag to orbit and pinch to zoom")

                    ViewThatFits(in: .horizontal) {
                        HStack(alignment: .firstTextBaseline) {
                            orbitControls
                            Label("Drag to orbit · Pinch to zoom", systemImage: "hand.draw")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Spacer()
                            previewCounts
                            resetViewButton
                        }
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                orbitControls
                                Spacer()
                                resetViewButton
                            }
                            HStack {
                                Label("Drag to orbit · Pinch to zoom", systemImage: "hand.draw")
                                Spacer()
                                previewCounts
                            }
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(.top, 4)
            }
        }
    }

    private var orbitControls: some View {
        ControlGroup {
            Button {
                orbit.yaw -= 0.35
            } label: {
                Label("Rotate left", systemImage: "rotate.left")
            }
            Button {
                orbit.yaw += 0.35
            } label: {
                Label("Rotate right", systemImage: "rotate.right")
            }
            Button {
                orbit.pitch = min(1.35, orbit.pitch + 0.25)
            } label: {
                Label("Rotate up", systemImage: "arrow.up")
            }
            Button {
                orbit.pitch = max(-1.35, orbit.pitch - 0.25)
            } label: {
                Label("Rotate down", systemImage: "arrow.down")
            }
            Button {
                orbit.zoom = max(0.45, orbit.zoom / 1.2)
            } label: {
                Label("Zoom out", systemImage: "minus.magnifyingglass")
            }
            Button {
                orbit.zoom = min(3.5, orbit.zoom * 1.2)
            } label: {
                Label("Zoom in", systemImage: "plus.magnifyingglass")
            }
        }
        .controlGroupStyle(.compactMenu)
    }

    private var previewCounts: some View {
        Text("\(points.count.formatted()) of \(pointCount.formatted()) points · \(cameras.count.formatted()) of \(cameraCount.formatted()) cameras")
            .font(.caption.monospacedDigit())
            .foregroundStyle(.secondary)
    }

    private var resetViewButton: some View {
        Button("Reset View") {
            orbit.reset()
        }
        .buttonStyle(.borderless)
    }

    private var orbitGesture: some Gesture {
        SimultaneousGesture(
            DragGesture()
                .onChanged { value in
                    let delta = CGSize(
                        width: value.translation.width - previousDrag.width,
                        height: value.translation.height - previousDrag.height
                    )
                    orbit.yaw += Double(delta.width) * 0.012
                    orbit.pitch = min(max(orbit.pitch + Double(delta.height) * 0.012, -1.35), 1.35)
                    previousDrag = value.translation
                }
                .onEnded { _ in previousDrag = .zero },
            MagnifyGesture()
                .onChanged { value in
                    if pinchStartZoom == nil { pinchStartZoom = orbit.zoom }
                    orbit.zoom = min(max((pinchStartZoom ?? 1) / Double(value.magnification), 0.45), 3.5)
                }
                .onEnded { _ in pinchStartZoom = nil }
        )
    }
}

private struct SparseModelCanvas: View {
    let cameras: [ReconstructionCamera]
    let points: [ReconstructionPoint]
    let orbit: ReconstructionOrbit

    var body: some View {
        let viewport = ReconstructionViewport(cameras: cameras, points: points)
        Canvas { context, size in
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(Color(red: 0.035, green: 0.045, blue: 0.065)))
            drawReferenceGrid(in: &context, size: size)
            drawPoints(in: &context, size: size, viewport: viewport)
            drawCameras(in: &context, size: size, viewport: viewport)
        }
    }

    private func drawReferenceGrid(in context: inout GraphicsContext, size: CGSize) {
        var path = Path()
        let spacing: CGFloat = 48
        for x in stride(from: 0, through: size.width, by: spacing) {
            path.move(to: CGPoint(x: x, y: 0))
            path.addLine(to: CGPoint(x: x, y: size.height))
        }
        for y in stride(from: 0, through: size.height, by: spacing) {
            path.move(to: CGPoint(x: 0, y: y))
            path.addLine(to: CGPoint(x: size.width, y: y))
        }
        context.stroke(path, with: .color(.white.opacity(0.035)), lineWidth: 1)
    }

    private func drawPoints(in context: inout GraphicsContext, size: CGSize, viewport: ReconstructionViewport) {
        for point in points {
            guard let projected = viewport.project(point.position, size: size, orbit: orbit) else { continue }
            let depthOpacity = min(max(0.22 + projected.depth * 0.14, 0.24), 0.9)
            let radius = min(max(0.8 + projected.depth * 0.35, 0.9), 2.0)
            let rect = CGRect(
                x: projected.position.x - radius,
                y: projected.position.y - radius,
                width: radius * 2,
                height: radius * 2
            )
            context.fill(
                Path(ellipseIn: rect),
                with: .color(Color(
                    red: Double(point.red) / 255,
                    green: Double(point.green) / 255,
                    blue: Double(point.blue) / 255
                ).opacity(depthOpacity))
            )
        }
    }

    private func drawCameras(in context: inout GraphicsContext, size: CGSize, viewport: ReconstructionViewport) {
        for camera in cameras {
            guard let projected = viewport.project(camera.position, size: size, orbit: orbit) else { continue }
            let radius: CGFloat = 4
            let rect = CGRect(
                x: projected.position.x - radius,
                y: projected.position.y - radius,
                width: radius * 2,
                height: radius * 2
            )
            context.fill(Path(ellipseIn: rect), with: .color(.cyan.opacity(0.95)))
            context.stroke(Path(ellipseIn: rect), with: .color(.white.opacity(0.8)), lineWidth: 1)
        }
    }
}

struct ReconstructionViewport {
    let center: SIMD3<Double>
    let scale: Double

    init(cameras: [ReconstructionCamera], points: [ReconstructionPoint]) {
        let positions = cameras.map(\.position) + points.map(\.position)
        guard let first = positions.first else {
            center = .zero
            scale = 1
            return
        }
        var minimum = first
        var maximum = first
        for position in positions.dropFirst() {
            minimum = simd_min(minimum, position)
            maximum = simd_max(maximum, position)
        }
        center = (minimum + maximum) / 2
        scale = 1 / max(simd_length(maximum - minimum) / 2, 0.001)
    }

    fileprivate func project(_ position: SIMD3<Double>, size: CGSize, orbit: ReconstructionOrbit) -> ProjectedReconstructionPoint? {
        var point = (position - center) * scale
        let yawCosine = cos(orbit.yaw)
        let yawSine = sin(orbit.yaw)
        point = SIMD3(
            yawCosine * point.x - yawSine * point.z,
            point.y,
            yawSine * point.x + yawCosine * point.z
        )
        let pitchCosine = cos(orbit.pitch)
        let pitchSine = sin(orbit.pitch)
        point = SIMD3(
            point.x,
            pitchCosine * point.y - pitchSine * point.z,
            pitchSine * point.y + pitchCosine * point.z
        )
        let depth = point.z + 3.2 / orbit.zoom
        guard depth > 0.05 else { return nil }
        let focalLength = Double(min(size.width, size.height)) * 0.92
        return ProjectedReconstructionPoint(
            position: CGPoint(
                x: size.width / 2 + CGFloat(point.x / depth * focalLength),
                y: size.height / 2 - CGFloat(point.y / depth * focalLength)
            ),
            depth: depth
        )
    }
}

private struct ProjectedReconstructionPoint {
    let position: CGPoint
    let depth: Double
}

private struct ReconstructionOrbit {
    var yaw = -0.65
    var pitch = 0.28
    var zoom = 1.0

    mutating func reset() {
        self = ReconstructionOrbit()
    }
}

private extension ReconstructionPoint {
    var position: SIMD3<Double> { SIMD3(x, y, z) }
}

private extension ReconstructionCamera {
    var position: SIMD3<Double> { SIMD3(x, y, z) }
}
