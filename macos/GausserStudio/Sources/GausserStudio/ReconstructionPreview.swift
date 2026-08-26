import SwiftUI

/// A truthful, dependency-free preview of an already written COLMAP model.
///
/// It deliberately draws only camera centres in the model's X/Z plane. COLMAP
/// does not publish stable camera poses while its mapper is still running, so
/// this view refreshes after the engine observes a completed sparse/0 model.
struct ReconstructionPreview: View {
    let cameras: [ReconstructionCamera]

    var body: some View {
        StudioContentCard("Camera reconstruction", systemImage: "camera.aperture") {
            if cameras.isEmpty {
                ContentUnavailableView(
                    "No completed camera model",
                    systemImage: "camera.metering.none",
                    description: Text("Camera positions appear here after COLMAP writes a valid sparse/0 reconstruction."))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 24)
            } else {
                VStack(alignment: .leading, spacing: 8) {
                    CameraMap(cameras: cameras)
                        .frame(minHeight: 260)
                        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                    Text("Top-down X/Z camera centres · \(cameras.count) registered views")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 4)
            }
        }
    }
}

private struct CameraMap: View {
    let cameras: [ReconstructionCamera]

    var body: some View {
        Canvas { context, size in
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(.black.opacity(0.78)))
            let padding: CGFloat = 20
            let xs = cameras.map(\.x)
            let zs = cameras.map(\.z)
            let minX = xs.min() ?? 0
            let maxX = xs.max() ?? 0
            let minZ = zs.min() ?? 0
            let maxZ = zs.max() ?? 0
            let spanX = max(maxX - minX, 0.001)
            let spanZ = max(maxZ - minZ, 0.001)
            let scale = min((size.width - padding * 2) / spanX, (size.height - padding * 2) / spanZ)
            let offsetX = (size.width - spanX * scale) / 2 - minX * scale
            let offsetZ = (size.height - spanZ * scale) / 2 - minZ * scale

            for camera in cameras {
                let point = CGPoint(x: camera.x * scale + offsetX, y: size.height - (camera.z * scale + offsetZ))
                context.fill(Path(ellipseIn: CGRect(x: point.x - 3, y: point.y - 3, width: 6, height: 6)), with: .color(.cyan))
            }
        }
        .accessibilityLabel("Top-down map of \(cameras.count) reconstructed camera positions")
    }
}
