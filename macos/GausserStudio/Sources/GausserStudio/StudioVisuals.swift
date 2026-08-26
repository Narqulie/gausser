import SwiftUI

/// Shared presentation primitives. The workflow engine remains the authority;
/// this file only gives that information a consistent macOS-native hierarchy.
struct StudioGlassActions<Content: View>: View {
    private let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    var body: some View {
        if #available(macOS 26.0, *) {
            GlassEffectContainer(spacing: 10) {
                content
            }
        } else {
            content
        }
    }
}

extension View {
    /// Liquid Glass is reserved for short, primary actions. Content and data
    /// panels deliberately use standard materials so accessibility appearance
    /// changes remain legible and predictable.
    @ViewBuilder
    func studioGlassButton() -> some View {
        if #available(macOS 26.0, *) {
            self.buttonStyle(.glass)
        } else {
            self.buttonStyle(.bordered)
        }
    }
}

struct StudioStatusBadge: View {
    let title: String
    let state: String

    private var tint: Color {
        switch state {
        case "ready", "completed": .green
        case "running", "queued": .blue
        case "unavailable": .secondary
        default: .orange
        }
    }

    var body: some View {
        Label {
            Text("\(title) · \(state.capitalized)")
                .font(.caption.weight(.medium))
        } icon: {
            Image(systemName: state == "ready" || state == "completed" ? "checkmark.circle.fill" : "circle.fill")
                .foregroundStyle(tint)
        }
        .foregroundStyle(.secondary)
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.quaternary, in: Capsule())
        .accessibilityLabel("\(title): \(state)")
    }
}

struct StudioContentCard<Content: View>: View {
    let title: String
    let systemImage: String
    private let content: Content

    init(_ title: String, systemImage: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.systemImage = systemImage
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label(title, systemImage: systemImage)
                .font(.headline)
            content
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
    }
}

func studioStageTitle(_ stage: String) -> String {
    switch stage {
    case "images": "Images"
    case "sharpframes": "Sharp Frames"
    case "sammer": "SAM3 Masking"
    case "colmap": "COLMAP"
    case "brush": "Brush"
    case "msplat": "Msplat"
    case "lod": "Streamed LOD"
    default: stage.capitalized
    }
}

func studioStageSymbol(_ stage: String) -> String {
    switch stage {
    case "images": "photo.stack"
    case "sharpframes": "film.stack"
    case "sammer": "square.dashed.inset.filled"
    case "colmap": "camera.aperture"
    case "brush": "sparkles.rectangle.stack"
    case "msplat": "point.3.connected.trianglepath.dotted"
    case "lod": "square.3.layers.3d.down.right"
    default: "gearshape"
    }
}
