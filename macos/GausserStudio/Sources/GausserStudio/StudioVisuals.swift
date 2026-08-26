import SwiftUI

/// The consistent spatial rhythm for pages, sheets, cards, and persistent
/// actions. macOS sheets have rounded edges; retaining this inset prevents
/// content and controls from crowding those edges at every supported size.
enum StudioLayout {
    static let pageInset: CGFloat = 28
    static let contentWidth: CGFloat = 860
    static let wideContentWidth: CGFloat = 1_100
    static let cardInset: CGFloat = 18
    static let cardCornerRadius: CGFloat = 18
}

/// A scrollable desktop page with a single, deliberate content measure. It
/// avoids every workflow screen independently choosing edge padding.
struct StudioScrollPage<Content: View>: View {
    let maxWidth: CGFloat
    private let content: Content

    init(maxWidth: CGFloat = StudioLayout.contentWidth, @ViewBuilder content: () -> Content) {
        self.maxWidth = maxWidth
        self.content = content()
    }

    var body: some View {
        ScrollView {
            content
                .frame(maxWidth: maxWidth, alignment: .leading)
                .padding(StudioLayout.pageInset)
                .frame(maxWidth: .infinity, alignment: .center)
        }
        .scrollBounceBehavior(.basedOnSize)
    }
}

/// A persistent, material-backed action area. Used with `safeAreaInset` so
/// crucial controls stay above the scrollable content rather than being pushed
/// below it by long forms or commands.
struct StudioActionBar<Content: View>: View {
    private let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    var body: some View {
        content
            .frame(maxWidth: StudioLayout.contentWidth, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .center)
            .padding(.horizontal, StudioLayout.pageInset)
            .padding(.vertical, 16)
            .background(.bar)
            .overlay(alignment: .top) { Divider() }
    }
}

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
        .padding(StudioLayout.cardInset)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background, in: RoundedRectangle(cornerRadius: StudioLayout.cardCornerRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: StudioLayout.cardCornerRadius, style: .continuous)
                .stroke(.quaternary, lineWidth: 1)
        }
    }
}

extension View {
    /// Applies the same standard content surface to interactive cards without
    /// using Liquid Glass as a content background.
    func studioCardSurface(cornerRadius: CGFloat = StudioLayout.cardCornerRadius) -> some View {
        background(.background, in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .stroke(.quaternary, lineWidth: 1)
            }
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
