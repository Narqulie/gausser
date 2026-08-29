import AppKit
import simd
import SwiftUI

struct SceneCompositionWorkspace: View {
    let scene: URL
    let artifacts: [ArtifactSummary]
    @Bindable var composition: SceneCompositionModel
    @Bindable var viewer: SplatViewerModel
    let onReveal: (URL) -> Void
    let onNavigate: (SceneStep) -> Void

    @State private var focusName = ""
    @State private var viewName = ""

    private var nativeArtifacts: [ArtifactSummary] {
        artifacts
            .filter { ["ply", "spz", "splat"].contains($0.format.lowercased()) }
            .sorted { $0.relativePath > $1.relativePath }
    }

    var body: some View {
        Group {
            if let artifact = composition.selectedArtifact {
                editor(artifact)
            } else {
                artifactPicker
            }
        }
        .alert(
            "Composition error",
            isPresented: Binding(
                get: { composition.errorMessage != nil },
                set: { if !$0 { composition.errorMessage = nil } }
            )
        ) {
            if composition.selectedArtifact != nil {
                Button("Reload", action: { composition.reload(viewer: viewer) })
            }
            Button("OK", role: .cancel) {}
        } message: {
            Text(composition.errorMessage ?? "")
        }
    }

    private var artifactPicker: some View {
        StudioScrollPage {
            VStack(alignment: .leading, spacing: 24) {
                StepPageHeader(
                    step: .compose,
                    title: "Compose points and views of interest",
                    message: "Choose a trained or cleaned splat. Composition changes only scene-local presentation metadata; the immutable artifact is never rewritten."
                )
                StudioContentCard("Splat artifacts", systemImage: "cube.transparent") {
                    if nativeArtifacts.isEmpty {
                        ContentUnavailableView(
                            "No viewable splat yet",
                            systemImage: "cube.transparent",
                            description: Text("Publish a PLY, SPZ, or .splat artifact from training or SuperSplat reimport first."))
                    } else {
                        ForEach(nativeArtifacts) { artifact in
                            artifactRow(artifact)
                        }
                    }
                }
                navigationFooter
            }
        }
    }

    private func artifactRow(_ artifact: ArtifactSummary) -> some View {
        HStack(alignment: .center, spacing: 14) {
            Image(systemName: "cube.transparent")
                .font(.title3)
                .foregroundStyle(.tint)
                .frame(width: 40, height: 40)
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 11, style: .continuous))
            VStack(alignment: .leading, spacing: 4) {
                Text(artifact.relativePath)
                    .font(.system(.subheadline, design: .monospaced))
                    .lineLimit(2)
                    .textSelection(.enabled)
                Text("\(artifact.kind) · \(artifact.format.uppercased()) · \(ByteCountFormatter.string(fromByteCount: Int64(artifact.sizeBytes), countStyle: .file))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Reveal") { onReveal(scene.appending(path: artifact.relativePath)) }
                .buttonStyle(.bordered)
            Button("Compose") {
                focusName = ""
                viewName = ""
                composition.select(artifact, viewer: viewer)
            }
            .studioGlassButton()
        }
        .padding(14)
        .studioCardSurface(cornerRadius: 12)
    }

    private func editor(_ artifact: ArtifactSummary) -> some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                Button("Artifacts", systemImage: "chevron.left") {
                    composition.clear(viewer: viewer)
                }
                .buttonStyle(.bordered)
                .disabled(composition.isSaving)
                VStack(alignment: .leading, spacing: 2) {
                    Text("SCENE COMPOSITION")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Text(artifact.relativePath)
                        .font(.system(.subheadline, design: .monospaced))
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer()
                if composition.isLoading {
                    ProgressView().controlSize(.small)
                    Text("Loading profile…").font(.caption).foregroundStyle(.secondary)
                } else if composition.isSaving {
                    ProgressView().controlSize(.small)
                    Text("Saving…").font(.caption).foregroundStyle(.secondary)
                } else if let revision = composition.presentation?.revision {
                    Label("Saved", systemImage: "checkmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(.green)
                        .help("Revision \(revision)")
                } else {
                    Text("Not saved yet").font(.caption).foregroundStyle(.secondary)
                }
                Button("Reveal", systemImage: "folder") {
                    onReveal(scene.appending(path: artifact.relativePath))
                }
                .buttonStyle(.bordered)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 12)

            Divider()

            HSplitView {
                viewerPane(artifact)
                    .frame(minWidth: 560, maxWidth: .infinity, maxHeight: .infinity)
                inspector
                    .frame(minWidth: 300, idealWidth: 350, maxWidth: 440, maxHeight: .infinity)
            }

            Divider()
            navigationFooter
                .padding(.horizontal, 20)
                .padding(.vertical, 10)
        }
    }

    @ViewBuilder
    private func viewerPane(_ artifact: ArtifactSummary) -> some View {
        ZStack {
            Color.black.opacity(0.94)
            switch viewer.state {
            case .idle:
                ContentUnavailableView("No artifact selected", systemImage: "cube.transparent")
            case .loading:
                VStack(spacing: 12) {
                    ProgressView().controlSize(.large)
                    Text("Loading \(artifact.format.uppercased()) artifact…")
                        .foregroundStyle(.secondary)
                }
            case .loaded(_, let splatCount):
                SplatViewerHost(model: viewer)
                CompositionOverlay(viewer: viewer, profile: composition.presentation)
                VStack {
                    Spacer()
                    HStack {
                        Label("\(splatCount.formatted()) splats", systemImage: "circle.grid.cross")
                        Spacer()
                        Text("Drag: orbit  ·  Shift-drag: move pivot  ·  Pinch: zoom  ·  RGB: origin axes")
                    }
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.82))
                    .padding(10)
                    .background(.black.opacity(0.48), in: Capsule())
                    .padding(16)
                }
            case .failed(_, let message):
                ContentUnavailableView(
                    "Unable to load splat",
                    systemImage: "exclamationmark.triangle",
                    description: Text(message))
            }
        }
    }

    private var inspector: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                inspectorHeader
                pivotCard
                originCard
                focusPointsCard
                savedViewsCard
            }
            .padding(16)
        }
        .background(.bar)
    }

    private var inspectorHeader: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Points & views")
                .font(.title2.weight(.semibold))
            Text("All coordinates remain in this artifact’s local space. Moving the presentation origin does not move or rewrite splats.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var pivotCard: some View {
        compositionCard("Orbit pivot", symbol: "scope") {
            Text(coordinates(viewer.camera.target))
                .font(.system(.caption, design: .monospaced))
                .textSelection(.enabled)
            Text("Shift-drag the model until the point of interest sits beneath the crosshair.")
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack {
                Button("Save Default", systemImage: "pin.fill") {
                    composition.saveDefaultPivot(viewer: viewer)
                }
                .disabled(!composition.canEdit)
                if composition.presentation?.defaultPivot != nil {
                    Button("Restore") { composition.restoreDefaultPivot(viewer: viewer) }
                        .disabled(!composition.canEdit)
                }
            }
            .buttonStyle(.bordered)
        }
    }

    private var originCard: some View {
        compositionCard("Presentation origin", symbol: "move.3d") {
            if let origin = composition.presentation?.frame.origin {
                Text(coordinates(origin.simd))
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                Text("The red, green, and blue axes in the viewer mark this saved artifact-local frame origin.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                HStack {
                    Button("Set at Pivot", systemImage: "scope") {
                        composition.setOriginFromPivot(viewer: viewer)
                    }
                    Button("Aim") { viewer.restorePivot(origin.simd) }
                    Button("Reset") { composition.resetOrigin(viewer: viewer) }
                }
                .buttonStyle(.bordered)
                .disabled(!composition.canEdit)
            } else {
                ProgressView()
            }
        }
    }

    private var focusPointsCard: some View {
        compositionCard("Focus points", symbol: "mappin.and.ellipse") {
            if let points = composition.presentation?.focusPoints, !points.isEmpty {
                ForEach(points) { point in
                    HStack(spacing: 8) {
                        Button(point.name) { composition.restoreFocusPoint(point, viewer: viewer) }
                            .buttonStyle(.plain)
                            .font(.callout.weight(.medium))
                        Spacer()
                        Text(coordinates(point.position.simd))
                            .font(.system(.caption2, design: .monospaced))
                            .foregroundStyle(.secondary)
                        Button("Delete", systemImage: "trash", role: .destructive) {
                            composition.deleteFocusPoint(id: point.id, viewer: viewer)
                        }
                        .labelStyle(.iconOnly)
                        .buttonStyle(.borderless)
                        .disabled(!composition.canEdit)
                    }
                    Divider()
                }
            } else {
                Text("No focus points saved.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            HStack {
                TextField("Point name", text: $focusName)
                Button("Add at Pivot") {
                    composition.addFocusPoint(named: focusName, viewer: viewer)
                    if composition.errorMessage == nil { focusName = "" }
                }
                .disabled(!composition.canEdit || focusName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
    }

    private var savedViewsCard: some View {
        compositionCard("Saved views", symbol: "camera.viewfinder") {
            if let profile = composition.presentation, !profile.views.isEmpty {
                ForEach(profile.views) { view in
                    HStack(spacing: 8) {
                        Button(view.name) { composition.restoreView(view, viewer: viewer) }
                            .buttonStyle(.plain)
                            .font(.callout.weight(.medium))
                        Spacer()
                        Button(
                            profile.homeViewID == view.id ? "Home view" : "Set as home",
                            systemImage: profile.homeViewID == view.id ? "star.fill" : "star"
                        ) {
                            composition.setHomeView(id: view.id, viewer: viewer)
                        }
                        .labelStyle(.iconOnly)
                        .buttonStyle(.borderless)
                        .disabled(!composition.canEdit || profile.homeViewID == view.id)
                        Button("Delete", systemImage: "trash", role: .destructive) {
                            composition.deleteView(id: view.id, viewer: viewer)
                        }
                        .labelStyle(.iconOnly)
                        .buttonStyle(.borderless)
                        .disabled(!composition.canEdit)
                    }
                    Divider()
                }
            } else {
                Text("No camera views saved.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            HStack {
                TextField("View name", text: $viewName)
                Button("Save Current") {
                    composition.saveView(named: viewName, viewer: viewer)
                    if composition.errorMessage == nil { viewName = "" }
                }
                .disabled(!composition.canEdit || viewName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
    }

    private func compositionCard<Content: View>(
        _ title: String,
        symbol: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(title, systemImage: symbol)
                .font(.headline)
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .studioCardSurface(cornerRadius: 14)
    }

    private var navigationFooter: some View {
        HStack {
            Button("Back to Training") { onNavigate(.training) }
                .buttonStyle(.bordered)
                .disabled(composition.isSaving)
            Spacer()
            Button("Continue to Publish & LOD") { onNavigate(.publish) }
                .studioGlassButton()
                .disabled(composition.isSaving)
        }
    }

    private func coordinates(_ value: SIMD3<Float>) -> String {
        "x \(value.x.formatted(.number.precision(.fractionLength(3))))  y \(value.y.formatted(.number.precision(.fractionLength(3))))  z \(value.z.formatted(.number.precision(.fractionLength(3))))"
    }
}

private struct CompositionOverlay: View {
    @Bindable var viewer: SplatViewerModel
    let profile: ScenePresentation?

    var body: some View {
        ZStack {
            Canvas { context, size in
                guard let profile else { return }
                drawOrigin(profile.frame.origin.simd, context: &context, size: size)
                for point in profile.focusPoints {
                    drawFocusPoint(point, context: &context, size: size)
                }
            }
            CompositionCrosshair()
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }

    private func drawOrigin(
        _ origin: SIMD3<Float>,
        context: inout GraphicsContext,
        size: CGSize
    ) {
        guard let projectedOrigin = project(origin, size: size) else { return }
        let axisLength = max(simd_distance(viewer.camera.position, viewer.camera.target) * 0.08, 0.001)
        let axes: [(SIMD3<Float>, Color, String)] = [
            (SIMD3(axisLength, 0, 0), .red, "X"),
            (SIMD3(0, axisLength, 0), .green, "Y"),
            (SIMD3(0, 0, axisLength), .blue, "Z"),
        ]
        for (axis, color, label) in axes {
            guard let endpoint = project(origin + axis, size: size) else { continue }
            var path = Path()
            path.move(to: projectedOrigin)
            path.addLine(to: endpoint)
            context.stroke(path, with: .color(color), lineWidth: 2)
            context.draw(
                Text(label).font(.caption2.bold()).foregroundStyle(color),
                at: endpoint,
                anchor: .center
            )
        }
        context.fill(
            Path(ellipseIn: CGRect(x: projectedOrigin.x - 4, y: projectedOrigin.y - 4, width: 8, height: 8)),
            with: .color(.white)
        )
        context.draw(
            Text("Origin").font(.caption2.weight(.semibold)).foregroundStyle(.white),
            at: CGPoint(x: projectedOrigin.x + 8, y: projectedOrigin.y - 10),
            anchor: .leading
        )
    }

    private func drawFocusPoint(
        _ point: PresentationFocusPoint,
        context: inout GraphicsContext,
        size: CGSize
    ) {
        guard let projected = project(point.position.simd, size: size) else { return }
        let bounds = CGRect(x: projected.x - 5, y: projected.y - 5, width: 10, height: 10)
        context.stroke(Path(ellipseIn: bounds), with: .color(.cyan), lineWidth: 2)
        context.draw(
            Text(point.name).font(.caption2.weight(.medium)).foregroundStyle(.white),
            at: CGPoint(x: projected.x + 9, y: projected.y),
            anchor: .leading
        )
    }

    private func project(_ point: SIMD3<Float>, size: CGSize) -> CGPoint? {
        projectCompositionPoint(
            point,
            camera: viewer.camera,
            nearZ: viewer.nearZ,
            farZ: viewer.farZ,
            size: size
        )
    }
}

func projectCompositionPoint(
    _ point: SIMD3<Float>,
    camera: SplatCameraPose,
    nearZ: Float,
    farZ: Float,
    size: CGSize
) -> CGPoint? {
    guard size.width > 0, size.height > 0 else { return nil }
    let projection = camera.projection(
        aspectRatio: Float(size.width / size.height),
        nearZ: nearZ,
        farZ: farZ
    )
    let clip = projection * camera.viewMatrix * SIMD4(point, 1)
    guard clip.w.isFinite, clip.w > 0.000_001 else { return nil }
    let normalized = SIMD3(clip.x, clip.y, clip.z) / clip.w
    guard normalized.x.isFinite,
          normalized.y.isFinite,
          normalized.z >= 0,
          normalized.z <= 1,
          abs(normalized.x) <= 1.2,
          abs(normalized.y) <= 1.2
    else { return nil }
    return CGPoint(
        x: (Double(normalized.x) + 1) * 0.5 * size.width,
        y: (1 - Double(normalized.y)) * 0.5 * size.height
    )
}

private struct CompositionCrosshair: View {
    var body: some View {
        ZStack {
            Circle()
                .stroke(.white.opacity(0.9), lineWidth: 1)
                .frame(width: 18, height: 18)
            Rectangle()
                .fill(.white.opacity(0.9))
                .frame(width: 30, height: 1)
            Rectangle()
                .fill(.white.opacity(0.9))
                .frame(width: 1, height: 30)
        }
        .shadow(color: .black.opacity(0.8), radius: 2)
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}
