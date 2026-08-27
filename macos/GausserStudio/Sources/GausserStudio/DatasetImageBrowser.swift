import AppKit
import ImageIO
import SwiftUI

/// Browses and edits the exact scene-owned dataset. The engine remains the
/// authority for image discovery, selection mutation, and downstream invalidation.
struct DatasetImageBrowser: View {
    let scene: URL
    let images: [DatasetImageSummary]
    let revision: String
    let imageCount: Int
    let hasMore: Bool
    let isLoading: Bool
    let isMutating: Bool
    let onLoadMore: () -> Void
    let onAdd: ([URL]) -> Void
    let onRemove: ([String]) -> Void

    @State private var selectedImage: DatasetImageSummary?
    @State private var selectedPaths = Set<String>()
    @State private var selectionAnchor: Int?
    @State private var showMaskOverlay = true
    @State private var confirmsRemoval = false

    private var maskCount: Int { images.filter { $0.maskRelativePath != nil }.count }

    var body: some View {
        StudioContentCard("Published image dataset", systemImage: "photo.on.rectangle.angled") {
            if imageCount == 0 {
                ContentUnavailableView(
                    "No images published",
                    systemImage: "photo",
                    description: Text("Import a photo folder or select frames from video to create this scene-owned dataset.")
                )
            } else {
                VStack(alignment: .leading, spacing: 14) {
                    header
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 142, maximum: 190), spacing: 12)], spacing: 12) {
                        ForEach(images) { image in
                            Button { select(image) } label: {
                                DatasetThumbnail(
                                    url: imageURL(for: image),
                                    maskURL: maskURL(for: image),
                                    name: image.name,
                                    revision: revision,
                                    showMaskOverlay: showMaskOverlay
                                )
                                .id(DatasetImageCacheKey(path: image.relativePath, revision: revision))
                                .overlay {
                                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                                        .stroke(selectedPaths.contains(image.relativePath) ? Color.accentColor : .clear, lineWidth: 3)
                                }
                            }
                            .buttonStyle(.plain)
                            .accessibilityLabel(selectedPaths.contains(image.relativePath) ? "Selected \(image.name)" : "Select \(image.name)")
                            .accessibilityHint("Command-click selects multiple images. Shift-click selects a range. Use Inspect Selection to preview one image.")
                        }
                    }

                    if hasMore {
                        Button("Show more images", systemImage: "plus") { onLoadMore() }
                            .disabled(isLoading || isMutating)
                            .buttonStyle(.bordered)
                    } else if isLoading {
                        ProgressView("Loading images…")
                    }
                }
            }
        }
        .onChange(of: revision) { _, _ in
            selectedPaths.removeAll()
            selectionAnchor = nil
            selectedImage = nil
        }
        .onDeleteCommand { requestRemoval() }
        .confirmationDialog(
            "Remove \(selectedPaths.count) photo\(selectedPaths.count == 1 ? "" : "s")?",
            isPresented: $confirmsRemoval,
            titleVisibility: .visible
        ) {
            Button("Remove and reset derived outputs", role: .destructive) {
                let removals = images.filter { selectedPaths.contains($0.relativePath) }.map(\.name)
                selectedPaths.removeAll()
                selectionAnchor = nil
                onRemove(removals)
            }
        } message: {
            Text("The selected published photos will be removed. Existing masks, COLMAP reconstruction, and training or LOD outputs are archived because they no longer match the dataset.")
        }
        .sheet(item: $selectedImage) { image in
            DatasetImageDetail(
                image: image,
                url: imageURL(for: image),
                maskURL: maskURL(for: image),
                revision: revision
            )
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text("\(imageCount) canonical images · \(images.count) shown")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Add Photos…", systemImage: "plus") {
                    let photos = SourcePicker.choosePhotoFiles()
                    guard !photos.isEmpty else { return }
                    onAdd(photos)
                }
                .buttonStyle(.bordered)
                .disabled(isMutating)

                Button("Inspect Selection", systemImage: "eye") {
                    selectedImage = images.first { selectedPaths.contains($0.relativePath) }
                }
                .buttonStyle(.bordered)
                .disabled(selectedPaths.count != 1 || isMutating)

                Button("Remove Selection", systemImage: "trash") { requestRemoval() }
                    .buttonStyle(.bordered)
                    .disabled(selectedPaths.isEmpty || isMutating)
            }
            if maskCount > 0 {
                Toggle("Show SAM masks on thumbnails (\(maskCount) available)", isOn: $showMaskOverlay)
                    .font(.callout)
            }
        }
    }

    private func requestRemoval() {
        guard !selectedPaths.isEmpty, !isMutating else { return }
        confirmsRemoval = true
    }

    private func select(_ image: DatasetImageSummary) {
        guard let index = images.firstIndex(where: { $0.relativePath == image.relativePath }) else { return }
        let flags = NSEvent.modifierFlags
        if flags.contains(.shift), let selectionAnchor {
            let range = min(selectionAnchor, index)...max(selectionAnchor, index)
            if flags.contains(.command) {
                selectedPaths.formUnion(range.map { images[$0].relativePath })
            } else {
                selectedPaths = Set(range.map { images[$0].relativePath })
            }
        } else if flags.contains(.command) {
            if !selectedPaths.insert(image.relativePath).inserted {
                selectedPaths.remove(image.relativePath)
            }
            selectionAnchor = index
        } else {
            selectedPaths = [image.relativePath]
            selectionAnchor = index
        }
    }

    private func imageURL(for image: DatasetImageSummary) -> URL {
        scene.appending(path: image.relativePath)
    }

    private func maskURL(for image: DatasetImageSummary) -> URL? {
        image.maskRelativePath.map { scene.appending(path: $0) }
    }
}

private struct DatasetImageCacheKey: Hashable {
    let path: String
    let revision: String
}

private struct DatasetThumbnail: View {
    let url: URL
    let maskURL: URL?
    let name: String
    let revision: String
    let showMaskOverlay: Bool
    @State private var thumbnail: NSImage?
    @State private var maskThumbnail: NSImage?

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            ZStack {
                if let thumbnail {
                    Image(nsImage: thumbnail)
                        .resizable()
                        .scaledToFill()
                } else {
                    ContentUnavailableView("", systemImage: "photo")
                        .symbolRenderingMode(.hierarchical)
                }
                if showMaskOverlay, let maskThumbnail {
                    MaskTint(image: maskThumbnail)
                }
            }
            .frame(height: 106)
            .frame(maxWidth: .infinity)
            .background(.quaternary, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))

            Text(name)
                .font(.caption)
                .lineLimit(1)
                .truncationMode(.middle)
                .foregroundStyle(.primary)
        }
        .task(id: DatasetImageCacheKey(path: url.path, revision: revision)) {
            thumbnail = nil
            maskThumbnail = nil
            thumbnail = await DatasetImageDecoder.shared.thumbnail(for: url)
            if let maskURL {
                maskThumbnail = await DatasetImageDecoder.shared.thumbnail(for: maskURL)
            }
        }
    }
}

private struct DatasetImageDetail: View {
    private enum PreviewState {
        case loading
        case loaded(NSImage)
        case failed
    }

    let image: DatasetImageSummary
    let url: URL
    let maskURL: URL?
    let revision: String
    @Environment(\.dismiss) private var dismiss
    @State private var previewState = PreviewState.loading
    @State private var maskPreviewState = PreviewState.loading
    @State private var retryID = UUID()

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                VStack(alignment: .leading) {
                    Text(image.name).font(.title3.weight(.semibold))
                    Text(ByteCountFormatter.string(fromByteCount: Int64(image.sizeBytes), countStyle: .file))
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Reveal in Finder", systemImage: "folder") {
                    NSWorkspace.shared.activateFileViewerSelecting([url])
                }
                Button("Done") { dismiss() }
            }
            Divider()
            preview
                .frame(minWidth: 560, minHeight: 420)
        }
        .padding(28)
        .task(id: DatasetImageCacheKey(path: url.path, revision: "\(revision):\(retryID.uuidString)")) {
            previewState = .loading
            maskPreviewState = .loading
            let preview = await DatasetImageDecoder.shared.preview(for: url)
            guard !Task.isCancelled else { return }
            previewState = preview.map(PreviewState.loaded) ?? .failed
            if let maskURL {
                let mask = await DatasetImageDecoder.shared.preview(for: maskURL)
                guard !Task.isCancelled else { return }
                maskPreviewState = mask.map(PreviewState.loaded) ?? .failed
            }
        }
    }

    @ViewBuilder
    private var preview: some View {
        switch previewState {
        case .loading:
            ProgressView("Loading preview…")
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        case .loaded(let source):
            if let maskURL {
                HStack(spacing: 18) {
                    previewImage(source, label: "Source image")
                    switch maskPreviewState {
                    case .loading:
                        ProgressView("Loading SAM mask…").frame(maxWidth: .infinity, maxHeight: .infinity)
                    case .loaded(let mask):
                        ZStack {
                            Image(nsImage: source).resizable().scaledToFit()
                            MaskTint(image: mask)
                        }
                        .accessibilityLabel("Source image with SAM mask")
                    case .failed:
                        ContentUnavailableView("Mask preview unavailable", systemImage: "exclamationmark.triangle")
                    }
                }
                .overlay(alignment: .topTrailing) {
                    Button("Reveal Mask", systemImage: "folder") {
                        NSWorkspace.shared.activateFileViewerSelecting([maskURL])
                    }
                    .buttonStyle(.bordered)
                    .padding(8)
                }
            } else {
                previewImage(source, label: "Source image")
            }
        case .failed:
            ContentUnavailableView {
                Label("Preview unavailable", systemImage: "exclamationmark.triangle")
            } description: {
                Text("Gausser could not read this published image.")
            } actions: {
                Button("Try Again") { retryID = UUID() }
            }
        }
    }

    private func previewImage(_ image: NSImage, label: String) -> some View {
        Image(nsImage: image)
            .resizable()
            .scaledToFit()
            .accessibilityLabel(label)
    }
}

private struct MaskTint: View {
    let image: NSImage

    var body: some View {
        Image(nsImage: image)
            .renderingMode(.template)
            .resizable()
            .scaledToFit()
            .foregroundStyle(Color.red.opacity(0.48))
            .accessibilityHidden(true)
    }
}

/// Serializes bounded ImageIO decoding away from the UI actor. A dismissed
/// sheet's cancelled task is ignored before queued work begins, so opening
/// multiple large images never creates an unbounded decode backlog.
private actor DatasetImageDecoder {
    static let shared = DatasetImageDecoder()

    func thumbnail(for url: URL) -> NSImage? {
        image(for: url, maximumPixelSize: 360)
    }

    func preview(for url: URL) -> NSImage? {
        image(for: url, maximumPixelSize: 2_048)
    }

    private func image(for url: URL, maximumPixelSize: Int) -> NSImage? {
        guard !Task.isCancelled, let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let options = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: maximumPixelSize,
            kCGImageSourceCreateThumbnailWithTransform: true,
        ] as CFDictionary
        guard !Task.isCancelled, let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options) else { return nil }
        return NSImage(cgImage: image, size: NSSize(width: image.width, height: image.height))
    }
}
