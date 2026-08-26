import AppKit
import ImageIO
import SwiftUI

/// A lazy, read-only gallery of the exact images published in ``images/``.
/// Image paths come from the engine; the native client never scans or mutates
/// the dataset independently.
struct DatasetImageBrowser: View {
    let scene: URL
    let images: [DatasetImageSummary]
    let revision: String
    let imageCount: Int
    let hasMore: Bool
    let isLoading: Bool
    let onLoadMore: () -> Void

    @State private var selectedImage: DatasetImageSummary?

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
                    Text("\(imageCount) canonical images · \(images.count) shown")
                        .font(.callout)
                        .foregroundStyle(.secondary)

                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 142, maximum: 190), spacing: 12)], spacing: 12) {
                        ForEach(images) { image in
                            Button {
                                selectedImage = image
                            } label: {
                                DatasetThumbnail(url: imageURL(for: image), name: image.name, revision: revision)
                                    // A transactional republish can replace a filename.
                                    .id(DatasetImageCacheKey(path: image.relativePath, revision: revision))
                            }
                            .buttonStyle(.plain)
                            .accessibilityLabel("Preview \(image.name)")
                        }
                    }

                    if hasMore {
                        Button("Show more images", systemImage: "plus") { onLoadMore() }
                            .disabled(isLoading)
                            .buttonStyle(.bordered)
                    } else if isLoading {
                        ProgressView("Loading images…")
                    }
                }
            }
        }
        .sheet(item: $selectedImage) { image in
            DatasetImageDetail(image: image, url: imageURL(for: image), revision: revision)
        }
    }

    private func imageURL(for image: DatasetImageSummary) -> URL {
        scene.appending(path: image.relativePath)
    }
}

private struct DatasetImageCacheKey: Hashable {
    let path: String
    let revision: String
}

private struct DatasetThumbnail: View {
    let url: URL
    let name: String
    let revision: String
    @State private var thumbnail: NSImage?

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            Group {
                if let thumbnail {
                    Image(nsImage: thumbnail)
                        .resizable()
                        .scaledToFill()
                } else {
                    ContentUnavailableView("", systemImage: "photo")
                        .symbolRenderingMode(.hierarchical)
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
            thumbnail = await DatasetImageDecoder.shared.thumbnail(for: url)
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
    let revision: String
    @Environment(\.dismiss) private var dismiss
    @State private var previewState = PreviewState.loading
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
            let preview = await DatasetImageDecoder.shared.preview(for: url)
            guard !Task.isCancelled else { return }
            previewState = preview.map(PreviewState.loaded) ?? .failed
        }
    }

    @ViewBuilder
    private var preview: some View {
        switch previewState {
        case .loading:
            ProgressView("Loading preview…")
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        case .loaded(let image):
            Image(nsImage: image)
                .resizable()
                .scaledToFit()
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
