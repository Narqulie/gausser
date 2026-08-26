import Foundation
import Metal
import MetalKit
import Observation
import SplatIO
import SwiftUI
import simd
import MetalSplatter

/// The lifecycle of one explicitly selected splat artifact in the native viewer.
///
/// This deliberately supports only the formats accepted by MetalSplatter's
/// `AutodetectSceneReader`: PLY, SPZ, and legacy `.splat`. It does not represent
/// SOG files or PlayCanvas streamed LOD trees.
public enum SplatViewerLoadState: Equatable {
    case idle
    case loading(URL)
    case loaded(URL, splatCount: Int)
    case failed(URL?, message: String)

    public var errorMessage: String? {
        guard case let .failed(_, message) = self else { return nil }
        return message
    }
}

/// SwiftUI-facing state for an explicitly selected, scene-owned splat artifact.
///
/// Call `load(_:)` with the artifact URL selected by the scene UI. The
/// accompanying `SplatViewerHost` performs the MetalSplatter load and records
/// its result here; this model does not discover artifacts or infer a scene
/// layout.
@MainActor @Observable
public final class SplatViewerModel {
    public private(set) var state: SplatViewerLoadState = .idle

    private var requestID = 0

    public init() {}

    /// Starts loading a local PLY, SPZ, or `.splat` file in a hosted viewer.
    public func load(_ artifactURL: URL) {
        do {
            let url = try Self.validatedArtifactURL(artifactURL)
            requestID += 1
            state = .loading(url)
        } catch {
            state = .failed(artifactURL, message: error.localizedDescription)
        }
    }

    public func clear() {
        requestID += 1
        state = .idle
    }

    fileprivate var pendingRequest: SplatViewerLoadRequest? {
        guard case let .loading(url) = state else { return nil }
        return SplatViewerLoadRequest(id: requestID, url: url)
    }

    /// Retains a successfully rendered request while SwiftUI refreshes after the
    /// state transition from `loading` to `loaded`.
    fileprivate var rendererRequest: SplatViewerLoadRequest? {
        switch state {
        case let .loading(url), let .loaded(url, _):
            return SplatViewerLoadRequest(id: requestID, url: url)
        case .idle, .failed:
            return nil
        }
    }

    fileprivate func finish(_ request: SplatViewerLoadRequest, splatCount: Int) {
        guard request == pendingRequest else { return }
        state = .loaded(request.url, splatCount: splatCount)
    }

    fileprivate func fail(_ request: SplatViewerLoadRequest, error: Error) {
        guard request == pendingRequest else { return }
        state = .failed(request.url, message: error.localizedDescription)
    }

    private static func validatedArtifactURL(_ artifactURL: URL) throws -> URL {
        guard artifactURL.isFileURL else {
            throw SplatViewerValidationError.notLocalFile
        }

        let url = artifactURL.standardizedFileURL
        guard SplatFileFormat(for: url) != nil else {
            throw SplatViewerValidationError.unsupportedFormat(url.pathExtension)
        }

        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), !isDirectory.boolValue else {
            throw SplatViewerValidationError.missingRegularFile
        }
        return url
    }
}

private struct SplatViewerLoadRequest: Equatable {
    let id: Int
    let url: URL
}

private enum SplatViewerValidationError: LocalizedError {
    case notLocalFile
    case unsupportedFormat(String)
    case missingRegularFile

    var errorDescription: String? {
        switch self {
        case .notLocalFile:
            return "The native viewer can load only local files."
        case let .unsupportedFormat(extensionName):
            let suffix = extensionName.isEmpty ? "this file" : ".\(extensionName)"
            return "MetalSplatter supports PLY, SPZ, and .splat artifacts; \(suffix) is not supported."
        case .missingRegularFile:
            return "The selected splat artifact is missing or is a directory."
        }
    }
}

/// A reusable MetalKit host for `SplatViewerModel`.
///
/// The host uses MetalSplatter 2b965de's public loading path:
/// `AutodetectSceneReader(url).readAll()`, then `SplatChunk(device:from:)`, then
/// `SplatRenderer.addChunk(_)`. This is intentionally a single-file proof of
/// integration, not a streamed SOG/LOD renderer.
public struct SplatViewerHost: NSViewRepresentable {
    @Bindable public var model: SplatViewerModel

    public init(model: SplatViewerModel) {
        _model = Bindable(wrappedValue: model)
    }

    public func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    public func makeNSView(context: Context) -> MTKView {
        context.coordinator.makeView()
    }

    public func updateNSView(_ view: MTKView, context: Context) {
        context.coordinator.loadIfNeeded(model.rendererRequest, into: model)
    }

    public static func dismantleNSView(_ view: MTKView, coordinator: Coordinator) {
        coordinator.cancelLoad()
        view.delegate = nil
    }

    @MainActor
    public final class Coordinator: NSObject, MTKViewDelegate {
        private let maxSimultaneousRenders = 3
        private var metalView: MTKView?
        private var commandQueue: MTLCommandQueue?
        private var renderer: SplatRenderer?
        private var camera = SplatViewerCamera.default
        private var activeRequest: SplatViewerLoadRequest?
        private var loadingTask: Task<Void, Never>?

        fileprivate func makeView() -> MTKView {
            let view = MTKView(frame: .zero, device: MTLCreateSystemDefaultDevice())
            view.colorPixelFormat = .bgra8Unorm_srgb
            view.depthStencilPixelFormat = .depth32Float
            view.sampleCount = 1
            view.clearColor = MTLClearColor(red: 0.06, green: 0.06, blue: 0.07, alpha: 1)
            view.enableSetNeedsDisplay = false
            view.isPaused = true
            view.delegate = self
            metalView = view
            commandQueue = view.device?.makeCommandQueue()
            return view
        }

        fileprivate func loadIfNeeded(_ request: SplatViewerLoadRequest?, into model: SplatViewerModel) {
            guard request != activeRequest else { return }
            cancelLoad()
            activeRequest = request
            renderer = nil
            metalView?.isPaused = true

            guard let request else {
                metalView?.draw()
                return
            }
#if arch(x86_64)
            // MetalSplatter's current SplatRenderer initializer terminates on Intel.
            model.fail(request, error: SplatViewerHostError.unsupportedArchitecture)
            return
#endif
            guard let view = metalView, let device = view.device, commandQueue != nil else {
                model.fail(request, error: SplatViewerHostError.metalUnavailable)
                return
            }

            loadingTask = Task { @MainActor [weak self, weak model] in
                do {
                    let reader = try AutodetectSceneReader(request.url)
                    let points = try await reader.readAll()
                    try Task.checkCancellation()
                    guard !points.isEmpty else {
                        throw SplatViewerHostError.emptyScene
                    }

                    let renderer = try SplatRenderer(
                        device: device,
                        colorFormat: view.colorPixelFormat,
                        depthFormat: view.depthStencilPixelFormat,
                        sampleCount: view.sampleCount,
                        maxViewCount: 1,
                        maxSimultaneousRenders: self?.maxSimultaneousRenders ?? 3
                    )
                    let chunk = try SplatChunk(device: device, from: points)
                    _ = await renderer.addChunk(chunk)
                    try Task.checkCancellation()

                    guard let self, self.activeRequest == request else { return }
                    self.renderer = renderer
                    self.camera = SplatViewerCamera(points: points)
                    model?.finish(request, splatCount: points.count)
                    self.metalView?.isPaused = false
                } catch is CancellationError {
                    // A newer explicit selection superseded this request.
                } catch {
                    guard let self, self.activeRequest == request else { return }
                    model?.fail(request, error: error)
                }
            }
        }

        fileprivate func cancelLoad() {
            loadingTask?.cancel()
            loadingTask = nil
            activeRequest = nil
        }

        public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            view.draw()
        }

        public func draw(in view: MTKView) {
            guard
                let renderer,
                renderer.isReadyToRender,
                let drawable = view.currentDrawable,
                let commandQueue,
                view.drawableSize.width > 0,
                view.drawableSize.height > 0
            else {
                return
            }

            guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
            let viewport = SplatRenderer.ViewportDescriptor(
                viewport: MTLViewport(
                    originX: 0,
                    originY: 0,
                    width: view.drawableSize.width,
                    height: view.drawableSize.height,
                    znear: 0,
                    zfar: 1
                ),
                projectionMatrix: camera.projection(aspectRatio: Float(view.drawableSize.width / view.drawableSize.height)),
                viewMatrix: camera.viewMatrix,
                screenSize: SIMD2(Int(view.drawableSize.width), Int(view.drawableSize.height))
            )

            do {
                let didRender = try renderer.render(
                    viewports: [viewport],
                    colorTexture: view.multisampleColorTexture ?? drawable.texture,
                    colorStoreAction: view.multisampleColorTexture == nil ? .store : .multisampleResolve,
                    depthTexture: view.depthStencilTexture,
                    rasterizationRateMap: nil,
                    renderTargetArrayLength: 0,
                    to: commandBuffer
                )
                if didRender {
                    commandBuffer.present(drawable)
                }
                commandBuffer.commit()
            } catch {
                // Rendering failures occur after load; retain the loaded artifact state so a
                // caller can retry it rather than reporting a false parsing failure.
                commandBuffer.commit()
            }
        }
    }
}

private enum SplatViewerHostError: LocalizedError {
    case metalUnavailable
    case unsupportedArchitecture
    case emptyScene

    var errorDescription: String? {
        switch self {
        case .metalUnavailable:
            "Metal is unavailable on this Mac."
        case .unsupportedArchitecture:
            "MetalSplatter's current renderer does not support Intel Macs."
        case .emptyScene:
            "The selected splat artifact contains no splats."
        }
    }
}

private struct SplatViewerCamera {
    let center: SIMD3<Float>
    let distance: Float
    let nearZ: Float
    let farZ: Float

    static let `default` = SplatViewerCamera(
        center: .zero,
        distance: 3,
        nearZ: 0.01,
        farZ: 100
    )

    private init(center: SIMD3<Float>, distance: Float, nearZ: Float, farZ: Float) {
        self.center = center
        self.distance = distance
        self.nearZ = nearZ
        self.farZ = farZ
    }

    init(points: [SplatPoint]) {
        var minimum = points[0].position
        var maximum = minimum
        for point in points.dropFirst() {
            minimum = simd_min(minimum, point.position)
            maximum = simd_max(maximum, point.position)
        }

        center = (minimum + maximum) / 2
        let radius = max(simd_length(maximum - minimum) / 2, 0.1)
        distance = radius / tan(.pi / 6) * 1.25
        nearZ = max(distance - radius * 2, 0.01)
        farZ = distance + radius * 4
    }

    var viewMatrix: simd_float4x4 {
        simd_float4x4(columns: (
            SIMD4(1, 0, 0, 0),
            SIMD4(0, 1, 0, 0),
            SIMD4(0, 0, 1, 0),
            SIMD4(-center.x, -center.y, -center.z - distance, 1)
        ))
    }

    func projection(aspectRatio: Float) -> simd_float4x4 {
        let yScale = 1 / tan(Float.pi / 6)
        let xScale = yScale / aspectRatio
        let zScale = farZ / (nearZ - farZ)
        return simd_float4x4(columns: (
            SIMD4(xScale, 0, 0, 0),
            SIMD4(0, yScale, 0, 0),
            SIMD4(0, 0, zScale, -1),
            SIMD4(0, 0, zScale * nearZ, 0)
        ))
    }
}
