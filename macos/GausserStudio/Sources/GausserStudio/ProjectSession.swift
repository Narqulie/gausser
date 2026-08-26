import Foundation
import Observation

/// Holds the one currently-open scene and balances its security-scoped access.
@Observable
final class ProjectSession {
    private(set) var scene: URL?
    private var scopedURL: URL?

    func select(_ url: URL) {
        relinquishAccess()
        scopedURL = url
        _ = url.startAccessingSecurityScopedResource()
        scene = url
    }

    func close() {
        relinquishAccess()
        scene = nil
    }

    private func relinquishAccess() {
        scopedURL?.stopAccessingSecurityScopedResource()
        scopedURL = nil
    }

    deinit {
        relinquishAccess()
    }
}
