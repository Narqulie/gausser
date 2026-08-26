import Foundation
import Observation

struct RecentScene: Codable, Identifiable, Equatable {
    let id: UUID
    let bookmark: Data
    let displayName: String
    let lastOpened: Date

    init(bookmark: Data, displayName: String, lastOpened: Date = .now) {
        self.id = UUID()
        self.bookmark = bookmark
        self.displayName = displayName
        self.lastOpened = lastOpened
    }
}

@Observable
final class RecentSceneStore {
    private static let key = "recent-scenes-v1"
    private let defaults: UserDefaults
    private(set) var scenes: [RecentScene]

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        self.scenes = Self.load(from: defaults)
    }

    func add(_ url: URL) throws {
        let bookmark = try url.bookmarkData(
            options: [.withSecurityScope],
            includingResourceValuesForKeys: nil,
            relativeTo: nil
        )
        let name = url.lastPathComponent
        scenes.removeAll { $0.displayName == name && $0.bookmark == bookmark }
        scenes.insert(RecentScene(bookmark: bookmark, displayName: name), at: 0)
        scenes = Array(scenes.prefix(10))
        persist()
    }

    func resolve(_ scene: RecentScene) throws -> URL {
        var isStale = false
        let url = try URL(
            resolvingBookmarkData: scene.bookmark,
            options: [.withSecurityScope, .withoutUI],
            relativeTo: nil,
            bookmarkDataIsStale: &isStale
        )
        if isStale {
            try add(url)
            scenes.removeAll { $0.id == scene.id }
            persist()
        }
        return url
    }

    func remove(_ scene: RecentScene) {
        scenes.removeAll { $0.id == scene.id }
        persist()
    }

    private func persist() {
        let data = try? JSONEncoder().encode(scenes)
        defaults.set(data, forKey: Self.key)
    }

    private static func load(from defaults: UserDefaults) -> [RecentScene] {
        guard
            let data = defaults.data(forKey: key),
            let scenes = try? JSONDecoder().decode([RecentScene].self, from: data)
        else {
            return []
        }
        return scenes
    }
}
