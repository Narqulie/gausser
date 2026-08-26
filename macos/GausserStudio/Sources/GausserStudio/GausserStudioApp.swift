import SwiftUI

@main
struct GausserStudioApp: App {
    var body: some Scene {
        WindowGroup("Gausser Studio") {
            StudioView()
                .frame(minWidth: 920, minHeight: 620)
        }
        .defaultSize(width: 1180, height: 760)
    }
}
