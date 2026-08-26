import Foundation
import Testing
@testable import GausserStudio

@Test @MainActor
func sharpFramesMultipleVideoSelectionUsesTheirCommonDirectory() throws {
    let source = try SourcePicker.commonSharpFramesSource(for: [
        URL(fileURLWithPath: "/tmp/clips/one.mov"),
        URL(fileURLWithPath: "/tmp/clips/two.mp4"),
    ])

    #expect(source.path == "/tmp/clips")
}

@Test @MainActor
func sharpFramesRejectsMultipleVideoDirectories() {
    #expect(throws: SourcePickerError.separateDirectories) {
        try SourcePicker.commonSharpFramesSource(for: [
            URL(fileURLWithPath: "/tmp/a/one.mov"),
            URL(fileURLWithPath: "/tmp/b/two.mov"),
        ])
    }
}
