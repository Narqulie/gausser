import Foundation
import Testing
@testable import GausserStudio

@Test @MainActor func stageWorkspaceAppliesPresetsAndHonorsVisibility() throws {
    let data = Data("""
    {"version":1,"title":"COLMAP","description":"SfM","preset":"video","groups":[],"presets":[{"id":"video","title":"Video","description":"Sequential","recommended":true,"values":{"matcher":"sequential"}},{"id":"photos","title":"Photos","description":"Exhaustive","values":{"matcher":"exhaustive"}}],"fields":[{"key":"matcher","label":"Matcher","type":"enum","default":"sequential","choices":["sequential","vocab_tree"]},{"key":"vocab_tree","label":"Vocabulary tree","type":"path","default":null,"visible_when":{"key":"matcher","equals":"vocab_tree"}}]}
    """.utf8)
    let definition = try JSONDecoder().decode(StageDefinition.self, from: data)
    let model = StageWorkspaceModel(definition: definition)

    #expect(model.values["matcher"] == .string("sequential"))
    #expect(!model.isVisible(definition.fields[1]))

    model.applyPreset(try #require(definition.presets.last))
    #expect(model.values["matcher"] == .string("exhaustive"))
    #expect(model.selectedPreset == "photos")

    model.set(.string("vocab_tree"), for: "matcher")
    #expect(model.isVisible(definition.fields[1]))
}

@Test @MainActor func stageWorkspaceInvalidatesAResolvedCommandOnEdit() throws {
    let data = Data("""
    {"version":1,"title":"Stage","description":"Test","preset":"default","presets":[],"fields":[{"key":"count","label":"Count","type":"integer","default":1}]}
    """.utf8)
    let model = StageWorkspaceModel(definition: try JSONDecoder().decode(StageDefinition.self, from: data))
    model.resolution = StageResolution(stage: "stage", specificationVersion: 1, preset: "default", values: ["count": .number(1)], arguments: ["--count", "1"], command: ["stage"])

    model.set(.number(2), for: "count")

    #expect(model.resolution == nil)
}
