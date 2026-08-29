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

    model.set(.string("/tmp/tree.bin"), for: "vocab_tree")
    model.applyPreset(try #require(definition.presets.first))
    #expect(model.values["matcher"] == .string("sequential"))
    #expect(model.values["vocab_tree"] == .null)
}

@Test @MainActor func stageWorkspaceLocksConfirmedReplacementAcrossPresetsAndEdits() throws {
    let data = Data("""
    {"version":1,"title":"SAM3","description":"Mask","preset":"standard","groups":[],"presets":[{"id":"standard","title":"Standard","description":"Base","values":{"force":false}},{"id":"other","title":"Other","description":"Alternate","values":{"force":false}}],"fields":[{"key":"force","label":"Replace masks","type":"boolean","default":false}]}
    """.utf8)
    let definition = try JSONDecoder().decode(StageDefinition.self, from: data)
    let model = StageWorkspaceModel(definition: definition)
    #expect(model.values["force"] == .bool(false))

    model.updateLockedValues(["force": .bool(true)])

    #expect(model.values["force"] == .bool(true))
    #expect(model.isLocked("force"))

    model.set(.bool(false), for: "force")
    model.applyPreset(try #require(definition.presets.last))

    #expect(model.values["force"] == .bool(true))
}

@Test @MainActor func stageWorkspacePreservesRequiredSubjectWhenChangingPreset() throws {
    let data = Data("""
    {"version":1,"title":"SAM3","description":"Mask","preset":"standard","groups":[],"presets":[{"id":"standard","title":"Standard","description":"Base","values":{"threshold":0.5}},{"id":"coverage","title":"Coverage","description":"Lower","values":{"threshold":0.3}}],"fields":[{"key":"prompt","label":"Prompt","type":"text","default":"","required":true},{"key":"threshold","label":"Threshold","type":"decimal","default":0.5}]}
    """.utf8)
    let definition = try JSONDecoder().decode(StageDefinition.self, from: data)
    let model = StageWorkspaceModel(definition: definition)
    model.set(.string("fountain"), for: "prompt")

    model.applyPreset(try #require(definition.presets.last))

    #expect(model.values["prompt"] == .string("fountain"))
    #expect(model.values["threshold"] == .number(0.3))
}

@Test @MainActor func stageWorkspaceSnapshotsEveryDraftEditImmediately() throws {
    let data = Data("""
    {"version":1,"title":"Stage","description":"Test","preset":"default","presets":[{"id":"default","title":"Default","description":"Base","values":{"count":1}},{"id":"other","title":"Other","description":"Changed","values":{"count":2}}],"fields":[{"key":"count","label":"Count","type":"integer","default":1}]}
    """.utf8)
    var snapshots: [[String: JSONValue]] = []
    let model = StageWorkspaceModel(
        definition: try JSONDecoder().decode(StageDefinition.self, from: data),
        onDraftChange: { snapshots.append($0) }
    )

    model.set(.number(3), for: "count")
    model.applyPreset(try #require(model.definition.presets.last))

    #expect(snapshots == [["count": .number(3)], ["count": .number(2)]])
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
