import Foundation
import Observation
import SwiftUI

/// Presentation state for a catalog-backed workspace. Python remains the
/// authority for validation and resolution; this model only manages drafts.
@MainActor
@Observable
final class StageWorkspaceModel {
    let definition: StageDefinition
    var values: [String: JSONValue]
    var selectedPreset: String
    var resolution: StageResolution?
    var validationError: String?
    var isResolving = false
    var showsAdvanced = false

    init(definition: StageDefinition, seededValues: [String: JSONValue] = [:]) {
        var initialValues = Dictionary(uniqueKeysWithValues: definition.fields.map { ($0.key, $0.defaultValue) })
        if let preset = definition.presets.first(where: { $0.id == definition.preset }) {
            initialValues.merge(preset.values) { _, presetValue in presetValue }
        }
        initialValues.merge(seededValues) { _, seeded in seeded }
        self.definition = definition
        self.values = initialValues
        self.selectedPreset = definition.preset
    }

    func set(_ value: JSONValue, for key: String) {
        guard values[key] != value else { return }
        values[key] = value
        resolution = nil
        validationError = nil
        selectedPreset = "custom"
    }

    func applyPreset(_ preset: StagePreset) {
        var resetValues = Dictionary(uniqueKeysWithValues: definition.fields.map { ($0.key, $0.defaultValue) })
        for field in definition.fields where field.readOnly == true || field.required == true {
            resetValues[field.key] = values[field.key] ?? field.defaultValue
        }
        resetValues.merge(preset.values) { _, presetValue in presetValue }
        values = resetValues
        selectedPreset = preset.id
        resolution = nil
        validationError = nil
    }

    func isVisible(_ field: StageField) -> Bool {
        guard let condition = field.visibleWhen else { return true }
        return values[condition.key] == condition.equals
    }

    func fields(in group: StageGroup) -> [StageField] {
        definition.fields.filter { field in
            isVisible(field) && (field.group == group.id || (field.group == nil && group.id == "primary"))
        }
    }
}

/// A dedicated, desktop-sized configuration workspace. Its content scrolls but
/// its validation and run actions are intentionally persistent, so a large
/// catalog cannot make the next action disappear below the sheet edge.
struct StageWorkspace: View {
    let definition: StageDefinition
    let seededValues: [String: JSONValue]
    let onResolve: ([String: JSONValue], @escaping (Result<StageResolution, Error>) -> Void) -> Void
    let onRun: ([String: JSONValue]) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var model: StageWorkspaceModel

    init(
        definition: StageDefinition,
        seededValues: [String: JSONValue] = [:],
        onResolve: @escaping ([String: JSONValue], @escaping (Result<StageResolution, Error>) -> Void) -> Void,
        onRun: @escaping ([String: JSONValue]) -> Void
    ) {
        self.definition = definition
        self.seededValues = seededValues
        self.onResolve = onResolve
        self.onRun = onRun
        _model = State(initialValue: StageWorkspaceModel(definition: definition, seededValues: seededValues))
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        workspaceHero
                        presetCard
                        fieldGroups
                        notices
                        outputContext
                        commandPreview
                    }
                    .frame(maxWidth: 860, alignment: .leading)
                    .padding(28)
                    .frame(maxWidth: .infinity, alignment: .center)
                }
                .scrollBounceBehavior(.basedOnSize)
                .disabled(model.isResolving)

                Divider()
                actionBar
            }
            .navigationTitle(definition.title)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
            }
        }
        .frame(minWidth: 720, idealWidth: 860, maxWidth: 1_050, minHeight: 620, idealHeight: 760)
    }

    private var workspaceHero: some View {
        HStack(alignment: .top, spacing: 16) {
            Image(systemName: studioStageSymbol(definitionStage))
                .font(.title2)
                .foregroundStyle(.tint)
                .frame(width: 48, height: 48)
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            VStack(alignment: .leading, spacing: 6) {
                Text(definition.title)
                    .font(.title2.weight(.semibold))
                Text(definition.description)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Text("Engine-owned settings · specification v\(definition.version)")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    @ViewBuilder
    private var presetCard: some View {
        if !definition.presets.isEmpty {
            StudioContentCard("Configuration", systemImage: "slider.horizontal.3") {
                Picker("Preset", selection: presetBinding) {
                    ForEach(definition.presets, id: \.id) { preset in
                        Text(preset.recommended == true ? "\(preset.title) (Recommended)" : preset.title)
                            .tag(preset.id)
                    }
                    if model.selectedPreset == "custom" {
                        Text("Custom").tag("custom")
                    }
                }
                .pickerStyle(.menu)
                if let preset = definition.presets.first(where: { $0.id == model.selectedPreset }) {
                    Text(preset.description)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var fieldGroups: some View {
        VStack(alignment: .leading, spacing: 20) {
            ForEach(primaryGroups) { group in
                if !model.fields(in: group).isEmpty {
                    settingsGroup(group)
                }
            }
            if !advancedGroups.isEmpty {
                StudioContentCard("Advanced settings", systemImage: "gearshape.2") {
                    DisclosureGroup("Show advanced settings", isExpanded: $model.showsAdvanced) {
                        VStack(alignment: .leading, spacing: 18) {
                            ForEach(advancedGroups) { group in
                                if let description = group.description {
                                    Text(description)
                                        .font(.callout)
                                        .foregroundStyle(.secondary)
                                }
                                ForEach(model.fields(in: group)) { field in
                                    fieldControl(field)
                                }
                            }
                        }
                        .padding(.top, 10)
                    }
                }
            }
        }
    }

    private func settingsGroup(_ group: StageGroup) -> some View {
        StudioContentCard(group.title, systemImage: "switch.2") {
            if let description = group.description {
                Text(description)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            VStack(alignment: .leading, spacing: 18) {
                ForEach(model.fields(in: group)) { field in
                    fieldControl(field)
                }
            }
        }
    }

    @ViewBuilder
    private var notices: some View {
        if let warnings = definition.warnings, !warnings.isEmpty {
            StudioContentCard("Run notes", systemImage: "exclamationmark.bubble") {
                ForEach(warnings) { warning in
                    Label(warning.message, systemImage: warning.level == "error" ? "exclamationmark.triangle.fill" : "exclamationmark.circle.fill")
                        .foregroundStyle(warning.level == "error" ? .red : .orange)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        if let error = model.validationError {
            StudioContentCard("Validation issue", systemImage: "exclamationmark.triangle") {
                Text(error)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }
        }
    }

    @ViewBuilder
    private var outputContext: some View {
        if let output = definition.outputContext {
            StudioContentCard(output.title, systemImage: "folder") {
                Text(output.path)
                    .font(.system(.callout, design: .monospaced))
                    .textSelection(.enabled)
                if let description = output.description {
                    Text(description)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    @ViewBuilder
    private var commandPreview: some View {
        if let resolution = model.resolution {
            StudioContentCard("Resolved command", systemImage: "terminal") {
                Text("The engine validated these exact options. Running records this resolved snapshot in the scene manifest.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Text(resolution.command.joined(separator: "\n"))
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            }
        }
    }

    private var actionBar: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                if model.isResolving {
                    Label("Validating settings…", systemImage: "clock")
                        .foregroundStyle(.secondary)
                } else if model.resolution != nil {
                    Label("Settings validated", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                } else {
                    Text("Validate before running")
                        .foregroundStyle(.secondary)
                }
            }
            .font(.callout)
            Spacer()
            StudioGlassActions {
                Button("Validate settings") { resolve() }
                    .disabled(model.isResolving)
                    .studioGlassButton()
                Button("Run \(definition.title)") {
                    onRun(model.values)
                    dismiss()
                }
                .disabled(model.resolution == nil || model.isResolving)
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(.horizontal, 28)
        .padding(.vertical, 16)
        .background(.bar)
    }

    private var definitionStage: String {
        definition.title.lowercased().contains("colmap") ? "colmap" :
        definition.title.lowercased().contains("sharp") ? "sharpframes" :
        definition.title.lowercased().contains("sam") ? "sammer" :
        definition.title.lowercased().contains("msplat") ? "msplat" :
        definition.title.lowercased().contains("lod") ? "lod" : "brush"
    }

    private var allGroups: [StageGroup] {
        definition.groups ?? [StageGroup(id: "primary", title: "Settings", description: nil, advanced: false)]
    }

    private var primaryGroups: [StageGroup] { allGroups.filter { $0.advanced != true } }
    private var advancedGroups: [StageGroup] { allGroups.filter { $0.advanced == true } }

    private var presetBinding: Binding<String> {
        Binding(
            get: { model.selectedPreset },
            set: { id in
                guard let preset = definition.presets.first(where: { $0.id == id }) else { return }
                model.applyPreset(preset)
            }
        )
    }

    @ViewBuilder
    private func fieldControl(_ field: StageField) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            switch field.type {
            case "enum":
                Picker(field.label, selection: stringBinding(field.key)) {
                    ForEach(field.choices ?? [], id: \.self) { choice in Text(choice).tag(choice) }
                }
                .pickerStyle(.menu)
            case "boolean":
                Toggle(field.label, isOn: booleanBinding(field.key))
            case "integer", "decimal", "text", "path":
                if field.readOnly == true {
                    LabeledContent(field.label, value: stringValue(field.key))
                } else {
                    TextField(field.label, text: textBinding(field), prompt: field.placeholder.map { Text($0) })
                        .textFieldStyle(.roundedBorder)
                }
            default:
                LabeledContent(field.label, value: stringValue(field.key))
            }
            if let help = field.help {
                Text(help)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func stringValue(_ key: String) -> String {
        guard let value = model.values[key] else { return "" }
        switch value {
        case .string(let string): return string
        case .number(let number): return number.rounded() == number ? String(Int(number)) : String(number)
        case .bool(let bool): return bool.description
        case .null: return ""
        default: return ""
        }
    }

    private func textBinding(_ field: StageField) -> Binding<String> {
        Binding(get: { stringValue(field.key) }, set: { text in
            if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, field.defaultValue == .null {
                model.set(.null, for: field.key)
                return
            }
            switch field.type {
            case "integer": model.set(Int(text).map { .number(Double($0)) } ?? .string(text), for: field.key)
            case "decimal": model.set(Double(text).map(JSONValue.number) ?? .string(text), for: field.key)
            default: model.set(.string(text), for: field.key)
            }
        })
    }

    private func stringBinding(_ key: String) -> Binding<String> {
        Binding(get: { stringValue(key) }, set: { model.set(.string($0), for: key) })
    }

    private func booleanBinding(_ key: String) -> Binding<Bool> {
        Binding(get: {
            if case .bool(let value) = model.values[key] { return value }
            return false
        }, set: { model.set(.bool($0), for: key) })
    }

    private func resolve() {
        let draft = model.values
        model.isResolving = true
        model.resolution = nil
        model.validationError = nil
        onResolve(draft) { result in
            guard model.values == draft else { return }
            model.isResolving = false
            switch result {
            case .success(let resolution): model.resolution = resolution
            case .failure(let error): model.validationError = error.localizedDescription
            }
        }
    }
}
