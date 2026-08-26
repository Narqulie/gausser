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

/// Generic native form for a versioned, Python-owned stage definition.
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
            Form {
                Section {
                    Text(definition.description)
                        .foregroundStyle(.secondary)
                    if !definition.presets.isEmpty {
                        Picker("Configuration", selection: presetBinding) {
                            ForEach(definition.presets, id: \.id) { preset in
                                Text(preset.recommended == true ? "\(preset.title) (Recommended)" : preset.title)
                                    .tag(preset.id)
                            }
                            if model.selectedPreset == "custom" {
                                Text("Custom").tag("custom")
                            }
                        }
                        if let preset = definition.presets.first(where: { $0.id == model.selectedPreset }) {
                            Text(preset.description).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }

                ForEach(primaryGroups) { group in
                    if !model.fields(in: group).isEmpty {
                        Section {
                            ForEach(model.fields(in: group)) { field in
                                fieldControl(field)
                            }
                        } header: {
                            Text(group.title)
                        } footer: {
                            if let description = group.description { Text(description) }
                        }
                    }
                }

                if !advancedGroups.isEmpty {
                    Section {
                        DisclosureGroup("Advanced settings", isExpanded: $model.showsAdvanced) {
                            ForEach(advancedGroups) { group in
                                if let description = group.description { Text(description).foregroundStyle(.secondary) }
                                ForEach(model.fields(in: group)) { field in
                                    fieldControl(field)
                                }
                            }
                        }
                    }
                }

                if let warnings = definition.warnings {
                    ForEach(warnings) { warning in
                        Section {
                            Label(warning.message, systemImage: warning.level == "error" ? "exclamationmark.triangle.fill" : "exclamationmark.circle")
                                .foregroundStyle(warning.level == "error" ? .red : .orange)
                        }
                    }
                }

                if let error = model.validationError {
                    Section { Text(error).foregroundStyle(.red) }
                }

                if let resolution = model.resolution {
                    Section("Resolved command") {
                        Text(resolution.command.joined(separator: "\n"))
                            .font(.system(.caption, design: .monospaced))
                            .textSelection(.enabled)
                    }
                }
            }
            .disabled(model.isResolving)
            .safeAreaInset(edge: .bottom, spacing: 0) {
                workspaceActions.disabled(model.isResolving)
            }
            .navigationTitle(definition.title)
        }
        .frame(minWidth: 560, idealWidth: 680, minHeight: 520, idealHeight: 760)
    }

    private var workspaceActions: some View {
        HStack {
            Button("Cancel") { dismiss() }
            Spacer()
            if model.resolution != nil {
                Button("Confirm & Run", role: .destructive) {
                    onRun(model.values)
                    dismiss()
                }
            } else {
                Button(model.isResolving ? "Validating…" : "Validate & Continue") {
                    resolve()
                }
                .disabled(model.isResolving)
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding()
        .background(.regularMaterial)
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
        switch field.type {
        case "enum":
            Picker(field.label, selection: stringBinding(field.key)) {
                ForEach(field.choices ?? [], id: \.self) { choice in Text(choice).tag(choice) }
            }
        case "boolean":
            Toggle(field.label, isOn: booleanBinding(field.key))
        case "integer", "decimal", "text", "path":
            if field.readOnly == true {
                LabeledContent(field.label, value: stringValue(field.key))
            } else {
                TextField(field.label, text: textBinding(field))
                    .textFieldStyle(.roundedBorder)
            }
        default:
            LabeledContent(field.label, value: stringValue(field.key))
        }
        if let help = field.help { Text(help).font(.caption).foregroundStyle(.secondary) }
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
