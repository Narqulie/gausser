import SwiftUI

/// Complete native form for the engine-owned Sharp Frames v1 specification.
struct SharpFramesWorkspace: View {
    let definition: StageDefinition
    let source: URL
    let onResolve: ([String: JSONValue], @escaping (Result<StageResolution, Error>) -> Void) -> Void
    let onRun: ([String: JSONValue]) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var values: [String: JSONValue]
    @State private var showsAdvanced = false
    @State private var resolution: StageResolution?
    @State private var validationError: String?
    @State private var isResolving = false

    init(
        definition: StageDefinition,
        source: URL,
        onResolve: @escaping ([String: JSONValue], @escaping (Result<StageResolution, Error>) -> Void) -> Void,
        onRun: @escaping ([String: JSONValue]) -> Void
    ) {
        self.definition = definition
        self.source = source
        self.onResolve = onResolve
        self.onRun = onRun
        var initial = Dictionary(uniqueKeysWithValues: definition.fields.map { ($0.key, $0.defaultValue) })
        initial["source"] = .string(source.path)
        _values = State(initialValue: initial)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text(definition.title)
                .font(.title2.bold())
            Text(definition.description)
                .foregroundStyle(.secondary)

            GroupBox("Source") {
                Text(source.path)
                    .font(.system(.subheadline, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.top, 2)
            }

            GroupBox("Preset · \(definition.preset)") {
                Text(definition.presets.first(where: { $0.id == definition.preset })?.description ?? "Engine-owned defaults.")
                    .foregroundStyle(.secondary)
                    .padding(.top, 2)
            }

            Form {
                ForEach(primaryFields) { field in
                    fieldControl(field)
                }
                if !advancedFields.isEmpty {
                    Section {
                        DisclosureGroup("Advanced", isExpanded: $showsAdvanced) {
                            ForEach(advancedFields) { field in
                                fieldControl(field)
                            }
                        }
                    }
                }
            }
            .formStyle(.grouped)
            .disabled(isResolving)
            .onChange(of: values) { _, _ in
                resolution = nil
                validationError = nil
            }

            if let validationError {
                Text(validationError)
                    .foregroundStyle(.red)
            }
            if let resolution {
                GroupBox("Resolved command · \(resolution.preset)") {
                    Text(resolution.command.joined(separator: "\n"))
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.top, 2)
                }
            }

            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button(isResolving ? "Validating…" : "Validate settings") { resolve() }
                    .disabled(isResolving)
                if resolution != nil {
                    Button("Confirm & Run") {
                        onRun(values)
                        dismiss()
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
        }
        .padding(24)
        .frame(minWidth: 560, minHeight: 600)
    }

    private var primaryFields: [StageField] {
        definition.fields.filter { $0.key != "source" && $0.advanced != true }
    }

    private var advancedFields: [StageField] {
        definition.fields.filter { $0.advanced == true }
    }

    @ViewBuilder
    private func fieldControl(_ field: StageField) -> some View {
        switch field.type {
        case "integer":
            LabeledContent(field.label) {
                TextField(field.help ?? field.label, text: integerText(for: field.key))
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 120)
            }
        case "enum":
            Picker(field.label, selection: stringValue(for: field.key)) {
                ForEach(field.choices ?? [], id: \.self) { choice in
                    Text(choice).tag(choice)
                }
            }
        default:
            EmptyView()
        }
    }

    private func resolve() {
        isResolving = true
        validationError = nil
        resolution = nil
        onResolve(values) { result in
            isResolving = false
            switch result {
            case .success(let resolution):
                self.resolution = resolution
            case .failure(let error):
                validationError = error.localizedDescription
            }
        }
    }

    private func integerText(for key: String) -> Binding<String> {
        Binding(
            get: {
                guard case .number(let value) = values[key] else { return "" }
                return String(Int(value))
            },
            set: { values[key] = Int($0).map { .number(Double($0)) } ?? .string($0) }
        )
    }

    private func stringValue(for key: String) -> Binding<String> {
        Binding(
            get: {
                guard case .string(let value) = values[key] else { return "" }
                return value
            },
            set: { values[key] = .string($0) }
        )
    }
}
