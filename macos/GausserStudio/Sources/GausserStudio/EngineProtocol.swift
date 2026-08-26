import Foundation

/// The recursive JSON value used by Gausser's deliberately dependency-free local protocol.
indirect enum JSONValue: Codable, Equatable, Sendable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case array([JSONValue])
    case object([String: JSONValue])
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
        } else {
            throw DecodingError.typeMismatch(
                JSONValue.self,
                .init(codingPath: decoder.codingPath, debugDescription: "Unsupported JSON value")
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }
}

extension JSONValue {
    var displayText: String {
        switch self {
        case .string(let value): value
        case .number(let value): value.formatted(.number.precision(.significantDigits(4)))
        case .bool(let value): value ? "yes" : "no"
        case .array(let values): "\(values.count) values"
        case .object(let values): "\(values.count) values"
        case .null: "–"
        }
    }
}

struct EngineRequest: Codable, Equatable, Sendable {
    let id: String
    let method: String
    let params: [String: JSONValue]
}

struct EngineRun: Codable, Equatable, Sendable {
    let id: String
    let scene: String
    let stage: String
    let state: String?
    let elapsedSeconds: Double?
    let lastObservedElapsedSeconds: Double?
    let phase: String?

    init(id: String, scene: String, stage: String, state: String?, elapsedSeconds: Double? = nil, lastObservedElapsedSeconds: Double? = nil, phase: String? = nil) {
        self.id = id
        self.scene = scene
        self.stage = stage
        self.state = state
        self.elapsedSeconds = elapsedSeconds
        self.lastObservedElapsedSeconds = lastObservedElapsedSeconds
        self.phase = phase
    }

    enum CodingKeys: String, CodingKey {
        case id, scene, stage, state, phase
        case elapsedSeconds = "elapsed_seconds"
        case lastObservedElapsedSeconds = "last_observed_elapsed_seconds"
    }
}

struct StageCatalogResponse: Decodable, Equatable, Sendable {
    let stages: [String: StageDefinition]
}

struct StageDefinition: Decodable, Equatable, Sendable {
    let version: Int
    let title: String
    let description: String
    let preset: String
    let groups: [StageGroup]?
    let presets: [StagePreset]
    let fields: [StageField]
    let warnings: [StageNotice]?
    let outputContext: StageOutputContext?

    enum CodingKeys: String, CodingKey {
        case version, title, description, preset, groups, presets, fields, warnings
        case outputContext = "output_context"
    }
}

struct StageGroup: Decodable, Equatable, Sendable, Identifiable {
    let id: String
    let title: String
    let description: String?
    let advanced: Bool?
}

struct StageNotice: Decodable, Equatable, Sendable, Identifiable {
    let id: String
    let level: String
    let message: String
}

struct StageOutputContext: Decodable, Equatable, Sendable {
    let title: String
    let path: String
    let description: String?
}

struct StageCondition: Decodable, Equatable, Sendable {
    let key: String
    let equals: JSONValue
}

struct StageResolution: Decodable, Equatable, Sendable {
    let stage: String
    let specificationVersion: Int
    let preset: String
    let values: [String: JSONValue]
    let arguments: [String]
    let command: [String]

    enum CodingKeys: String, CodingKey {
        case stage, preset, values, arguments, command
        case specificationVersion = "specification_version"
    }
}

struct StagePreset: Decodable, Equatable, Sendable {
    let id: String
    let title: String
    let description: String
    let recommended: Bool?
    let warnings: [String]?
    let values: [String: JSONValue]
}

struct StageField: Decodable, Equatable, Sendable, Identifiable {
    let key: String
    let label: String
    let type: String
    let defaultValue: JSONValue
    let required: Bool?
    let minimum: Int?
    let maximum: Int?
    let choices: [String]?
    let help: String?
    let advanced: Bool?
    let group: String?
    let visibleWhen: StageCondition?
    let requiredWhen: StageCondition?
    let placeholder: String?
    let unit: String?
    let readOnly: Bool?

    var id: String { key }

    enum CodingKeys: String, CodingKey {
        case key, label, type, required, minimum, maximum, choices, help, advanced, group, placeholder, unit
        case defaultValue = "default"
        case visibleWhen = "visible_when"
        case requiredWhen = "required_when"
        case readOnly = "read_only"
    }
}

struct EngineError: Decodable, Equatable, Sendable {
    let code: String
    let message: String
}

struct EngineMessage: Decodable, Sendable {
    let id: String?
    let event: String?
    let result: JSONValue?
    let error: EngineError?
    let run: EngineRun?
    let runID: String?
    let line: String?
    let elapsedSeconds: Double?
    let observedElapsedSeconds: Double?
    let phase: String?
    let current: Int?
    let total: Int?
    let unit: String?
    let message: String?
    let step: Int?
    let values: [String: JSONValue]?

    init(
        id: String?, event: String?, result: JSONValue?, error: EngineError?, run: EngineRun?, runID: String?, line: String?,
        elapsedSeconds: Double? = nil, observedElapsedSeconds: Double? = nil, phase: String? = nil,
        current: Int? = nil, total: Int? = nil, unit: String? = nil, message: String? = nil,
        step: Int? = nil, values: [String: JSONValue]? = nil
    ) {
        self.id = id
        self.event = event
        self.result = result
        self.error = error
        self.run = run
        self.runID = runID
        self.line = line
        self.elapsedSeconds = elapsedSeconds
        self.observedElapsedSeconds = observedElapsedSeconds
        self.phase = phase
        self.current = current
        self.total = total
        self.unit = unit
        self.message = message
        self.step = step
        self.values = values
    }

    enum CodingKeys: String, CodingKey {
        case id, event, result, error, run, line, phase, current, total, unit, message, step, values
        case runID = "run_id"
        case elapsedSeconds = "elapsed_seconds"
        case observedElapsedSeconds = "observed_elapsed_seconds"
    }
}
