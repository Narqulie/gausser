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

    enum CodingKeys: String, CodingKey {
        case id, event, result, error, run, line
        case runID = "run_id"
    }
}
