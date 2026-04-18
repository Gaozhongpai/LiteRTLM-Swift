import Foundation

/// Declaration of a tool the model can call during a conversation.
///
/// `parametersJSON` is a JSON string of a JSON-schema object, e.g.
/// `{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`.
public struct ToolDeclaration: Sendable {
    public let name: String
    public let description: String
    public let parametersJSON: String

    public init(name: String, description: String, parametersJSON: String) {
        self.name = name
        self.description = description
        self.parametersJSON = parametersJSON
    }

    /// Build a declaration from a JSON-schema-shaped dictionary.
    /// Returns nil if `parameters` cannot be serialized.
    public init?(name: String, description: String, parameters: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: parameters),
              let json = String(data: data, encoding: .utf8) else {
            return nil
        }
        self.name = name
        self.description = description
        self.parametersJSON = json
    }
}

/// A single tool invocation requested by the model.
///
/// `argumentsJSON` is the raw JSON of the arguments object (e.g. `{"location":"Paris"}`).
/// Consumers decode it into whatever shape their handler expects.
public struct ToolCall: Sendable, Hashable {
    public let name: String
    public let argumentsJSON: String

    public init(name: String, argumentsJSON: String) {
        self.name = name
        self.argumentsJSON = argumentsJSON
    }
}

/// The outcome of running a tool, to be sent back to the model.
///
/// `contentJSON` is the raw JSON of the result object. By convention the
/// object includes a `tool_name` field, but the runtime treats the content
/// as an opaque payload.
public struct ToolResult: Sendable {
    public let toolName: String
    public let contentJSON: String

    public init(toolName: String, contentJSON: String) {
        self.toolName = toolName
        self.contentJSON = contentJSON
    }

    /// Build a result from a dictionary; returns nil if it can't be serialized.
    public init?(toolName: String, content: [String: Any]) {
        var merged = content
        if merged["tool_name"] == nil { merged["tool_name"] = toolName }
        guard let data = try? JSONSerialization.data(withJSONObject: merged),
              let json = String(data: data, encoding: .utf8) else {
            return nil
        }
        self.toolName = toolName
        self.contentJSON = json
    }
}

/// A single turn from the model. Either natural-language text or one or more
/// tool calls the application is expected to execute and feed back in.
public enum ConversationTurn: Sendable {
    case text(String)
    case toolCalls([ToolCall])
}
