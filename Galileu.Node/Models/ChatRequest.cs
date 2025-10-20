using System.Text.Json.Serialization;

namespace Galileu.Node.Models;

public record ChatRequest(
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("messages")] List<ChatMessage> Messages,
    [property: JsonPropertyName("stream")] bool Stream = false
);