using System.Text.Json.Serialization;

namespace Galileu.Node.Models;

public record ChatMessage(
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("content")] string Content
);