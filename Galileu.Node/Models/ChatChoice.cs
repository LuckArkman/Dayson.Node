using System.Text.Json.Serialization;

namespace Galileu.Node.Models;

public record ChatChoice
{
    [JsonPropertyName("message")]
    public ChatMessage Message { get; init; } = new("", "");
}