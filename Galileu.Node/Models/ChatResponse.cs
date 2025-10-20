using System.Text.Json.Serialization;

namespace Galileu.Node.Models;

public record ChatResponse
{
    [JsonPropertyName("choices")]
    public List<ChatChoice> Choices { get; init; } = new();
}