using MongoDB.Bson;

namespace Galileu.Node.Data;

/// <summary>
/// Representa um único exemplo de treinamento gerado pelo modelo professor,
/// formatado para ser armazenado em uma coleção do MongoDB.
/// </summary>
public class GeneratedExample
{
    public ObjectId Id { get; set; }
    public string Input { get; set; }
    public string Output { get; set; }
    public DateTime CreatedAt { get; set; }

    public GeneratedExample(ObjectId id, string input, string output, DateTime createdAt)
    {
        Id = id;
        Input = input;
        Output = output;
        CreatedAt = createdAt;
    }
}