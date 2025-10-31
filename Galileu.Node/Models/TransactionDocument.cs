using MongoDB.Bson;
using MongoDB.Bson.Serialization.Attributes;

namespace Galileu.Node.Models;

public class TransactionDocument
{
    public string Id { get; set; } = null!;

    [BsonElement("fromAddress")] public string? fromAddress { get; set; } = null!;

    [BsonElement("toAddress")] public string toAddress { get; set; } = null!;
    public decimal amount { get; set; }

    [BsonElement("timestamp")] public DateTime timestamp { get; set; }

    [BsonElement("notes")] public string? notes { get; set; } = null!;

    [BsonElement("hash")] public string hash { get; set; } = null!;
}