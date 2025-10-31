using MongoDB.Bson;
using MongoDB.Bson.Serialization.Attributes;

namespace Galileu.Node.Models;

public class WalletDocument
{
    [BsonId] // Usa o ObjectId do MongoDB como chave primária
    [BsonRepresentation(BsonType.ObjectId)]
    public string? Id { get; set; }

    [BsonElement("address")] public string? Address { get; set; } // A chave pública/endereço da carteira

    [BsonElement("createdAt")] public DateTime CreatedAt { get; set; }
}