using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;

namespace Services;

public static class CryptoUtils
{
    // Gera um par de chaves para um novo usuário (carteira)
    public static (string publicKey, string privateKey) GenerateKeyPair()
    {
        using var rsa = RSA.Create();
        return (
            publicKey: Convert.ToBase64String(rsa.ExportRSAPublicKey()),
            privateKey: Convert.ToBase64String(rsa.ExportRSAPrivateKey())
        );
    }

    public static string CreateTransactionHash(Guid id, DateTime timestamp, string from, string to, decimal amount)
    {
        using var sha256 = SHA256.Create();
        var data = $"{id}{timestamp:O}{from}{to}{amount}";
        var hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(data));
        return Convert.ToBase64String(hashBytes);
    }

    public static string NormalizePublicKey(string rawKey)
    {
        if (string.IsNullOrWhiteSpace(rawKey)) return string.Empty;
        var key = rawKey.Replace("-----BEGIN PUBLIC KEY-----", "")
            .Replace("-----END PUBLIC KEY-----", "");

        // Remove todos os caracteres de espaço em branco (espaços, tabs, quebras de linha \r e \n)
        key = Regex.Replace(key, @"\s+", "");

        return key;
    }
}