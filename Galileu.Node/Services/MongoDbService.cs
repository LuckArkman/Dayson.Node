using MongoDB.Driver;
using System.Collections.Generic;
using System.Threading.Tasks;
using Galileu.Node.Data;
using System;
using Galileu.Node.Models;

namespace Galileu.Node.Services;

/// <summary>
/// Gerencia a conexão e as operações com o banco de dados MongoDB
/// para armazenar e recuperar exemplos de treinamento.
/// </summary>
public class MongoDbService
{
    private readonly IMongoCollection<GeneratedExample> _examplesCollection;

    public MongoDbService()
    {
        // IMPORTANTE: Em produção, use um arquivo de configuração para a connection string.
        const string connectionString = "mongodb://dayson:3702959@209.209.40.167:27017/";
        const string databaseName = "dyson_training_cache";
        const string collectionName = "generated_examples";

        var client = new MongoClient(connectionString);
        var database = client.GetDatabase(databaseName);
        _examplesCollection = database.GetCollection<GeneratedExample>(collectionName);
    }

    /// <summary>
    /// Insere uma lista de exemplos de treinamento no banco de dados.
    /// </summary>
    public async Task InsertManyAsync(List<GeneratedExample> examples)
    {
        if (examples == null || !examples.Any()) return;
        try
        {
            await _examplesCollection.InsertManyAsync(examples, new InsertManyOptions { IsOrdered = false });
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n[MongoDB] Erro ao salvar exemplos em lote: {ex.Message}");
        }
    }

    /// <summary>
    /// Busca um único exemplo aleatório da coleção.
    /// Usado como fallback quando a API do modelo professor está indisponível.
    /// </summary>
    public async Task<GeneratedExample?> GetRandomExampleAsync()
    {
        try
        {
            // A agregação com $sample é a forma mais eficiente de obter documentos aleatórios.
            var sample = await _examplesCollection.Aggregate().Sample(1).FirstOrDefaultAsync();
            return sample;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n[MongoDB] Erro ao buscar exemplo aleatório: {ex.Message}");
            return null;
        }
    }
}