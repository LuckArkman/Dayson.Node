using Galileu.Node.Core;
using Galileu.Node.TreeSwapFile;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace Galileu.Node.Services;

/// <summary>
/// Gerencia o carregamento de dados do dataset de forma otimizada para memória.
/// Processa o texto, cria pares de input/target como índices e os armazena em disco.
/// Durante o treinamento, carrega apenas os índices em lotes, evitando alocar
/// grandes tensores de alvo (one-hot) na memória da CPU.
/// </summary>
public class DatasetService : IDisposable
{
    private readonly BinaryTreeFileStorage _storage;
    private List<long> _trainOffsets;
    private List<long> _validationOffsets;
    private int _currentTrainIndex = 0;
    private int _currentValidationIndex = 0;
    private int _batchSize;
    private readonly object _lock = new object();

    public DatasetService(string storageFilePath)
    {
        _storage = new BinaryTreeFileStorage(storageFilePath);
        _trainOffsets = new List<long>();
        _validationOffsets = new List<long>();
    }

    /// <summary>
    /// Processa o texto do dataset, cria pares de (InputIndex, TargetIndex)
    /// e os armazena em disco, dividindo-os em conjuntos de treino e validação.
    /// </summary>
    public void InitializeAndSplit(string text, int contextWindow,
        Dictionary<string, int> vocab, string padToken, int initialBatchSize, double validationSplit)
    {
        lock (_lock)
        {
            _batchSize = initialBatchSize;
            _storage.Clear();
            var allOffsets = new List<long>();
            var tokens = text.ToLower().Split(new[] { ' ', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
            int unkIndex = vocab["<UNK>"];
            int totalSamples = 0;

            Console.WriteLine("[DatasetService] Processando e armazenando amostras em disco...");
            for (int i = 0; i < tokens.Length - contextWindow; i++)
            {
                var inputToken = tokens[i + contextWindow - 1];
                var targetToken = tokens[i + contextWindow];

                int inputIndex = vocab.TryGetValue(inputToken, out var idx) ? idx : unkIndex;
                int targetIndex = vocab.TryGetValue(targetToken, out var tidx) ? tidx : unkIndex;

                var pairData = new SampleIndexData
                {
                   InputIndex = inputIndex,
                   TargetIndex = targetIndex
                };

                string json = JsonSerializer.Serialize(pairData);
                long offset = _storage.StoreData(json);
                allOffsets.Add(offset);
                totalSamples++;
            }
            Console.WriteLine($"[DatasetService] Amostras armazenadas: {totalSamples}");

            _storage.Flush();
            Console.WriteLine($"[DatasetService] Armazenamento concluído. {allOffsets.Count} amostras válidas totais.");

            var rnd = new Random();
            allOffsets = allOffsets.OrderBy(x => rnd.Next()).ToList();
            int validationCount = (int)(allOffsets.Count * validationSplit);
            _validationOffsets = allOffsets.Take(validationCount).ToList();
            _trainOffsets = allOffsets.Skip(validationCount).ToList();

            Console.WriteLine($"[DatasetService] Offsets divididos: {_trainOffsets.Count / _batchSize} lotes de treino, {_validationOffsets.Count / _batchSize} lotes de validação.");
        }
    }
    
    public void SetBatchSize(int newSize)
    {
        lock (_lock) { _batchSize = Math.Max(1, newSize); }
    }

    /// <summary>
    /// Retorna o próximo lote de dados de treino como uma lista de tuplas (InputIndex, TargetIndex).
    /// </summary>
    public List<(int InputIndex, int TargetIndex)>? GetNextTrainChunk() => GetNextChunk(ref _currentTrainIndex, _trainOffsets);

    /// <summary>
    /// Retorna o próximo lote de dados de validação como uma lista de tuplas (InputIndex, TargetIndex).
    /// </summary>
    public List<(int InputIndex, int TargetIndex)>? GetNextValidationChunk() => GetNextChunk(ref _currentValidationIndex, _validationOffsets);

    /// <summary>
    /// Lógica central para ler os índices do disco para um determinado lote.
    /// Esta implementação é otimizada para alocar o mínimo de memória possível na CPU.
    /// </summary>
    private List<(int InputIndex, int TargetIndex)>? GetNextChunk(ref int currentIndex, List<long> offsets)
    {
        var chunk = new List<(int, int)>();
        lock (_lock)
        {
            if (currentIndex >= offsets.Count) return null;

            int endIndex = Math.Min(currentIndex + _batchSize, offsets.Count);
            for (int i = currentIndex; i < endIndex; i++)
            {
                string json = _storage.GetData(offsets[i]);
                var pairData = JsonSerializer.Deserialize<SampleIndexData>(json);
                if (pairData != null)
                {
                    // Adiciona apenas os índices. Nenhuma alocação de array grande ocorre aqui.
                    chunk.Add((pairData.InputIndex, pairData.TargetIndex));
                }
            }
            currentIndex = endIndex;
        }
        return chunk;
    }

    public void ResetTrain() => _currentTrainIndex = 0;
    public void ResetValidation() => _currentValidationIndex = 0;

    public void Dispose() => _storage?.Dispose();
}