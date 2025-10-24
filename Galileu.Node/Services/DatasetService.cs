using Galileu.Node.Core;
using Galileu.Node.TreeSwapFile;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace Galileu.Node.Services;

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

    public void InitializeAndSplit(List<(string input, string output)> dataset, int contextWindow,
        Dictionary<string, int> vocab, string padToken, int initialBatchSize, double validationSplit)
    {
        lock (_lock)
        {
            _batchSize = initialBatchSize;
            _storage.Clear();
            var allOffsets = new List<long>();

            if (dataset == null || dataset.Count == 0)
            {
                Console.WriteLine("[DatasetService] AVISO: O dataset fornecido está vazio.");
                _trainOffsets = new List<long>();
                _validationOffsets = new List<long>();
                return;
            }

            int unkIndex = vocab.ContainsKey("<UNK>") ? vocab["<UNK>"] : 0;
            int totalSamples = 0;

            Console.WriteLine($"[DatasetService] Processando {dataset.Count} pares de input/output para gerar amostras...");

            foreach (var pair in dataset)
            {
                var text = pair.input + " " + pair.output;
                var tokens = text.Split(new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries)
                                 .Select(t => t.ToLower())
                                 .ToArray();

                if (tokens.Length < 2) continue;

                for (int i = 0; i < tokens.Length - 1; i++)
                {
                    var inputToken = tokens[i];
                    var targetToken = tokens[i + 1];

                    int inputIndex = vocab.TryGetValue(inputToken, out var idx) ? idx : unkIndex;
                    int targetIndex = vocab.TryGetValue(targetToken, out var tidx) ? tidx : unkIndex;

                    var pairData = new SampleIndexData { InputIndex = inputIndex, TargetIndex = targetIndex };
                    string json = JsonSerializer.Serialize(pairData);
                    long offset = _storage.StoreData(json);
                    allOffsets.Add(offset);
                    totalSamples++;
                }
            }
            Console.WriteLine($"[DatasetService] Amostras armazenadas: {totalSamples}");

            _storage.Flush();

            var rnd = new Random(42);
            allOffsets = allOffsets.OrderBy(x => rnd.Next()).ToList();
            int validationCount = (int)(allOffsets.Count * validationSplit);
            _validationOffsets = allOffsets.Take(validationCount).ToList();
            _trainOffsets = allOffsets.Skip(validationCount).ToList();

            Console.WriteLine($"[DatasetService] Offsets divididos: {_trainOffsets.Count / Math.Max(1, _batchSize)} lotes de treino, {_validationOffsets.Count / Math.Max(1, _batchSize)} lotes de validação.");
        }
    }
    
    public void InitializeAndSplit(string text, int contextWindow, Dictionary<string, int> vocab, string padToken, int initialBatchSize, double validationSplit)
    {
        InitializeAndSplit(new List<(string, string)> { (text, "") }, contextWindow, vocab, padToken, initialBatchSize, validationSplit);
    }
    
    public void SetBatchSize(int newSize) { _batchSize = Math.Max(1, newSize); }
    public List<(int InputIndex, int TargetIndex)>? GetNextTrainChunk() => GetNextChunk(ref _currentTrainIndex, _trainOffsets);
    public List<(int InputIndex, int TargetIndex)>? GetNextValidationChunk() => GetNextChunk(ref _currentValidationIndex, _validationOffsets);
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
                if (pairData != null) chunk.Add((pairData.InputIndex, pairData.TargetIndex));
            }
            currentIndex = endIndex;
        }
        return chunk.Count > 0 ? chunk : null;
    }
    public void ResetTrain() => _currentTrainIndex = 0;
    public void ResetValidation() => _currentValidationIndex = 0;
    public void Dispose() => _storage?.Dispose();
}