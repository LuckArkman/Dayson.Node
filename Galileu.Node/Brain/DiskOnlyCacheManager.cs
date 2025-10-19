using Galileu.Node.Core;
using Galileu.Node.Interfaces;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Galileu.Node.Brain;

/// <summary>
/// Cache manager que GARANTE zero retenção de cache na RAM.
/// VERSÃO CORRIGIDA: Agora deleta arquivos temporários e libera toda memória no Dispose.
/// </summary>
public class DiskOnlyCacheManager : IDisposable
{
    private readonly IMathEngine _mathEngine;
    private readonly string _cacheFilePath;
    private readonly string _indexFilePath;
    private List<long> _stepIndexOffsets;
    private Dictionary<string, int[]> _tensorShapes;

    public static class TensorNames
    {
        public const string Input = "Input";
        public const string HiddenPrev = "HiddenPrev";
        public const string CellPrev = "CellPrev";
        public const string ForgetGate = "ForgetGate";
        public const string InputGate = "InputGate";
        public const string CellCandidate = "CellCandidate";
        public const string OutputGate = "OutputGate";
        public const string CellNext = "CellNext";
        public const string TanhCellNext = "TanhCellNext";
        public const string HiddenNext = "HiddenNext";
    }

    public DiskOnlyCacheManager(IMathEngine mathEngine, int embeddingSize, int hiddenSize)
    {
        _mathEngine = mathEngine;
        _stepIndexOffsets = new List<long>(); 
        _tensorShapes = new Dictionary<string, int[]>
        {
            { TensorNames.Input, new[] { 1, embeddingSize } },
            { TensorNames.HiddenPrev, new[] { 1, hiddenSize } },
            { TensorNames.CellPrev, new[] { 1, hiddenSize } },
            { TensorNames.ForgetGate, new[] { 1, hiddenSize } },
            { TensorNames.InputGate, new[] { 1, hiddenSize } },
            { TensorNames.CellCandidate, new[] { 1, hiddenSize } },
            { TensorNames.OutputGate, new[] { 1, hiddenSize } },
            { TensorNames.CellNext, new[] { 1, hiddenSize } },
            { TensorNames.TanhCellNext, new[] { 1, hiddenSize } },
            { TensorNames.HiddenNext, new[] { 1, hiddenSize } }
        };

        var guid = Guid.NewGuid();
        _cacheFilePath = Path.Combine(Environment.CurrentDirectory, "Dayson", $"lstm_cache_data_{guid}.bin");
        _indexFilePath = Path.Combine(Environment.CurrentDirectory, "Dayson", $"lstm_cache_index_{guid}.bin");

        var directory = Path.GetDirectoryName(_cacheFilePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        using (File.Create(_cacheFilePath)) { }
        using (File.Create(_indexFilePath)) { }

        Console.WriteLine($"[DiskOnlyCache] Inicializado. Dados: {_cacheFilePath}");
        Console.WriteLine($"[DiskOnlyCache] Política: Metadados em disco. Índice: {_indexFilePath}");
    }

    public void CacheStep(LstmStepCache stepCache)
    {
        using (var dataStream = new FileStream(_cacheFilePath, FileMode.Append, FileAccess.Write, FileShare.None))
        using (var dataWriter = new BinaryWriter(dataStream))
        using (var indexStream = new FileStream(_indexFilePath, FileMode.Append, FileAccess.Write, FileShare.None))
        using (var indexWriter = new BinaryWriter(indexStream))
        {
            _stepIndexOffsets.Add(indexStream.Position);

            WriteTensorAndIndex(dataWriter, dataStream, indexWriter, TensorNames.Input, stepCache.Input!);
            WriteTensorAndIndex(dataWriter, dataStream, indexWriter, TensorNames.HiddenPrev, stepCache.HiddenPrev!);
            WriteTensorAndIndex(dataWriter, dataStream, indexWriter, TensorNames.CellPrev, stepCache.CellPrev!);
            WriteTensorAndIndex(dataWriter, dataStream, indexWriter, TensorNames.ForgetGate, stepCache.ForgetGate!);
            WriteTensorAndIndex(dataWriter, dataStream, indexWriter, TensorNames.InputGate, stepCache.InputGate!);
            WriteTensorAndIndex(dataWriter, dataStream, indexWriter, TensorNames.CellCandidate, stepCache.CellCandidate!);
            WriteTensorAndIndex(dataWriter, dataStream, indexWriter, TensorNames.OutputGate, stepCache.OutputGate!);
            WriteTensorAndIndex(dataWriter, dataStream, indexWriter, TensorNames.CellNext, stepCache.CellNext!);
            WriteTensorAndIndex(dataWriter, dataStream, indexWriter, TensorNames.TanhCellNext, stepCache.TanhCellNext!);
            WriteTensorAndIndex(dataWriter, dataStream, indexWriter, TensorNames.HiddenNext, stepCache.HiddenNext!);

            dataWriter.Flush();
            indexWriter.Flush();
        }
    }
    
    private void WriteTensorAndIndex(BinaryWriter dataWriter, FileStream dataStream,
        BinaryWriter indexWriter, string name, IMathTensor tensor)
    {
        long dataOffset = dataStream.Position;
        indexWriter.Write(name);
        indexWriter.Write(dataOffset);
        WriteTensorData(dataWriter, tensor);
    }
    
    public Dictionary<string, IMathTensor> RetrieveMultipleTensors(int timeStep, params string[] tensorNames)
    {
        if (timeStep >= _stepIndexOffsets.Count)
        {
            throw new KeyNotFoundException($"Timestep {timeStep} não encontrado no cache.");
        }

        var offsets = new Dictionary<string, long>();
        using (var indexStream = new FileStream(_indexFilePath, FileMode.Open, FileAccess.Read, FileShare.Read))
        using (var indexReader = new BinaryReader(indexStream))
        {
            long indexStartOffset = _stepIndexOffsets[timeStep];
            indexStream.Seek(indexStartOffset, SeekOrigin.Begin);

            for (int i = 0; i < _tensorShapes.Count; i++)
            {
                string name = indexReader.ReadString();
                long offset = indexReader.ReadInt64();
                offsets[name] = offset;
            }
        }

        var tensors = new Dictionary<string, IMathTensor>();
        var sortedNames = tensorNames
            .Where(name => offsets.ContainsKey(name))
            .OrderBy(name => offsets[name])
            .ToArray();

        using (var dataStream = new FileStream(_cacheFilePath, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 131072))
        using (var dataReader = new BinaryReader(dataStream))
        {
            foreach (var name in sortedNames)
            {
                long offset = offsets[name];
                int[] shape = _tensorShapes[name];

                dataStream.Seek(offset, SeekOrigin.Begin);
                long length = dataReader.ReadInt64();
                double[] data = new double[length];
                
                for (long i = 0; i < length; i++)
                {
                    data[i] = dataReader.ReadDouble();
                }

                tensors[name] = _mathEngine.CreateTensor(data, shape);
            }
        }

        return tensors;
    }
    
    public IMathTensor RetrieveTensor(int timeStep, string tensorName)
    {
        var result = RetrieveMultipleTensors(timeStep, tensorName);
        if (result.TryGetValue(tensorName, out var tensor))
        {
            return tensor;
        }
        throw new KeyNotFoundException($"Tensor '{tensorName}' no timestep {timeStep} não encontrado.");
    }

    public void Reset()
{
    try
    {
        // ========================================
        // FASE 1: LIMPAR ESTRUTURAS EM MEMÓRIA
        // ========================================
        
        // Limpa lista de índices
        if (_stepIndexOffsets != null)
        {
            _stepIndexOffsets.Clear();
            
            // Libera capacidade extra se lista cresceu muito
            if (_stepIndexOffsets.Capacity > 1000)
            {
                _stepIndexOffsets.TrimExcess();
            }
        }
        
        // 🔥 NOVO: Limpa dicionário de shapes também
        if (_tensorShapes != null)
        {
            _tensorShapes.Clear();
            
            // Recria com tamanho inicial para evitar retenção de memória
            _tensorShapes = new Dictionary<string, int[]>
            {
                { TensorNames.Input, new[] { 1, _tensorShapes.ContainsKey(TensorNames.Input) ? _tensorShapes[TensorNames.Input][1] : 0 } },
                { TensorNames.HiddenPrev, new[] { 1, _tensorShapes.ContainsKey(TensorNames.HiddenPrev) ? _tensorShapes[TensorNames.HiddenPrev][1] : 0 } },
                { TensorNames.CellPrev, new[] { 1, _tensorShapes.ContainsKey(TensorNames.CellPrev) ? _tensorShapes[TensorNames.CellPrev][1] : 0 } },
                { TensorNames.ForgetGate, new[] { 1, _tensorShapes.ContainsKey(TensorNames.ForgetGate) ? _tensorShapes[TensorNames.ForgetGate][1] : 0 } },
                { TensorNames.InputGate, new[] { 1, _tensorShapes.ContainsKey(TensorNames.InputGate) ? _tensorShapes[TensorNames.InputGate][1] : 0 } },
                { TensorNames.CellCandidate, new[] { 1, _tensorShapes.ContainsKey(TensorNames.CellCandidate) ? _tensorShapes[TensorNames.CellCandidate][1] : 0 } },
                { TensorNames.OutputGate, new[] { 1, _tensorShapes.ContainsKey(TensorNames.OutputGate) ? _tensorShapes[TensorNames.OutputGate][1] : 0 } },
                { TensorNames.CellNext, new[] { 1, _tensorShapes.ContainsKey(TensorNames.CellNext) ? _tensorShapes[TensorNames.CellNext][1] : 0 } },
                { TensorNames.TanhCellNext, new[] { 1, _tensorShapes.ContainsKey(TensorNames.TanhCellNext) ? _tensorShapes[TensorNames.TanhCellNext][1] : 0 } },
                { TensorNames.HiddenNext, new[] { 1, _tensorShapes.ContainsKey(TensorNames.HiddenNext) ? _tensorShapes[TensorNames.HiddenNext][1] : 0 } }
            };
        }
        
        // ========================================
        // FASE 2: TRUNCAR ARQUIVOS
        // ========================================
        
        // Trunca arquivos (mantém os arquivos mas remove conteúdo)
        TruncateFile(_cacheFilePath);
        TruncateFile(_indexFilePath);
        
        // ========================================
        // FASE 3: FORÇAR FLUSH DE BUFFERS DO SO
        // ========================================
        
        // Força o sistema operacional a liberar buffers de I/O
        GC.Collect(0, GCCollectionMode.Optimized, false);
        GC.WaitForPendingFinalizers();
        
    }
    catch (Exception ex)
    {
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine($"[DiskOnlyCache] Aviso durante Reset: {ex.Message}");
        Console.ResetColor();
    }
}

// NOVO MÉTODO: Reset mais agressivo (usado antes do Dispose)
public void ResetAndClear()
{
    // Primeiro faz o reset normal
    Reset();
    
    // Depois anula as estruturas
    _stepIndexOffsets = null;
    _tensorShapes = null;
    
    Console.WriteLine("[DiskOnlyCache] Reset completo executado (estruturas anuladas)");
}

    private void TruncateFile(string filePath)
    {
        try
        {
            using (var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Write, FileShare.None))
            {
                fileStream.SetLength(0);
            }
        }
        catch (FileNotFoundException) { /* OK */ }
    }

    public void Dispose()
    {
        
        Console.WriteLine($"[DiskOnlyCache] Iniciando limpeza completa...");
        
        // PASSO 1: Limpa estruturas de memória
        _stepIndexOffsets?.Clear();
        _stepIndexOffsets = null;
        
        _tensorShapes?.Clear();
        _tensorShapes = null;
        
        // PASSO 2: Deleta arquivos físicos
        TryDeleteFile(_cacheFilePath);
        TryDeleteFile(_indexFilePath);
        GC.SuppressFinalize(this);
        
        Console.WriteLine($"[DiskOnlyCache] Limpeza completa concluída.");
    }
    
    private void TryDeleteFile(string filePath)
    {
        try
        {
            if (File.Exists(filePath))
            {
                File.Delete(filePath);
                Console.WriteLine($"[DiskOnlyCache] Arquivo deletado: {Path.GetFileName(filePath)}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[DiskOnlyCache] Aviso: Não foi possível deletar '{Path.GetFileName(filePath)}': {ex.Message}");
        }
    }
    
    private void WriteTensorData(BinaryWriter writer, IMathTensor tensor)
    {
        tensor.WriteToStream(writer);
    }
}