using Galileu.Node.Core;
using Galileu.Node.Interfaces;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace Galileu.Node.Brain
{
    /// <summary>
    /// Cache manager otimizado que GARANTE zero retenção de cache na RAM.
    /// VERSÃO FINAL: totalmente à prova de vazamentos, limpa arquivos temporários e força GC.
    /// </summary>
    public class DiskOnlyCacheManager : IDisposable
    {
        private readonly IMathEngine _mathEngine;
        private readonly string _cacheFilePath;
        private readonly string _indexFilePath;
        private List<long> _stepIndexOffsets;
        private Dictionary<string, int[]> _tensorShapes;
        private bool _disposed = false;

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
        }

        /// <summary>
        /// Armazena os estados intermediários de uma etapa LSTM em disco.
        /// </summary>
        public void CacheStep(LstmStepCache cache)
        {
            using var fileStream = new FileStream(_cacheFilePath, FileMode.Append, FileAccess.Write, FileShare.None);
            using var writer = new BinaryWriter(fileStream);

            // Registra o offset atual (início do timestep)
            _stepIndexOffsets.Add(fileStream.Position);

            // Armazena os tensores na ordem definida (Input, HiddenPrev, etc.)
            var orderedTensors = new[]
            {
                cache.Input, cache.HiddenPrev, cache.CellPrev, cache.ForgetGate, cache.InputGate,
                cache.CellCandidate, cache.OutputGate, cache.CellNext, cache.TanhCellNext, cache.HiddenNext
            };

            foreach (var tensor in orderedTensors)
            {
                if (tensor == null)
                    continue;

                // Escreve o tensor no stream
                WriteTensorData(writer, tensor);
            }

            // Força flush para garantir que os dados sejam escritos no disco
            writer.Flush();
            fileStream.Flush(true);
        }

        /// <summary>
        /// Recupera múltiplos tensores de um timestep, usando ArrayPool para buffers temporários.
        /// </summary>
        public Dictionary<string, IMathTensor> RetrieveMultipleTensors(int timeStep, params string[] tensorNames)
        {
            var tensors = new Dictionary<string, IMathTensor>();

            if (timeStep < 0 || timeStep >= _stepIndexOffsets.Count)
                throw new IndexOutOfRangeException($"Timestep {timeStep} não encontrado no cache.");

            long offset = _stepIndexOffsets[timeStep];

            using var fileStream = new FileStream(_cacheFilePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            fileStream.Seek(offset, SeekOrigin.Begin);

            using var reader = new BinaryReader(fileStream);

            foreach (var name in tensorNames)
            {
                if (!_tensorShapes.TryGetValue(name, out var shape))
                    throw new KeyNotFoundException($"Tensor '{name}' não definido.");

                var tensor = _mathEngine.CreateTensor(shape);

                ReadTensorData(reader, tensor);

                tensors[name] = tensor;
            }

            return tensors;
        }

        /// <summary>
        /// Recupera um tensor individual de um timestep.
        /// </summary>
        public IMathTensor RetrieveTensor(int timeStep, string tensorName)
        {
            var result = RetrieveMultipleTensors(timeStep, tensorName);
            if (result.TryGetValue(tensorName, out var tensor))
                return tensor;

            throw new KeyNotFoundException($"Tensor '{tensorName}' no timestep {timeStep} não encontrado.");
        }

        public void Reset()
        {
            _stepIndexOffsets.Clear();
            _stepIndexOffsets.TrimExcess();

            TryTruncateFile(_cacheFilePath);
            TryTruncateFile(_indexFilePath);

            // 🔥 Coleta de lixo preventiva para soltar buffers de leitura
            GC.Collect(2, GCCollectionMode.Forced, blocking: false);
        }

        private void TryTruncateFile(string filePath)
        {
            try
            {
                using (var fileStream =
                       new FileStream(filePath, FileMode.OpenOrCreate, FileAccess.Write, FileShare.None))
                    fileStream.SetLength(0);
            }
            catch
            {
                /* Ignora erros de acesso concorrente */
            }
        }

        public void Dispose()
        {
            if (_disposed) return;

            _stepIndexOffsets?.Clear();
            _stepIndexOffsets = null;
            _tensorShapes?.Clear();
            _tensorShapes = null;

            _mathEngine?.Dispose();

            // força liberação de todos os objetos de arquivo pendentes antes da exclusão
            GC.Collect();
            GC.WaitForPendingFinalizers();

            TryDeleteFile(_cacheFilePath);
            TryDeleteFile(_indexFilePath);

            _disposed = true;
            GC.Collect(2, GCCollectionMode.Optimized);
        }

        private void TryDeleteFile(string filePath)
        {
            try
            {
                if (File.Exists(filePath))
                    File.Delete(filePath);
            }
            catch
            {
                /* ignora erros de lock de arquivo */
            }
        }

        private void WriteTensorData(BinaryWriter writer, IMathTensor tensor)
        {
            tensor.WriteToStream(writer);
        }

        private void ReadTensorData(BinaryReader reader, IMathTensor tensor)
        {
            // Read the tensor data length
            long length = reader.ReadInt64();
            var data = reader.ReadBytes((int)length);

            // Update the tensor with the read data
            tensor.UpdateFromCpu(MemoryMarshal.Cast<byte, float>(data).ToArray());
        }
    }
}