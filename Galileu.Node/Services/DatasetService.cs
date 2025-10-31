using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Galileu.Node.Services
{
    public class DatasetService : IDisposable
    {
        private readonly string _swapFilePath;
        private List<int> _trainBatchIndices;
        private List<int> _validationBatchIndices;
        private int _trainIndex;
        private int _validationIndex;
        private int _batchSize;
        private int _contextWindowSize;
        private string _batchDirectory;

        public DatasetService(string swapFilePath)
        {
            _swapFilePath = swapFilePath;
            _trainBatchIndices = new List<int>();
            _validationBatchIndices = new List<int>();
            _trainIndex = 0;
            _validationIndex = 0;
            _batchSize = 1;
            _contextWindowSize = 1;
            _batchDirectory = Path.Combine(Path.GetDirectoryName(swapFilePath) ?? "Dayson", "batches");
            Directory.CreateDirectory(_batchDirectory);
            Console.WriteLine($"[DatasetService] Diretório de lotes criado: {_batchDirectory}");
        }

        public void InitializeAndSplit(string text, int contextWindowSize, Dictionary<string, int> vocab, string padToken, int batchSize, float validationSplit)
        {
            Console.WriteLine($"[DatasetService] Inicializando dataset com contextWindowSize={contextWindowSize}, vocabSize={vocab.Count}, batchSize={batchSize}, validationSplit={validationSplit}");
            long memoryBefore = GC.GetTotalMemory(true) / (1024 * 1024);
            Console.WriteLine($"[DatasetService] Memória antes da inicialização: {memoryBefore} MB");

            var tokens = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            var indices = tokens.Select(t => vocab.ContainsKey(t) ? vocab[t] : vocab[padToken]).ToArray();

            var sequences = new List<(int[] InputIndices, int[] TargetIndices)>();
            for (int i = 0; i <= indices.Length - contextWindowSize; i++)
            {
                var inputIndices = new int[contextWindowSize - 1];
                var targetIndices = new int[contextWindowSize - 1];
                for (int j = 0; j < contextWindowSize - 1; j++)
                {
                    inputIndices[j] = indices[i + j];
                    targetIndices[j] = indices[i + j + 1];
                }
                sequences.Add((inputIndices, targetIndices));
            }

            Console.WriteLine($"[DatasetService] Geradas {sequences.Count} sequências de tamanho {contextWindowSize - 1}");

            int validationSize = (int)(sequences.Count * validationSplit);
            int trainSize = sequences.Count - validationSize;

            _batchSize = batchSize;
            _contextWindowSize = contextWindowSize;

            for (int i = 0; i < trainSize; i += batchSize)
            {
                var batch = sequences.Skip(i).Take(batchSize).ToList();
                if (batch.Count > 0)
                {
                    SaveBatchToDisk(batch, $"train_batch_{i / batchSize}.bin");
                    _trainBatchIndices.Add(i / batchSize);
                }
            }

            for (int i = trainSize; i < sequences.Count; i += batchSize)
            {
                var batch = sequences.Skip(i).Take(batchSize).ToList();
                if (batch.Count > 0)
                {
                    SaveBatchToDisk(batch, $"validation_batch_{(i - trainSize) / batchSize}.bin");
                    _validationBatchIndices.Add((i - trainSize) / batchSize);
                }
            }

            sequences = null; // Libera memória
            GC.Collect(2, GCCollectionMode.Forced, true, true);
            GC.WaitForPendingFinalizers();

            long memoryAfter = GC.GetTotalMemory(true) / (1024 * 1024);
            //Console.WriteLine($"[DatasetService] Memória após inicialização: {memoryAfter} MB (Delta: {memoryAfter - memoryBefore} MB)");
            Console.WriteLine($"[DatasetService] Dados de treino: {_trainBatchIndices.Count} lotes, Dados de validação: {_validationBatchIndices.Count} lotes, Tamanho do lote: {_batchSize}");
        }

        private void SaveBatchToDisk(List<(int[] InputIndices, int[] TargetIndices)> batch, string fileName)
        {
            string filePath = Path.Combine(_batchDirectory, fileName);
            try
            {
                using var fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
                using var writer = new BinaryWriter(fileStream);
                writer.Write(batch.Count);
                foreach (var (inputIndices, targetIndices) in batch)
                {
                    writer.Write(inputIndices.Length);
                    writer.Write(targetIndices.Length);
                    foreach (var index in inputIndices)
                        writer.Write(index);
                    foreach (var index in targetIndices)
                        writer.Write(index);
                }
                writer.Flush();
                fileStream.Flush(true);
                //Console.WriteLine($"[DatasetService] Lote salvo em: {filePath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[DatasetService] Erro ao salvar lote {fileName}: {ex.Message}");
                throw;
            }
        }

        public List<(int[] InputIndices, int[] TargetIndices)> LoadBatchFromDisk(int batchIndex, bool isValidation)
        {
            string fileName = isValidation ? $"validation_batch_{batchIndex}.bin" : $"train_batch_{batchIndex}.bin";
            string filePath = Path.Combine(_batchDirectory, fileName);
            if (!File.Exists(filePath))
            {
                Console.WriteLine($"[DatasetService] Arquivo de lote não encontrado: {filePath}");
                return new List<(int[], int[])>();
            }

            long memoryBefore = GC.GetTotalMemory(true) / (1024 * 1024);
            //Console.WriteLine($"[DatasetService] Memória antes de carregar {filePath}: {memoryBefore} MB");

            try
            {
                using var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
                using var reader = new BinaryReader(fileStream);
                var batch = new List<(int[] InputIndices, int[] TargetIndices)>();

                int sequenceCount = reader.ReadInt32();
                for (int i = 0; i < sequenceCount; i++)
                {
                    int inputLength = reader.ReadInt32();
                    int targetLength = reader.ReadInt32();
                    if (inputLength != _contextWindowSize - 1 || targetLength != _contextWindowSize - 1)
                    {
                        //Console.WriteLine($"[DatasetService] Aviso: Sequência inválida em {filePath}. Entrada: {inputLength}, Alvo: {targetLength}, Esperado: {_contextWindowSize - 1}");
                        continue;
                    }

                    var inputIndices = new int[inputLength];
                    var targetIndices = new int[targetLength];
                    for (int j = 0; j < inputLength; j++)
                        inputIndices[j] = reader.ReadInt32();
                    for (int j = 0; j < targetLength; j++)
                        targetIndices[j] = reader.ReadInt32();

                    batch.Add((inputIndices, targetIndices));
                }

                long memoryAfter = GC.GetTotalMemory(true) / (1024 * 1024);
                //Console.WriteLine($"[DatasetService] Lote carregado de {filePath} com {batch.Count} sequências. Memória após: {memoryAfter} MB (Delta: {memoryAfter - memoryBefore} MB)");
                return batch;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[DatasetService] Erro ao carregar lote {filePath}: {ex.Message}");
                throw;
            }
        }

        public List<int> GetTrainBatchIndices()
        {
            return _trainBatchIndices;
        }

        public List<int> GetValidationBatchIndices()
        {
            return _validationBatchIndices;
        }

        public void ResetTrain()
        {
            _trainIndex = 0;
            Console.WriteLine("[DatasetService] Resetando índice de treino");
        }

        public void ResetValidation()
        {
            _validationIndex = 0;
            Console.WriteLine("[DatasetService] Resetando índice de validação");
        }

        public void Dispose()
        {
            long memoryBefore = GC.GetTotalMemory(true) / (1024 * 1024);
            _trainBatchIndices.Clear();
            _validationBatchIndices.Clear();
            try
            {
                if (Directory.Exists(_batchDirectory))
                {
                    foreach (var file in Directory.GetFiles(_batchDirectory))
                    {
                        File.Delete(file);
                        //Console.WriteLine($"[DatasetService] Arquivo deletado: {file}");
                    }
                    Directory.Delete(_batchDirectory);
                    //Console.WriteLine($"[DatasetService] Diretório de lotes deletado: {_batchDirectory}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[DatasetService] Erro ao deletar arquivos de lotes: {ex.Message}");
            }
            long memoryAfter = GC.GetTotalMemory(true) / (1024 * 1024);
            Console.WriteLine($"[DatasetService] Memória após Dispose: {memoryAfter} MB (Delta: {memoryAfter - memoryBefore} MB)");
        }
    }
}