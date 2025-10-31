// --- START OF FILE ModelTrainerLSTM.cs ---

using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Galileu.Node.Core;
using Galileu.Node.Gpu;
using Galileu.Node.Interfaces;
using Galileu.Node.Services;

namespace Galileu.Node.Brain
{
    public class ModelTrainerLSTM
    {
        private readonly IMathEngine _mathEngine;
        private readonly Stopwatch _stopwatch = new Stopwatch();
        private readonly Process _currentProcess;
        private readonly ISearchService _searchService;
        private long _peakMemoryUsageMB = 0;
        private readonly string logPath = Path.Combine(Environment.CurrentDirectory, "Dayson", "training_log.txt");

        public ModelTrainerLSTM(IMathEngine mathEngine)
        {
            _mathEngine = mathEngine ?? throw new ArgumentNullException(nameof(mathEngine));
            _currentProcess = Process.GetCurrentProcess();
            _searchService = new MockSearchService(); // Instancia um mock, já que é um serviço de suporte.
        }

        public void TrainModel(
            GenerativeNeuralNetworkLSTM initialModel,
            string datasetPath,
            string finalModelPath,
            float learningRate,
            int epochs,
            int batchSize,
            int contextWindowSize,
            float validationSplit)
        {
            Console.WriteLine($"{nameof(TrainModel)} >> {datasetPath}");
            if (!File.Exists(datasetPath))
                throw new FileNotFoundException("Arquivo de dataset não encontrado.", datasetPath);

            var swapFilePath = Path.Combine(Environment.CurrentDirectory, "Dayson", "memory.bin");
            using (var datasetService = new DatasetService(swapFilePath))
            {
                datasetService.InitializeAndSplit(File.ReadAllText(datasetPath), contextWindowSize,
                    initialModel.vocabularyManager.Vocab, "<PAD>", batchSize, validationSplit);

                Console.WriteLine($"\n[Trainer] Configuração do Ciclo de Treinamento com Liberação Total de Memória:");
                Console.WriteLine($"  - Estratégia: Salvar → Descartar TUDO → GC Agressivo → Recarregar");
                Console.WriteLine($"  - Objetivo: 100% de liberação de memória garantida entre épocas.\n");

                GenerativeNeuralNetworkLSTM? currentModel = initialModel;
                TimeSpan totalElapsedTime = TimeSpan.Zero;

                var trainBatchIndices = datasetService.GetTrainBatchIndices();
                var validationBatchIndices = datasetService.GetValidationBatchIndices();

                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    _stopwatch.Restart();
                    Console.WriteLine($"\n{'═',60}");
                    Console.WriteLine($"ÉPOCA {epoch + 1}/{epochs} >> Learning Rate : {learningRate} >> {DateTime.UtcNow}");
                    Console.WriteLine($"{'═',60}");

                    double totalEpochLoss = 0;
                    int batchCount = 0;

                    // ==================== FASE 1: TREINAMENTO ====================
                    foreach (var batchIndex in trainBatchIndices)
                    {
                        var batch = datasetService.LoadBatchFromDisk(batchIndex, isValidation: false);
                        if (batch == null || !batch.Any()) continue;

                        // Validação do tamanho do lote
                        foreach (var (inputIndices, targetIndices) in batch)
                        {
                            // A classe base TrainSequence agora lida com o estado e limpeza.
                            totalEpochLoss += currentModel.TrainSequence(inputIndices, targetIndices, learningRate);
                        }

                        batchCount++;
                        Console.WriteLine($"Época: {epoch + 1}/{epochs} | Lotes Processados: {batchCount}/{trainBatchIndices.Count}");
                        
                        // Sincroniza a GPU para obter medições de tempo mais precisas, se aplicável.
                        if (_mathEngine is GpuMathEngine gpuEngine)
                            gpuEngine.Synchronize();
                        
                        // Libera a memória do lote lido do disco.
                        batch = null;
                        GC.Collect(0, GCCollectionMode.Optimized);
                    }

                    _stopwatch.Stop();
                    totalElapsedTime += _stopwatch.Elapsed;
                    double avgLoss = batchCount > 0 ? totalEpochLoss / (batchCount * batchSize) : float.PositiveInfinity;
                    string elapsedFormatted = $"{(int)_stopwatch.Elapsed.TotalHours:D2}:{_stopwatch.Elapsed.Minutes:D2}:{_stopwatch.Elapsed.Seconds:D2}";

                    File.AppendAllText(logPath, $"Época {epoch + 1}/{epochs} concluída. Perda média: {avgLoss:F4}. Duração: {elapsedFormatted}" + Environment.NewLine);
                    Console.WriteLine($"\nÉpoca {epoch + 1}/{epochs} concluída. Perda média: {avgLoss:F4} | Duração: {elapsedFormatted}");

                    // ==================== FASE 2: VALIDAÇÃO ====================
                    double validationLoss = ValidateModel(currentModel, datasetService, validationBatchIndices);
                    File.AppendAllText(logPath, $"[Época {epoch + 1}] Perda Média de Validação: {validationLoss:F4}" + Environment.NewLine);
                    Console.WriteLine($"[Época {epoch + 1}] Perda Média de Validação: {validationLoss:F4}");
                    
                    // ==================== FASE 2.5: LIMPEZA DOS TENSORES DA ÉPOCA ====================
                    // Destrói todos os tensores temporários criados (intermediários, gradientes, etc.)
                    currentModel.ClearEpochTemporaryTensors();

                    // ==================== FASE 3: LIBERAÇÃO TOTAL DE MEMÓRIA ====================
                    long memoryBefore = GetCurrentMemoryUsageMB();
                    string modelPathForEpoch = Path.Combine(Path.GetDirectoryName(finalModelPath)!, $"dayson_{epoch + 1}.json");

                    Console.WriteLine($"\n╔════════════════════════════════════════════════════════════╗");
                    Console.WriteLine($"║  INICIANDO LIBERAÇÃO COMPLETA DE MEMÓRIA (ÉPOCA {epoch + 1})       ║");
                    Console.WriteLine($"╚════════════════════════════════════════════════════════════╝");

                    Console.WriteLine($"[Passo 1/5] Salvando modelo em {modelPathForEpoch}...");
                    currentModel.SaveModel(modelPathForEpoch);

                    // REMOVIDO: Limpeza de TensorPool, pois não existe mais.
                    Console.WriteLine("[Passo 2/5] Resetando estados do AdamOptimizer...");
                    currentModel.ResetOptimizerState();

                    // REMOVIDO: Descarte do DiskOnlyCacheManager, pois não existe mais.
                    Console.WriteLine("[Passo 3/5] Descartando modelo atual...");
                    currentModel.Dispose();
                    currentModel = null;

                    Console.WriteLine("[Passo 4/5] Forçando coleta de lixo em 3 estágios...");
                    ForceAggressiveGarbageCollection();

                    long memoryAfter = GetCurrentMemoryUsageMB();
                    long memoryFreed = memoryBefore - memoryAfter;

                    Console.ForegroundColor = memoryFreed > 0 ? ConsoleColor.Green : ConsoleColor.Yellow;
                    Console.WriteLine($"[Resultado] Memória ANTES: {memoryBefore}MB → DEPOIS: {memoryAfter}MB");
                    Console.WriteLine($"[Resultado] Memória LIBERADA: {memoryFreed}MB");
                    Console.ResetColor();

                    if (epoch < epochs - 1)
                    {
                        Console.WriteLine($"\n[Passo 5/5] Recarregando modelo limpo para Época {epoch + 2}...");

                        var vocabManager = new VocabularyManager();
                        vocabManager.LoadVocabulary();

                        // CORRIGIDO: Usa o método de fábrica estático 'Load' para recarregar o modelo.
                        currentModel = GenerativeNeuralNetworkLSTM.Load(modelPathForEpoch, _mathEngine, vocabManager, _searchService);
                        
                        if (currentModel == null)
                        {
                            throw new InvalidOperationException($"CRÍTICO: Falha ao recarregar o modelo {modelPathForEpoch}. Treinamento abortado.");
                        }

                        long memoryAfterReload = GetCurrentMemoryUsageMB();
                        Console.WriteLine($"[Recarga] Memória atual: {memoryAfterReload}MB");

                        var avgEpochTime = TimeSpan.FromMilliseconds(totalElapsedTime.TotalMilliseconds / (epoch + 1));
                        var estimatedTimeRemaining = TimeSpan.FromMilliseconds(avgEpochTime.TotalMilliseconds * (epochs - epoch - 1));
                        Console.WriteLine($"[Estimativa] Tempo restante: ~{estimatedTimeRemaining:hh\\:mm\\:ss}");
                    }
                    else
                    {
                        Console.WriteLine("\n[Passo 5/5] Última época concluída. Não é necessário recarregar.");
                    }
                    Console.WriteLine($"╚════════════════════════════════════════════════════════════╝\n");
                }
                Console.WriteLine("\nTREINAMENTO CONCLUÍDO COM SUCESSO!");
            }
        }

        private void ForceAggressiveGarbageCollection()
        {
            GC.Collect(2, GCCollectionMode.Forced, true, true);
            GC.WaitForPendingFinalizers();
            GC.Collect(2, GCCollectionMode.Forced, true, true);
        }

        private long GetCurrentMemoryUsageMB()
        {
            _currentProcess.Refresh();
            long currentMemory = _currentProcess.WorkingSet64 / (1024 * 1024);
            if (currentMemory > _peakMemoryUsageMB) _peakMemoryUsageMB = currentMemory;
            return currentMemory;
        }

        private double ValidateModel(GenerativeNeuralNetworkLSTM modelToValidate, DatasetService datasetService, List<int> validationBatchIndices)
        {
            Console.WriteLine("\n[Validação] Iniciando validação do modelo...");
            double totalLoss = 0;
            int sequenceCount = 0;
            Stopwatch validationStopwatch = Stopwatch.StartNew();

            foreach (var batchIndex in validationBatchIndices)
            {
                var batch = datasetService.LoadBatchFromDisk(batchIndex, isValidation: true);
                if (batch == null || !batch.Any()) continue;

                foreach (var (inputIndices, targetIndices) in batch)
                {
                    // O método CalculateSequenceLoss na classe do modelo agora lida com o estado e a limpeza.
                    totalLoss += modelToValidate.CalculateSequenceLoss(inputIndices, targetIndices);
                    sequenceCount++;
                }
            }

            validationStopwatch.Stop();
            Console.WriteLine($"\r[Validação] Concluída em {validationStopwatch.Elapsed:mm\\:ss} | RAM: {GetCurrentMemoryUsageMB()}MB");

            return sequenceCount > 0 ? totalLoss / sequenceCount : double.PositiveInfinity;
        }
    }
}