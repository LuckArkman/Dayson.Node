using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Galileu.Node.Core;
using Galileu.Node.Gpu;
using Galileu.Node.Interfaces;
using Galileu.Node.Services;

namespace Galileu.Node.Brain;

/// <summary>
/// Orquestra o processo de treinamento de um modelo GenerativeNeuralNetworkLSTM.
/// VERSÃO CORRIGIDA: Implementa uma cadeia de descarte "à prova de falhas" para garantir
/// que todos os recursos, gerenciados e não gerenciados, sejam liberados mesmo que
/// uma das etapas de limpeza encontre um erro.
/// </summary>
public class ModelTrainerLSTM
{
    private readonly IMathEngine _mathEngine;
    private readonly Stopwatch _stopwatch = new Stopwatch();
    private readonly Process _currentProcess;
    private long _peakMemoryUsageMB = 0;
    private readonly string logPath = Path.Combine(Environment.CurrentDirectory, "Dayson", "training_log.txt");
    private readonly string memoryLogPath = Path.Combine(Environment.CurrentDirectory, "Dayson", "memory_validation.txt");

    public ModelTrainerLSTM(IMathEngine mathEngine)
    {
        _mathEngine = mathEngine ?? throw new ArgumentNullException(nameof(mathEngine));
        _currentProcess = Process.GetCurrentProcess();
    }

    public void TrainModel(
    GenerativeNeuralNetworkLSTM initialModel,
    string datasetPath,
    string finalModelPath,
    double learningRate,
    int epochs,
    int batchSize,
    int contextWindowSize,
    double validationSplit)
    {
        if (!File.Exists(datasetPath))
            throw new FileNotFoundException("Arquivo de dataset não encontrado.", datasetPath);
        
        var swapFilePath = Path.Combine(Environment.CurrentDirectory, "Dayson", "memory.bin");
        using (var datasetService = new DatasetService(swapFilePath))
        {
            datasetService.InitializeAndSplit(text: File.ReadAllText(datasetPath), contextWindowSize,
                initialModel.VocabularyManager.Vocab, "<PAD>", batchSize, validationSplit);

            Console.WriteLine($"\n[Trainer] Configuração do Ciclo de Treinamento com Liberação Total de Memória:");
            Console.WriteLine($"  - Estratégia: Salvar → Descartar TUDO → GC Agressivo → Recarregar");
            Console.WriteLine($"  - Objetivo: 100% de liberação de memória garantida entre épocas.\n");

            GenerativeNeuralNetworkLSTM? currentModel = initialModel;
            DiskOnlyCacheManager? currentCacheManager = initialModel._cacheManager;
            TimeSpan totalElapsedTime = TimeSpan.Zero;
            
            var memoryValidator = new MemoryValidationHelper(memoryLogPath);

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                memoryValidator.RecordEpochStart(epoch + 1);
                
                _stopwatch.Restart();
                Console.WriteLine($"\n{'═',60}");
                Console.WriteLine($"ÉPOCA {epoch + 1}/{epochs} >> Learning Rate : {learningRate} >> {DateTime.UtcNow}");
                Console.WriteLine($"{'═',60}");
                
                double totalEpochLoss = 0;
                int batchCount = 0;
                datasetService.ResetTrain();

                while (true)
                {
                    var batch = datasetService.GetNextTrainChunk();
                    if (batch == null || batch.Count == 0) break;
                    
                    var sequenceInputIndices = batch.Select(p => p.InputIndex).ToArray();
                    var sequenceTargetIndices = batch.Select(p => p.TargetIndex).ToArray();

                    currentModel.ResetHiddenState();
                    currentModel._cacheManager.Reset();
                    
                    totalEpochLoss += currentModel.TrainSequence(sequenceInputIndices, sequenceTargetIndices, learningRate);
                    batchCount++;
                    Console.Write($"\rÉpoca: {epoch + 1}/{epochs} | Lotes: {batchCount} ...");
                    
                    
                }

                _stopwatch.Stop();
                totalElapsedTime += _stopwatch.Elapsed;
                double avgLoss = batchCount > 0 ? totalEpochLoss / batchCount : double.PositiveInfinity;
                string elapsedFormatted = string.Format("{0:D2}:{1:D2}:{2:D2}", 
                    (int)_stopwatch.Elapsed.TotalHours, _stopwatch.Elapsed.Minutes, _stopwatch.Elapsed.Seconds);
                
                File.AppendAllText(logPath, $"Época {epoch + 1}/{epochs} concluída. Perda média: {avgLoss} : Concluída em {elapsedFormatted}" + Environment.NewLine);
                Console.WriteLine($"\nÉpoca {epoch + 1}/{epochs} concluída. Perda média: {avgLoss:F4} | Duração: {elapsedFormatted}");
                
                double validationLoss = ValidateModel(currentModel, datasetService);
                File.AppendAllText(logPath, $"[Época {epoch + 1}] Perda Média de Validação: {validationLoss:F4}" + Environment.NewLine);
                Console.WriteLine($"[Época {epoch + 1}] Perda Média de Validação: {validationLoss:F4}");

                long memoryBefore = GetCurrentMemoryUsageMB();
                string modelPathForEpoch = Path.Combine(Path.GetDirectoryName(finalModelPath)!, $"dayson_{epoch + 1}.json");

                Console.WriteLine($"\n╔════════════════════════════════════════════════════════════╗");
                Console.WriteLine($"║  INICIANDO LIBERAÇÃO COMPLETA DE MEMÓRIA (ÉPOCA {epoch + 1})       ║");
                Console.WriteLine($"╚════════════════════════════════════════════════════════════╝");

                // PASSO 1: SALVAR O MODELO
                Console.WriteLine($"[Passo 1/7] Salvando modelo em {modelPathForEpoch}...");
                currentModel.SaveModel(modelPathForEpoch);

                // PASSO 1.5: SINCRONIZAR A GPU (ESSENCIAL)
                if (_mathEngine.IsGpu)
                {
                    Console.WriteLine("[Passo 1.5/7] Sincronizando fila de comandos da GPU...");
                    (_mathEngine as GpuMathEngine)?.Synchronize();
                }

                // ==============================================================
                // 🔥 CORREÇÃO: CADEIA DE DESCARTE À PROVA DE FALHAS
                // Cada Dispose é tentado individualmente para maximizar a liberação
                // de recursos, mesmo que uma das etapas falhe.
                // ==============================================================

                // PASSO 2: LIMPAR TENSOR POOL
                try
                {
                    Console.WriteLine("[Passo 2/7] Limpando TensorPool...");
                    currentModel._tensorPool?.Trim();
                    currentModel._tensorPool?.Dispose();
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine($"  [AVISO] Erro ao descartar TensorPool: {ex.Message}");
                    Console.ResetColor();
                }

                // PASSO 3: RESETAR OTIMIZADOR
                try
                {
                    Console.WriteLine("[Passo 3/7] Resetando estados do AdamOptimizer...");
                    currentModel.ResetOptimizerState();
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine($"  [AVISO] Erro ao resetar AdamOptimizer: {ex.Message}");
                    Console.ResetColor();
                }

                // PASSO 4: DESCARTAR CACHE MANAGER
                try
                {
                    Console.WriteLine("[Passo 4/7] Descartando DiskOnlyCacheManager...");
                    currentCacheManager?.Dispose();
                    currentCacheManager = null;
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine($"  [AVISO] Erro ao descartar DiskOnlyCacheManager: {ex.Message}");
                    Console.ResetColor();
                }
                
                // PASSO 5: DESCARTAR O MODELO ATUAL (LIBERA PESOS NA GPU)
                try
                {
                    Console.WriteLine("[Passo 5/7] Descartando modelo atual...");
                    currentModel.Dispose();
                    currentModel = null;
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine($"  [AVISO] Erro ao descartar o modelo principal: {ex.Message}");
                    Console.ResetColor();
                }

                // PASSO 6: FORÇAR COLETA DE LIXO
                Console.WriteLine("[Passo 6/7] Forçando coleta de lixo em 3 estágios...");
                ForceAggressiveGarbageCollection();

                long memoryAfter = GetCurrentMemoryUsageMB();
                long memoryFreed = memoryBefore - memoryAfter;
                
                Console.ForegroundColor = memoryFreed > 0 ? ConsoleColor.Green : ConsoleColor.Yellow;
                Console.WriteLine($"[Resultado] Memória ANTES: {memoryBefore}MB → DEPOIS: {memoryAfter}MB");
                Console.WriteLine($"[Resultado] Memória LIBERADA: {memoryFreed}MB ({(double)memoryFreed / memoryBefore * 100:F1}%)");
                Console.ResetColor();
                
                var validationResult = memoryValidator.ValidateMemoryRelease(epoch + 1);

                // PASSO 7: RECARREGAR MODELO
                if (epoch < epochs - 1)
                {
                    Console.WriteLine($"\n[Passo 7/7] Recarregando modelo limpo para Época {epoch + 2}...");
                    
                    var vocabManager = new VocabularyManager();
                    vocabManager.LoadVocabulary();

                    var baseModel = NeuralNetworkLSTM.LoadModel(modelPathForEpoch, _mathEngine);
                    if (baseModel == null)
                    {
                        throw new InvalidOperationException($"CRÍTICO: Falha ao recarregar o modelo {modelPathForEpoch}. Treinamento abortado.");
                    }
                    
                    currentModel = new GenerativeNeuralNetworkLSTM(baseModel, vocabManager, new MockSearchService());
                    
                    int embeddingSize = currentModel.weightsEmbedding!.Shape[1];
                    int hiddenSize = currentModel.HiddenSize;
                    currentModel._cacheManager = new DiskOnlyCacheManager(_mathEngine, embeddingSize, hiddenSize);
                    currentCacheManager = currentModel._cacheManager;
                    
                    long memoryAfterReload = GetCurrentMemoryUsageMB();
                    Console.WriteLine($"[Recarga] Memória atual: {memoryAfterReload}MB (Delta: +{memoryAfterReload - memoryAfter}MB)");

                    var avgEpochTime = TimeSpan.FromMilliseconds(totalElapsedTime.TotalMilliseconds / (epoch + 1));
                    var estimatedTimeRemaining = TimeSpan.FromMilliseconds(avgEpochTime.TotalMilliseconds * (epochs - epoch - 1));
                    Console.WriteLine($"[Estimativa] Tempo restante: ~{estimatedTimeRemaining:hh\\:mm\\:ss}");
                }
                else
                {
                    Console.WriteLine("\n[Passo 7/7] Última época concluída. Não é necessário recarregar.");
                }

                Console.WriteLine($"╚════════════════════════════════════════════════════════════╝\n");
            }
            
            memoryValidator.GenerateFinalReport();
            
            Console.WriteLine("\n═══════════════════════════════════════════════════════════");
            Console.WriteLine("          TREINAMENTO CONCLUÍDO COM SUCESSO!              ");
            Console.WriteLine($"          Pico de Memória: {_peakMemoryUsageMB}MB              ");
            Console.WriteLine("═══════════════════════════════════════════════════════════\n");
        }
    }

    private void ForceAggressiveGarbageCollection()
    {
        GC.Collect(2, GCCollectionMode.Forced, true, true);
        GC.WaitForPendingFinalizers();
        GC.Collect(2, GCCollectionMode.Forced, true, true);
        GC.WaitForPendingFinalizers();
        GC.Collect(2, GCCollectionMode.Forced, true, true);
        System.Threading.Thread.Sleep(500);
    }

    private long GetCurrentMemoryUsageMB()
    {
        _currentProcess.Refresh();
        long currentMemory = _currentProcess.WorkingSet64 / (1024 * 1024);
        if (currentMemory > _peakMemoryUsageMB)
        {
            _peakMemoryUsageMB = currentMemory;
        }
        return currentMemory;
    }

    private double ValidateModel(GenerativeNeuralNetworkLSTM modelToValidate, DatasetService datasetService)
    {
        Console.WriteLine("\n[Validação] Iniciando validação do modelo...");
        double totalLoss = 0;
        int batchCount = 0;
        datasetService.ResetValidation();
        Stopwatch validationStopwatch = Stopwatch.StartNew();

        while (true)
        {
            var batch = datasetService.GetNextValidationChunk();
            if (batch == null || batch.Count == 0) break;

            var sequenceInputIndices = batch.Select(p => p.InputIndex).ToArray();
            var sequenceTargetIndices = batch.Select(p => p.TargetIndex).ToArray();

            totalLoss += modelToValidate.CalculateSequenceLoss(sequenceInputIndices, sequenceTargetIndices);
            batchCount++;
        
            Console.Write($"\r[Validação] Processando lote {batchCount}...");
            
            modelToValidate._cacheManager.Reset();
        }

        validationStopwatch.Stop();
        Console.WriteLine($"\r[Validação] Concluída em {validationStopwatch.Elapsed:mm\\:ss} | RAM: {GetCurrentMemoryUsageMB()}MB");
    
        return batchCount > 0 ? totalLoss / batchCount : double.PositiveInfinity;
    }
}