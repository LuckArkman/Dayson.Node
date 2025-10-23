using Galileu.Node.Interfaces;
using Galileu.Node.Services;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Text.Json;
using Galileu.Node.Data;
using MongoDB.Bson;
using System.Diagnostics;
using System.Text.Json.Serialization;
using Galileu.Node.Gpu;
using Galileu.Node.Core;

namespace Galileu.Node.Brain;

/// <summary>
/// Orquestra um processo de treinamento híbrido e CONTÍNUO, com lógica de cache
/// de disco e um ciclo de liberação de memória centralizado e robusto.
/// </summary>
public class HybridTrainer
{
    private readonly IMathEngine _mathEngine;
    private readonly TeacherModelService _teacherService;
    private readonly MemoryValidationHelper _memoryValidator;
    private readonly Process _currentProcess;

    public HybridTrainer(IMathEngine mathEngine, TeacherModelService teacherService)
    {
        _mathEngine = mathEngine;
        _teacherService = teacherService;
        _currentProcess = Process.GetCurrentProcess();
        _memoryValidator = new MemoryValidationHelper(Path.Combine(Environment.CurrentDirectory, "Dayson", "memory_validation.txt"));
    }

    public async Task TrainModelAsync(
        GenerativeNeuralNetworkLSTM initialModel,
        string finalModelPath,
        double learningRate,
        int totalEpochs,
        int sftEpochs,
        int batchSize,
        double validationSplit)
    {
        Console.WriteLine("\n" + new string('=', 80));
        Console.WriteLine("INICIANDO PROCESSO DE TREINAMENTO HÍBRIDO CONTÍNUO");
        Console.WriteLine($"  - Total de Épocas: {totalEpochs}, Épocas SFT: {sftEpochs}");
        Console.WriteLine(new string('=', 80));

        var vocabulary = initialModel.VocabularyManager.Vocab;
        string sftCacheFilePath = Path.Combine(Environment.CurrentDirectory, "Dayson", "sft_synthetic_dataset.json");
        
        List<(string input, string output)> syntheticData = new();

        if (File.Exists(sftCacheFilePath) && new FileInfo(sftCacheFilePath).Length > 0)
        {
            Console.WriteLine($"[HybridTrainer] Dataset SFT (JSON) encontrado em disco. Carregando...");
            try
            {
                var jsonString = await File.ReadAllTextAsync(sftCacheFilePath);
                var cachedData = JsonSerializer.Deserialize<List<SftSampleCache>>(jsonString);
                syntheticData = cachedData?.Select(c => (c.Input, c.Output)).ToList() ?? new List<(string, string)>();
            }
            catch (Exception ex) when (ex is JsonException || ex is IOException)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"[AVISO] Não foi possível ler o cache JSON: {ex.Message}. O dataset será gerado novamente.");
                Console.ResetColor();
            }
            Console.WriteLine($"[HybridTrainer] {syntheticData.Count} exemplos carregados do cache local.");
        }

        if (syntheticData.Count == 0)
        {
            Console.WriteLine("[HybridTrainer] Gerando dataset SFT via Teacher Model (API)...");
            syntheticData = await _teacherService.GenerateSyntheticDataAsync(vocabulary.Keys);

            var dataToSerialize = syntheticData.Select(p => new SftSampleCache(p.input, p.output)).ToList();
            var jsonString = JsonSerializer.Serialize(dataToSerialize, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(sftCacheFilePath, jsonString);
            Console.WriteLine($"[HybridTrainer] Dataset sintético para SFT gerado e salvo em: {sftCacheFilePath}");
        }

        if (syntheticData.Count == 0)
        {
            throw new InvalidOperationException("Falha ao gerar ou carregar o dataset sintético. O treinamento não pode continuar.");
        }
        
        string sftDatasetPath = Path.Combine(Environment.CurrentDirectory, "Dayson", "sft_synthetic_dataset_flat.txt");
        await File.WriteAllLinesAsync(sftDatasetPath, syntheticData.SelectMany(p => new[] { p.input, p.output }));
        Console.WriteLine($"[HybridTrainer] Dataset sintético plano pronto com {syntheticData.Count} exemplos.");

        // 🔥 CORREÇÃO: Criação única do DatasetService para as épocas SFT
        using var sftDatasetService = new DatasetService(Path.Combine(Environment.CurrentDirectory, "Dayson", "sft_memory.bin"));
        sftDatasetService.InitializeAndSplit(File.ReadAllText(sftDatasetPath), 5, vocabulary, "<PAD>", batchSize, validationSplit);

        const double QUALITY_THRESHOLD = 0.6;
        const double PERFORMANCE_CHECKPOINT_THRESHOLD = 0.7;
        const double MICRO_LEARNING_RATE = 0.0001;
        // var metricsLogPath = Path.Combine(Environment.CurrentDirectory, "Dayson", "hybrid_training_metrics.csv");
        // using var metricsLogger = new MetricsLogger(metricsLogPath);

        GenerativeNeuralNetworkLSTM? currentModel = initialModel;

        for (int epoch = 0; epoch < totalEpochs; epoch++)
        {
            Console.WriteLine($"\n{'═',60}");
            Console.WriteLine($"ÉPOCA CONTÍNUA {epoch + 1}/{totalEpochs} >> {DateTime.UtcNow}");
            Console.WriteLine($"{'═',60}");

            _memoryValidator.RecordEpochStart(epoch + 1);
            long memoryBeforeRelease;

            if (epoch < sftEpochs)
            {
                Console.WriteLine($"[MODO: Destilação Supervisionada (SFT)]");
                var sftTrainer = new ModelTrainerLSTM(_mathEngine);
                // 🔥 CORREÇÃO: Passa o DatasetService já inicializado
                sftTrainer.TrainSingleEpoch(currentModel, sftDatasetService, learningRate);
                _currentProcess.Refresh();
                memoryBeforeRelease = _currentProcess.WorkingSet64 / (1024 * 1024);
            }
            else
            {
                Console.WriteLine($"[MODO: Reforço e Correção Direcionada (RLAIF)]");
                var prompts = GenerateRlaifPrompts(vocabulary.Keys.ToList(), 100);
                int correctionsMade = 0;
                int goodResponses = 0;

                for (int i = 0; i < prompts.Count; i++)
                {
                    var prompt = prompts[i];
                    Console.Write($"\rProcessando prompt RLAIF {i + 1}/{prompts.Count}...");
                    string candidateResponse = currentModel.GenerateResponse(prompt, maxLength: 30);
                    string referenceResponse = await _teacherService.GetReferenceResponseAsync(prompt, new HashSet<string>(vocabulary.Keys));
                    
                    if (string.IsNullOrEmpty(referenceResponse) || referenceResponse == "não sei") continue;

                    double cosineSimilarity = TextMetricsCalculator.CalculateCosineSimilarity(candidateResponse, referenceResponse, vocabulary, currentModel.weightsEmbedding!);
                    // metricsLogger.LogMetrics(...)
                    
                    if (cosineSimilarity < QUALITY_THRESHOLD)
                    {
                        correctionsMade++;
                        currentModel.CorrectiveFineTuningStep(prompt, referenceResponse, MICRO_LEARNING_RATE);
                    }
                    else
                    {
                        goodResponses++;
                    }
                }
                double accuracyRate = prompts.Count > 0 ? (double)goodResponses / prompts.Count : 0;
                Console.WriteLine($"\nÉpoca {epoch + 1} (RLAIF) concluída. Taxa de Acerto: {accuracyRate:P1}. Correções: {correctionsMade}.");
                _currentProcess.Refresh();
                memoryBeforeRelease = _currentProcess.WorkingSet64 / (1024 * 1024);
            }

            string epochModelPath = Path.Combine(Path.GetDirectoryName(finalModelPath)!, $"dayson_epoch_{epoch + 1}.json");
            currentModel = PerformFullMemoryReleaseAndReload(currentModel, epochModelPath, memoryBeforeRelease, epoch + 1);
        }

        _memoryValidator.GenerateFinalReport();
        Console.WriteLine("\n[HybridTrainer] Treinamento híbrido contínuo concluído com sucesso!");
    }

    private GenerativeNeuralNetworkLSTM PerformFullMemoryReleaseAndReload(
        GenerativeNeuralNetworkLSTM modelToRelease,
        string modelPathForEpoch,
        long memoryBeforeRelease,
        int epochNumber)
    {
        Console.WriteLine($"\n╔════════════════════════════════════════════════════════════╗");
        Console.WriteLine($"║  INICIANDO LIBERAÇÃO COMPLETA DE MEMÓRIA (ÉPOCA {epochNumber})      ║");
        Console.WriteLine($"╚════════════════════════════════════════════════════════════╝");

        Console.WriteLine($"[Passo 1/7] Salvando modelo em {modelPathForEpoch}...");
        modelToRelease.SaveModel(modelPathForEpoch);

        if (_mathEngine.IsGpu)
        {
            Console.WriteLine("[Passo 1.5/7] Sincronizando fila de comandos da GPU...");
            (_mathEngine as GpuMathEngine)?.Synchronize();
        }

        var cacheManager = modelToRelease._cacheManager;
        try { Console.WriteLine("[Passo 2/7] Limpando TensorPool..."); modelToRelease._tensorPool?.Dispose(); } catch (Exception ex) { Console.WriteLine($"  [AVISO] Erro ao descartar TensorPool: {ex.Message}"); }
        try { Console.WriteLine("[Passo 3/7] Resetando AdamOptimizer..."); modelToRelease.ResetOptimizerState(); } catch (Exception ex) { Console.WriteLine($"  [AVISO] Erro ao resetar AdamOptimizer: {ex.Message}"); }
        try { Console.WriteLine("[Passo 4/7] Descartando DiskOnlyCacheManager..."); cacheManager?.Dispose(); } catch (Exception ex) { Console.WriteLine($"  [AVISO] Erro ao descartar CacheManager: {ex.Message}"); }
        try { Console.WriteLine("[Passo 5/7] Descartando modelo principal..."); modelToRelease.Dispose(); } catch (Exception ex) { Console.WriteLine($"  [AVISO] Erro ao descartar o modelo: {ex.Message}"); }
        
        modelToRelease = null;

        Console.WriteLine("[Passo 6/7] Forçando coleta de lixo em 3 estágios...");
        GC.Collect(2, GCCollectionMode.Forced, true, true); GC.WaitForPendingFinalizers();
        GC.Collect(2, GCCollectionMode.Forced, true, true); GC.WaitForPendingFinalizers();
        GC.Collect(2, GCCollectionMode.Forced, true, true); GC.WaitForPendingFinalizers();

        _currentProcess.Refresh();
        long memoryAfter = _currentProcess.WorkingSet64 / (1024 * 1024);
        long memoryFreed = memoryBeforeRelease - memoryAfter;
        Console.ForegroundColor = memoryFreed > 0 ? ConsoleColor.Green : ConsoleColor.Yellow;
        Console.WriteLine($"[Resultado] Memória ANTES: {memoryBeforeRelease}MB → DEPOIS: {memoryAfter}MB");
        Console.WriteLine($"[Resultado] Memória LIBERADA: {memoryFreed}MB");
        Console.ResetColor();

        _memoryValidator.ValidateMemoryRelease(epochNumber);

        Console.WriteLine($"\n[Passo 7/7] Recarregando modelo limpo para a próxima época...");
        var baseModel = NeuralNetworkLSTM.LoadModel(modelPathForEpoch, _mathEngine);
        if (baseModel == null) throw new InvalidOperationException($"CRÍTICO: Falha ao recarregar modelo.");

        // 🔥 CORREÇÃO: Cria um novo VocabularyManager para o modelo recarregado.
        var newVocabManager = new VocabularyManager();
        newVocabManager.LoadVocabulary();

        var newModel = new GenerativeNeuralNetworkLSTM(baseModel, newVocabManager, null);
        newModel._cacheManager = new DiskOnlyCacheManager(_mathEngine, newModel.weightsEmbedding!.Shape[1], newModel.HiddenSize);
        
        _currentProcess.Refresh();
        Console.WriteLine($"[Recarga] Memória atual: {_currentProcess.WorkingSet64 / (1024*1024)}MB");
        return newModel;
    }

    private List<string> GenerateRlaifPrompts(List<string> vocabulary, int count)
    {
        var prompts = new List<string>();
        var random = new Random();
        for (int i = 0; i < count; i++)
        {
            if (vocabulary.Count < 2) continue;
            string token1 = vocabulary[random.Next(vocabulary.Count)];
            string token2 = vocabulary[random.Next(vocabulary.Count)];
            prompts.Add($"compare e contraste os conceitos de {token1} e {token2}");
        }
        return prompts;
    }
}

/// <summary>
/// Estrutura auxiliar para serializar pares de dados de treinamento SFT no cache local JSON.
/// </summary>
public record SftSampleCache(
    [property: JsonPropertyName("input")] string Input,
    [property: JsonPropertyName("output")] string Output
);