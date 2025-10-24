using Galileu.Node.Interfaces;
using Galileu.Node.Services;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Text.Json;
using Galileu.Node.Core;
using System.Diagnostics;
using Galileu.Node.Gpu;
using System.Text.Json.Serialization;
using Galileu.Node.Data;

namespace Galileu.Node.Brain;

/// <summary>
/// Estrutura para o cache do dataset SFT já tokenizado, armazenando IDs em vez de texto.
/// </summary>
public record SftSampleTokenizedCache(
    [property: JsonPropertyName("input_ids")] int[] InputIds,
    [property: JsonPropertyName("output_ids")] int[] OutputIds
);

/// <summary>
/// Orquestra o treinamento híbrido. Utiliza um cache de IDs de tokens para os dados SFT
/// para garantir robustez e consistência, e gerencia o ciclo de vida do modelo
/// para estabilidade de memória a longo prazo.
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

        var vocabManager = initialModel.VocabularyManager;
        var vocabulary = vocabManager.Vocab;
        string sftCacheFilePath = Path.Combine(Environment.CurrentDirectory, "Dayson", "sft_synthetic_dataset.json");
        
        List<(int InputIndex, int TargetIndex)> allSamples = new();

        if (File.Exists(sftCacheFilePath) && new FileInfo(sftCacheFilePath).Length > 0)
        {
            Console.WriteLine($"[HybridTrainer] Dataset SFT TOKENIZADO encontrado em disco. Carregando...");
            try
            {
                var jsonString = await File.ReadAllTextAsync(sftCacheFilePath);
                var tokenizedData = JsonSerializer.Deserialize<List<GeneratedExample>>(jsonString);
                
                if (tokenizedData != null)
                {
                    foreach(var sample in tokenizedData)
                    {
                        var sequence = sample.Input.Concat(sample.Output).ToArray();
                        if (sequence.Length < 2) continue;
                        for (int i = 0; i < sequence.Length - 1; i++)
                        {
                            allSamples.Add((sequence[i], sequence[i+1]));
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"[AVISO] Falha ao carregar cache tokenizado: {ex.Message}. Gerando novamente.");
                allSamples.Clear();
            }
            Console.WriteLine($"[HybridTrainer] {allSamples.Count} amostras de treinamento geradas do cache.");
        }

        if (allSamples.Count == 0)
        {
            Console.WriteLine("[HybridTrainer] Gerando e tokenizando dataset SFT via Teacher Model...");
            var syntheticDataText = await _teacherService.GenerateSyntheticDataAsync(vocabulary.Keys);
            
            var tokenizedDataToCache = new List<SftSampleTokenizedCache>();
            int unkIndex = vocabulary["<UNK>"];

            foreach (var pair in syntheticDataText)
            {
                var inputTokens = vocabManager.Tokenize(pair.input);
                var outputTokens = vocabManager.Tokenize(pair.output);

                var inputIds = inputTokens.Select(t => vocabulary.TryGetValue(t, out int id) ? id : unkIndex).ToArray();
                var outputIds = outputTokens.Select(t => vocabulary.TryGetValue(t, out int id) ? id : unkIndex).ToArray();
                
                tokenizedDataToCache.Add(new SftSampleTokenizedCache(inputIds, outputIds));

                var sequence = inputIds.Concat(outputIds).ToArray();
                if (sequence.Length < 2) continue;
                for (int i = 0; i < sequence.Length - 1; i++)
                {
                    allSamples.Add((sequence[i], sequence[i+1]));
                }
            }
            
            var jsonString = JsonSerializer.Serialize(tokenizedDataToCache, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(sftCacheFilePath, jsonString);
            Console.WriteLine($"[HybridTrainer] Dataset tokenizado e salvo em: {Path.GetFileName(sftCacheFilePath)}");
        }

        if (allSamples.Count == 0)
        {
             throw new InvalidOperationException("Nenhuma amostra de treinamento foi gerada. O treinamento não pode continuar.");
        }

        var rnd = new Random(42);
        allSamples = allSamples.OrderBy(x => rnd.Next()).ToList();
        int validationCount = (int)(allSamples.Count * validationSplit);
        var validationSamples = allSamples.Take(validationCount).ToList();
        var trainingSamples = allSamples.Skip(validationCount).ToList();
        
        const double QUALITY_THRESHOLD = 0.6;
        const double PERFORMANCE_CHECKPOINT_THRESHOLD = 0.7;
        const double MICRO_LEARNING_RATE = 0.0001;

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
                sftTrainer.TrainSingleEpoch(currentModel, trainingSamples, validationSamples, learningRate, batchSize);
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
                    if (cosineSimilarity < QUALITY_THRESHOLD) { correctionsMade++; currentModel.CorrectiveFineTuningStep(prompt, referenceResponse, MICRO_LEARNING_RATE); } else { goodResponses++; }
                }
                double accuracyRate = prompts.Count > 0 ? (double)goodResponses / prompts.Count : 0;
                Console.WriteLine($"\nÉpoca {epoch + 1} (RLAIF) concluída. Taxa de Acerto: {accuracyRate:P1}. Correções: {correctionsMade}.");
                
                if (accuracyRate >= PERFORMANCE_CHECKPOINT_THRESHOLD)
                {
                    string snapshotDir = Path.Combine(Environment.CurrentDirectory, "Dayson", "snapshots");
                    Directory.CreateDirectory(snapshotDir);
                    string snapshotPath = Path.Combine(snapshotDir, $"model_epoch-{epoch + 1}_accuracy-{(int)(accuracyRate * 100)}pct.json");
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine($"\n>>> PERFORMANCE ATINGIDA! Salvando snapshot em: {Path.GetFileName(snapshotPath)}");
                    Console.ResetColor();
                    currentModel.SaveModel(snapshotPath);
                }

                _currentProcess.Refresh();
                memoryBeforeRelease = _currentProcess.WorkingSet64 / (1024 * 1024);
            }

            string epochModelPath = Path.Combine(Path.GetDirectoryName(finalModelPath)!, $"dayson_epoch_{epoch + 1}.json");
            currentModel = PerformFullMemoryReleaseAndReload(currentModel, epochModelPath, memoryBeforeRelease, epoch + 1, totalEpochs);
            if (currentModel == null) break;
        }

        _memoryValidator.GenerateFinalReport();
        Console.WriteLine("\n[HybridTrainer] Treinamento híbrido contínuo concluído com sucesso!");
    }

    private GenerativeNeuralNetworkLSTM? PerformFullMemoryReleaseAndReload(
        GenerativeNeuralNetworkLSTM? modelToRelease,
        string modelPathForEpoch,
        long memoryBeforeRelease,
        int epochNumber,
        int totalEpochs)
    {
        if (modelToRelease == null) return null;

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

        if (epochNumber < totalEpochs)
        {
            Console.WriteLine($"\n[Passo 7/7] Recarregando modelo limpo para a próxima época...");
            var baseModel = NeuralNetworkLSTM.LoadModel(modelPathForEpoch, _mathEngine);
            if (baseModel == null) throw new InvalidOperationException($"CRÍTICO: Falha ao recarregar modelo.");

            var newVocabManager = new VocabularyManager();
            newVocabManager.LoadVocabulary();

            var newModel = new GenerativeNeuralNetworkLSTM(baseModel, newVocabManager, null);
            newModel._cacheManager = new DiskOnlyCacheManager(_mathEngine, newModel.weightsEmbedding!.Shape[1], newModel.HiddenSize);
            
            _currentProcess.Refresh();
            Console.WriteLine($"[Recarga] Memória atual: {_currentProcess.WorkingSet64 / (1024*1024)}MB");
            return newModel;
        }
        
        Console.WriteLine("\n[Treinamento] Última época concluída. Não é necessário recarregar.");
        return null;
    }

    private List<string> GenerateRlaifPrompts(List<string> vocabulary, int count)
    {
        var prompts = new List<string>();
        var random = new Random();
        for (int i = 0; i < count; i++)
        {
            if (vocabulary.Count < 2) continue;
            var validTokens = vocabulary.Where(v => !v.Contains('<') && !v.Contains('>') && v.Length > 2).ToList();
            if (validTokens.Count < 2) continue;

            string token1 = validTokens[random.Next(validTokens.Count)];
            string token2 = validTokens[random.Next(validTokens.Count)];
            while (token1 == token2)
            {
                token2 = validTokens[random.Next(validTokens.Count)];
            }
            prompts.Add($"compare e contraste os conceitos de {token1} e {token2}");
        }
        return prompts;
    }
}