using Galileu.Node.Interfaces;
using Galileu.Node.Services;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Galileu.Node.Data;
using MongoDB.Bson;

namespace Galileu.Node.Brain;

/// <summary>
/// Orquestra um processo de treinamento híbrido e CONTÍNUO, combinando
/// destilação de conhecimento (SFT) e reforço por feedback de IA (RLAIF),
/// com checkpointing automático baseado em performance.
/// </summary>
public class HybridTrainer
{
    private readonly IMathEngine _mathEngine;
    private readonly TeacherModelService _teacherService;

    public HybridTrainer(IMathEngine mathEngine, TeacherModelService teacherService)
    {
        _mathEngine = mathEngine;
        _teacherService = teacherService;
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
        Console.WriteLine($"  - Total de Épocas: {totalEpochs}");
        Console.WriteLine($"  - Épocas de Destilação (SFT): {sftEpochs}");
        Console.WriteLine($"  - Épocas de Reforço (RLAIF): {totalEpochs - sftEpochs}");
        Console.WriteLine(new string('=', 80));

        // --- PREPARAÇÃO INICIAL E VERIFICAÇÃO DE CACHE SFT (JSON) ---
        var vocabulary = initialModel.VocabularyManager.Vocab.Keys;
        string sftCacheFilePath = Path.Combine(Environment.CurrentDirectory, "Dayson", "sft_synthetic_dataset.json");
        List<(string input, string output)> syntheticData;

        if (File.Exists(sftCacheFilePath) && new FileInfo(sftCacheFilePath).Length > 0)
        {
            Console.WriteLine($"[HybridTrainer] Dataset SFT (JSON) encontrado em disco. Carregando...");
            
            try
            {
                var jsonString = await File.ReadAllTextAsync(sftCacheFilePath);
                
                // Desserializa a lista de GeneratedExample (formato MongoDB/Cache)
                var cachedData = JsonSerializer.Deserialize<List<GeneratedExample>>(jsonString);
                
                // Converte de volta para a estrutura de tupla (input, output) para uso interno
                syntheticData = cachedData?.Select(c => (c.Input, c.Output)).ToList() 
                                ?? new List<(string, string)>();
            }
            catch (Exception ex) when (ex is JsonException || ex is IOException)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[HybridTrainer] ERRO ao ler cache JSON: {ex.Message}. Forçando regeração.");
                Console.ResetColor();
                syntheticData = new List<(string, string)>(); // Força a regeração
            }
            
            Console.WriteLine($"[HybridTrainer] {syntheticData.Count} exemplos carregados do cache local.");
        }
        else
        {
            // Lógica de Geração via Teacher Model
            Console.WriteLine("[HybridTrainer] Dataset SFT não encontrado. Gerando via Teacher Model (API)...");
            
            syntheticData = await _teacherService.GenerateSyntheticDataAsync(vocabulary);

            // Salva os dados gerados em formato JSON (GeneratedExample)
            var dataToSerialize = syntheticData.Select(p => 
                new GeneratedExample(ObjectId.GenerateNewId(), p.input, p.output, DateTime.UtcNow)).ToList();
            var jsonString = JsonSerializer.Serialize(dataToSerialize, new JsonSerializerOptions { WriteIndented = true });
            
            await File.WriteAllTextAsync(sftCacheFilePath, jsonString);
            Console.WriteLine($"[HybridTrainer] Dataset sintético para SFT gerado e salvo em: {sftCacheFilePath}");
        }

        if (syntheticData.Count == 0)
        {
            throw new InvalidOperationException("Falha ao gerar o dataset sintético. O treinamento não pode continuar.");
        }
        
        // CRÍTICO: Cria o arquivo de dataset plano que o ModelTrainerLSTM espera
        string sftDatasetPath = Path.Combine(Environment.CurrentDirectory, "Dayson", "sft_synthetic_dataset_flat.txt");
        await File.WriteAllLinesAsync(sftDatasetPath, syntheticData.SelectMany(p => new[] { p.input, p.output }));
        Console.WriteLine($"[HybridTrainer] Dataset sintético pronto com {syntheticData.Count} exemplos.");


        const double QUALITY_THRESHOLD = 0.6; 
        const double PERFORMANCE_CHECKPOINT_THRESHOLD = 0.7; 
        const double MICRO_LEARNING_RATE = 0.0001;
        var metricsLogPath = Path.Combine(Environment.CurrentDirectory, "Dayson", "hybrid_training_metrics.csv");
        using var metricsLogger = new MetricsLogger(metricsLogPath);

        GenerativeNeuralNetworkLSTM? currentModel = initialModel;
        
        // --- LOOP DE TREINAMENTO CONTÍNUO ---
        for (int epoch = 0; epoch < totalEpochs; epoch++)
        {
            Console.WriteLine($"\n{'═',60}");
            Console.WriteLine($"ÉPOCA CONTÍNUA {epoch + 1}/{totalEpochs} >> {DateTime.UtcNow}");
            Console.WriteLine($"{'═',60}");

            if (epoch < sftEpochs)
            {
                // --- MODO SFT (DESTILAÇÃO) ---
                Console.WriteLine($"[MODO: Destilação Supervisionada (SFT)]");
                var sftTrainer = new ModelTrainerLSTM(_mathEngine);
                
                sftTrainer.TrainModel(
                    currentModel, sftDatasetPath, finalModelPath, learningRate, 1, // Usa o arquivo plano
                    batchSize, 5, validationSplit
                );

                string lastModelPath = Path.Combine(Path.GetDirectoryName(finalModelPath)!, "dayson_1.json");
                var baseModel = NeuralNetworkLSTM.LoadModel(lastModelPath, _mathEngine);
                currentModel = new GenerativeNeuralNetworkLSTM(baseModel, initialModel.VocabularyManager, null);
            }
            else
            {
                // --- MODO RLAIF (REFORÇO E CORREÇÃO) ---
                Console.WriteLine($"[MODO: Reforço e Correção Direcionada (RLAIF)]");
                
                if (currentModel._cacheManager == null)
                {
                    int embSize = currentModel.weightsEmbedding!.Shape[1];
                    int hidSize = currentModel.HiddenSize;
                    currentModel._cacheManager = new DiskOnlyCacheManager(_mathEngine, embSize, hidSize);
                }

                var prompts = GenerateRlaifPrompts(vocabulary.ToList(), 100);
                int correctionsMade = 0;
                int goodResponses = 0;
                
                for(int i = 0; i < prompts.Count; i++)
                {
                    var prompt = prompts[i];
                    Console.Write($"\rProcessando prompt {i + 1}/{prompts.Count}...");

                    string candidateResponse = currentModel.GenerateResponse(prompt, maxLength: 30);
                    
                    // NOVO: Usa GetReferenceResponseAsync, que inclui cache MongoDB e fallback
                    string referenceResponse = await _teacherService.GetReferenceResponseAsync($"Responda de forma concisa e precisa: {prompt}", new HashSet<string>(vocabulary));
                    
                    if (string.IsNullOrEmpty(referenceResponse) || referenceResponse == "não sei") continue;

                    double cosineSimilarity = TextMetricsCalculator.CalculateCosineSimilarity(
                        candidateResponse, referenceResponse, currentModel.VocabularyManager.Vocab, currentModel.weightsEmbedding!
                    );

                    metricsLogger.LogMetrics(new MetricsRecord(epoch + 1, i + 1, prompt, referenceResponse, 
                        candidateResponse, TextMetricsCalculator.CalculateBleuScore(candidateResponse, referenceResponse), 
                        TextMetricsCalculator.CalculateRougeLScore(candidateResponse, referenceResponse), cosineSimilarity));

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
                
                double accuracyRate = (double)goodResponses / prompts.Count;
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
                
                string epochModelPath = Path.Combine(Path.GetDirectoryName(finalModelPath)!, $"dayson_epoch_{epoch + 1}.json");
                currentModel.SaveModel(epochModelPath);
                
                // Libera e recarrega para a próxima época de RLAIF
                currentModel.Dispose();
                var reloadedBase = NeuralNetworkLSTM.LoadModel(epochModelPath, _mathEngine);
                currentModel = new GenerativeNeuralNetworkLSTM(reloadedBase, initialModel.VocabularyManager, null);
            }
        }
        
        string finalEpochModel = Path.Combine(Path.GetDirectoryName(finalModelPath)!, $"dayson_epoch_{totalEpochs}.json");
        if (File.Exists(finalEpochModel))
        {
            File.Copy(finalEpochModel, finalModelPath, overwrite: true);
        }

        Console.WriteLine("\n[HybridTrainer] Treinamento híbrido contínuo concluído com sucesso!");
    }

    private List<string> GenerateRlaifPrompts(List<string> vocabulary, int count)
    {
        var prompts = new List<string>();
        var random = new Random();
        for (int i = 0; i < count; i++)
        {
            string token1 = vocabulary[random.Next(vocabulary.Count)];
            string token2 = vocabulary[random.Next(vocabulary.Count)];
            prompts.Add($"compare e contraste os conceitos de {token1} e {token2}");
        }
        return prompts;
    }
}