using Galileu.Node.Brain;
using Galileu.Node.Core;
using Galileu.Node.Interfaces;
using Galileu.Node.Models;
using System.IO;
using System.Threading.Tasks;
using Galileu.Node.Cpu;
using Galileu.Node.Gpu;

namespace Galileu.Node.Services;

public class GenerativeService
{
    private readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Dayson", "Dayson.json");
    private readonly ISearchService _searchService = new MockSearchService();
    private readonly IMathEngine _mathEngine;
    private readonly PrimingService _primingService;
    private GenerativeNeuralNetworkLSTM? _model;
    public bool IsModelLoaded => _model != null;

    public GenerativeService(PrimingService primingService)
    {
        _primingService = primingService;
        try
        {
            _mathEngine = new GpuMathEngine();
            Console.WriteLine("[GenerativeService] Usando GpuMathEngine para aceleração.");
        }
        catch (Exception)
        {
            _mathEngine = new CpuMathEngine();
            Console.WriteLine("[GenerativeService] Usando CpuMathEngine como fallback.");
        }
    }

    public async Task TrainModelAsync(Trainer trainerOptions, int vocabSize, int embeddingSize, int hiddenSize, int contextWindow)
    {
        if (!File.Exists(trainerOptions.datasetPath))
        {
            throw new FileNotFoundException($"Arquivo de dataset não encontrado em: {trainerOptions.datasetPath}");
        }

        await Task.Run(() =>
        {
            Console.WriteLine(
                $"[GenerativeService] Arquitetura: Vocab={vocabSize}, Emb={embeddingSize}, Hidden={hiddenSize}");

            var trainingModel = new GenerativeNeuralNetworkLSTM(
                vocabSize,
                embeddingSize,
                hiddenSize,
                trainerOptions.datasetPath,
                _searchService,
                _mathEngine
            );

            Console.WriteLine("\n" + new string('═', 80));
            Console.WriteLine("EXECUTANDO VALIDAÇÃO DE INTEGRIDADE DO MODELO");
            Console.WriteLine(new string('═', 80));

            try
            {
                trainingModel.RunSanityCheck();
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("\n╔════════════════════════════════════════════════════════════╗");
                Console.WriteLine("║  FALHA NO SANITY CHECK - TREINAMENTO ABORTADO             ║");
                Console.WriteLine("╚════════════════════════════════════════════════════════════╝");
                Console.WriteLine($"\nMotivo: {ex.Message}");
                Console.ResetColor();
                throw;
            }

            Console.WriteLine(new string('═', 80) + "\n");

            long modelMemoryMB = CalculateModelMemoryMB(vocabSize, embeddingSize, hiddenSize);
            Console.WriteLine($"[GenerativeService] Memória estimada do modelo: ~{modelMemoryMB}MB");

            var modelTrainer = new ModelTrainerLSTM(_mathEngine);

            Console.WriteLine("\n[GenerativeService] Iniciando treinamento com otimizações de memória...");

            modelTrainer.TrainModel(
                trainingModel,
                trainerOptions.datasetPath,
                _modelPath,
                trainerOptions.learningRate,
                trainerOptions.epochs,
                trainerOptions.batchSize,
                contextWindow,
                trainerOptions.validationSplit
            );

            Console.WriteLine($"\n[GenerativeService] Salvando modelo final em {_modelPath}...");
            trainingModel.SaveModel(_modelPath);
            Console.WriteLine($"[GenerativeService] Modelo salvo com sucesso!");

            _model = trainingModel;

            Console.WriteLine("[GenerativeService] Executando limpeza pós-treinamento...");
            ForceAggressiveGarbageCollection();
        });
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

    private long CalculateModelMemoryMB(int vocabSize, int embeddingSize, int hiddenSize)
    {
        long totalParams = 0;

        // Embedding layer
        totalParams += vocabSize * embeddingSize;

        // LSTM weights (4 gates × input + hidden)
        totalParams += 4 * (embeddingSize * hiddenSize); // Input weights
        totalParams += 4 * (hiddenSize * hiddenSize); // Hidden weights
        totalParams += 4 * hiddenSize; // Biases

        // Output layer
        totalParams += hiddenSize * vocabSize;
        totalParams += vocabSize;

        // Cada parâmetro: 8 bytes (float) + Adam state (2× float) = 24 bytes
        long bytesPerParam = 24;
        long totalBytes = totalParams * bytesPerParam;

        return totalBytes / (1024 * 1024);
    }

    public async Task<string?> GenerateAsync(GenerateResponse generateResponse)
    {
        if (_model == null) return "Erro: O modelo não está carregado.";
        return await Task.Run(() => _model.GenerateResponse(generateResponse.input, maxLength: 50));
    }

    public void InitializeFromDisk()
    {
        if (!File.Exists(_modelPath))
        {
            Console.WriteLine($"[GenerativeService] Modelo não encontrado em {_modelPath}");
            return;
        }

        try
        {
            Console.WriteLine($"[GenerativeService] Carregando modelo de {_modelPath}...");
            _model = ModelSerializerLSTM.LoadModel(_modelPath, _mathEngine);

            if (_model != null)
            {
                Console.WriteLine("[GenerativeService] Modelo carregado com sucesso!");
            }
            else
            {
                Console.WriteLine("[GenerativeService] Falha ao carregar modelo.");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GenerativeService] Erro ao carregar modelo: {ex.Message}");
        }
    }
}