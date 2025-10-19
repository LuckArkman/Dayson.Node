using Galileu.Node.Brain;
using Galileu.Node.Core;
using Galileu.Node.Interfaces;
using Galileu.Node.Models;
using Galileu.Node.Cpu;
using Galileu.Node.Gpu;
using System;
using System.IO;
using System.Threading.Tasks;

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

    public async Task TrainModelAsync(Trainer trainerOptions)
    {
        if (!File.Exists(trainerOptions.datasetPath))
        {
            throw new FileNotFoundException($"Arquivo de dataset não encontrado em: {trainerOptions.datasetPath}");
        }

        await Task.Run(() =>
        {
            const int VOCAB_SIZE = 20000;
            const int EMBEDDING_SIZE = 128;
            const int HIDDEN_SIZE = 196;
            const int CONTEXT_WINDOW = 5;

            GenerativeNeuralNetworkLSTM? initialModel = null;
            try
            {
                Console.WriteLine("[GenerativeService] Inicializando modelo LSTM para treinamento padrão...");
                initialModel = new GenerativeNeuralNetworkLSTM(
                    VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE,
                    trainerOptions.datasetPath, _searchService, _mathEngine
                );
                initialModel._cacheManager = new DiskOnlyCacheManager(_mathEngine, EMBEDDING_SIZE, HIDDEN_SIZE);
                initialModel.RunSanityCheck();

                var trainer = new ModelTrainerLSTM(_mathEngine);
                Console.WriteLine("\n[GenerativeService] Iniciando treinamento com liberação total de memória...");

                trainer.TrainModel(
                    initialModel, trainerOptions.datasetPath, _modelPath, trainerOptions.learningRate,
                    trainerOptions.epochs, trainerOptions.batchSize, CONTEXT_WINDOW, trainerOptions.validationSplit
                );

                Console.WriteLine($"\n[GenerativeService] Treinamento concluído! Carregando modelo final...");
                string lastEpochModelPath = Path.Combine(Path.GetDirectoryName(_modelPath)!, $"dayson_{trainerOptions.epochs}.json");
                if (File.Exists(lastEpochModelPath))
                {
                    File.Copy(lastEpochModelPath, _modelPath, overwrite: true);
                }
                InitializeFromDisk();
                initialModel = null;
                ForceAggressiveGarbageCollection();
                Console.WriteLine("[GenerativeService] Serviço pronto para inferência!");
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"\n[ERRO CRÍTICO] Falha no treinamento: {ex.Message}\nStack Trace: {ex.StackTrace}");
                Console.ResetColor();
                initialModel?.Dispose();
                throw;
            }
        });
    }

    public async Task TrainWithTeacherAsync(Trainer trainerOptions)
    {
        if (!File.Exists(trainerOptions.datasetPath))
        {
            throw new FileNotFoundException($"Arquivo de dataset não encontrado em: {trainerOptions.datasetPath}");
        }

        await Task.Run(async () =>
        {
            const int VOCAB_SIZE = 20000;
            const int EMBEDDING_SIZE = 128;
            const int HIDDEN_SIZE = 196;

            GenerativeNeuralNetworkLSTM? initialModel = null;
            try
            {
                Console.WriteLine("[GenerativeService] Inicializando modelo LSTM para treinamento híbrido...");
                initialModel = new GenerativeNeuralNetworkLSTM(
                    VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE,
                    trainerOptions.datasetPath, _searchService, _mathEngine
                );
                initialModel.RunSanityCheck();

                var teacherService = new TeacherModelService();
                var hybridTrainer = new HybridTrainer(_mathEngine, teacherService);

                Console.WriteLine("\n[GenerativeService] Iniciando treinamento HÍBRIDO CONTÍNUO com professor de IA...");

                await hybridTrainer.TrainModelAsync(
                    initialModel,
                    _modelPath,
                    trainerOptions.learningRate,
                    totalEpochs: trainerOptions.epochs,
                    sftEpochs: 5, // As 5 primeiras épocas serão de destilação
                    trainerOptions.batchSize,
                    trainerOptions.validationSplit
                );

                Console.WriteLine($"\n[GenerativeService] Treinamento híbrido concluído! Carregando modelo final...");
                InitializeFromDisk();
                
                initialModel = null;
                ForceAggressiveGarbageCollection();
                Console.WriteLine("[GenerativeService] Serviço pronto para inferência com modelo treinado por professor!");
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"\n[ERRO CRÍTICO] Falha no treinamento híbrido: {ex.Message}\nStack Trace: {ex.StackTrace}");
                Console.ResetColor();
                initialModel?.Dispose();
                throw;
            }
        });
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
                if (File.Exists(Path.Combine(Environment.CurrentDirectory, "Dayson", "priming_prompt.txt")))
                {
                    _primingService.PrimeModel(_model);
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GenerativeService] Erro ao carregar modelo: {ex.Message}");
        }
    }

    private void ForceAggressiveGarbageCollection()
    {
        GC.Collect(2, GCCollectionMode.Forced, true, true);
        GC.WaitForPendingFinalizers();
        GC.Collect(2, GCCollectionMode.Forced, true, true);
    }
}