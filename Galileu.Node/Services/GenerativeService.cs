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

/// <summary>
/// Ponto de entrada para as operações de treinamento e geração.
/// Delega a lógica de treinamento complexa para os trainers especializados (HybridTrainer).
/// </summary>
public class GenerativeService
{
    private readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Dayson", "Dayson.json");
    private readonly ISearchService _searchService = new MockSearchService();
    private readonly IMathEngine _mathEngine;
    private readonly PrimingService _primingService;
    private readonly MongoDbService _mongoDbService;
    private GenerativeNeuralNetworkLSTM? _model;
    public bool IsModelLoaded => _model != null;

    public GenerativeService(PrimingService primingService, MongoDbService mongoDbService)
    {
        _primingService = primingService;
        _mongoDbService = mongoDbService;
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

    /// <summary>
    /// Ponto de entrada para o treinamento padrão (agora obsoleto, redireciona para o híbrido).
    /// </summary>
    public async Task TrainModelAsync(Trainer trainerOptions)
    {
        // Redireciona para o fluxo de treinamento híbrido, que é o padrão agora.
        Console.WriteLine("[GenerativeService] O treinamento padrão foi invocado, redirecionando para o fluxo de Treinamento Híbrido.");
        await TrainWithTeacherAsync(trainerOptions);
    }

    /// <summary>
    /// Inicia o processo de treinamento híbrido de longo prazo.
    /// </summary>
    public async Task TrainWithTeacherAsync(Trainer trainerOptions)
    {
        if (!File.Exists(trainerOptions.datasetPath))
        {
            throw new FileNotFoundException($"Arquivo de dataset não encontrado em: {trainerOptions.datasetPath}");
        }

        // Executa o treinamento em uma thread de background para não bloquear a API.
        await Task.Run(async () =>
        {
            const int VOCAB_SIZE = 20000;
            const int EMBEDDING_SIZE = 128;
            const int HIDDEN_SIZE = 256;

            GenerativeNeuralNetworkLSTM? initialModel = null;
            try
            {
                Console.WriteLine("[GenerativeService] Inicializando modelo LSTM para treinamento híbrido...");
                initialModel = new GenerativeNeuralNetworkLSTM(
                    VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE,
                    trainerOptions.datasetPath, _searchService, _mathEngine
                );
                initialModel.RunSanityCheck();

                var teacherService = new TeacherModelService(_mongoDbService);
                var hybridTrainer = new HybridTrainer(_mathEngine, teacherService);

                Console.WriteLine("\n[GenerativeService] Iniciando treinamento HÍBRIDO CONTÍNUO com professor de IA...");

                // Delega toda a lógica de treinamento para o HybridTrainer.
                await hybridTrainer.TrainModelAsync(
                    initialModel,
                    _modelPath,
                    trainerOptions.learningRate,
                    totalEpochs: trainerOptions.epochs,
                    sftEpochs: 5, // As 5 primeiras épocas serão de destilação.
                    trainerOptions.batchSize,
                    trainerOptions.validationSplit
                );

                Console.WriteLine($"\n[GenerativeService] Treinamento híbrido concluído! Carregando o modelo mais recente para inferência...");
                
                // Carrega o modelo da última época salva pelo HybridTrainer.
                string lastEpochModelPath = Path.Combine(Path.GetDirectoryName(_modelPath)!, $"dayson_epoch_{trainerOptions.epochs}.json");
                if (File.Exists(lastEpochModelPath))
                {
                     // Copia para o caminho padrão "Dayson.json" para inferência.
                    File.Copy(lastEpochModelPath, _modelPath, overwrite: true);
                }

                InitializeFromDisk();
                
                initialModel = null; // A instância inicial já foi descartada pelo HybridTrainer.
                ForceAggressiveGarbageCollection();
                Console.WriteLine("[GenerativeService] Serviço pronto para inferência com o modelo final treinado.");
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"\n[ERRO CRÍTICO] Falha no processo de treinamento híbrido: {ex.Message}\nStack Trace: {ex.StackTrace}");
                Console.ResetColor();
                initialModel?.Dispose(); // Garante a limpeza em caso de falha na inicialização.
                throw; // Re-lança a exceção para que o chamador (API) saiba da falha.
            }
        });
    }

    /// <summary>
    /// Gera uma resposta de texto a partir de um prompt.
    /// </summary>
    public async Task<string?> GenerateAsync(GenerateResponse generateResponse)
    {
        if (_model == null) return "Erro: O modelo não está carregado. Treine ou carregue um modelo primeiro.";
        return await Task.Run(() => _model.GenerateResponse(generateResponse.input, maxLength: 50));
    }

    /// <summary>
    /// Carrega o modelo a partir do caminho padrão em disco.
    /// </summary>
    public void InitializeFromDisk()
    {
        if (!File.Exists(_modelPath))
        {
            Console.WriteLine($"[GenerativeService] Modelo não encontrado em {_modelPath}. Nenhuma ação realizada.");
            return;
        }
        try
        {
            Console.WriteLine($"[GenerativeService] Carregando modelo de {_modelPath} para inferência...");
            _model?.Dispose(); // Descarta qualquer modelo antigo em memória.
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
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"[GenerativeService] ERRO ao carregar modelo de disco: {ex.Message}");
            Console.ResetColor();
            _model = null;
        }
    }

    private void ForceAggressiveGarbageCollection()
    {
        GC.Collect(2, GCCollectionMode.Forced, true, true);
        GC.WaitForPendingFinalizers();
        GC.Collect(2, GCCollectionMode.Forced, true, true);
        GC.WaitForPendingFinalizers();
    }
}