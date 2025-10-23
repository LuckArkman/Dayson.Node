using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Galileu.Node.Core;
using Galileu.Node.Interfaces;
using Galileu.Node.Services;

namespace Galileu.Node.Brain;

/// <summary>
/// Orquestra o processo de treinamento de um modelo GenerativeNeuralNetworkLSTM para uma única época.
/// A lógica de gerenciamento de memória entre épocas foi centralizada no HybridTrainer.
/// </summary>
public class ModelTrainerLSTM
{
    private readonly IMathEngine _mathEngine;
    private readonly Stopwatch _stopwatch = new Stopwatch();
    private readonly Process _currentProcess;
    private long _peakMemoryUsageMB = 0;
    private readonly string logPath = Path.Combine(Environment.CurrentDirectory, "Dayson", "training_log.txt");

    public ModelTrainerLSTM(IMathEngine mathEngine)
    {
        _mathEngine = mathEngine ?? throw new ArgumentNullException(nameof(mathEngine));
        _currentProcess = Process.GetCurrentProcess();
    }

    /// <summary>
    /// Executa o treinamento e validação para uma única época.
    /// </summary>
    public void TrainSingleEpoch(
        GenerativeNeuralNetworkLSTM model,
        string datasetPath,
        double learningRate,
        int batchSize,
        int contextWindowSize,
        double validationSplit)
    {
        using (var datasetService = new DatasetService(Path.Combine(Environment.CurrentDirectory, "Dayson", "memory.bin")))
        {
            datasetService.InitializeAndSplit(File.ReadAllText(datasetPath), contextWindowSize, model.VocabularyManager.Vocab, "<PAD>", batchSize, validationSplit);

            _stopwatch.Restart();
            Console.WriteLine($"\n{'═',60}");
            Console.WriteLine($"INICIANDO ÉPOCA DE TREINAMENTO (SFT) >> LR: {learningRate} >> {DateTime.UtcNow}");
            Console.WriteLine($"{'═',60}");

            double totalEpochLoss = 0;
            int batchCount = 0;
            datasetService.ResetTrain();

            // Loop de treinamento
            while (true)
            {
                var batch = datasetService.GetNextTrainChunk();
                if (batch == null || batch.Count == 0) break;

                var sequenceInputIndices = batch.Select(p => p.InputIndex).ToArray();
                var sequenceTargetIndices = batch.Select(p => p.TargetIndex).ToArray();

                model.ResetHiddenState();
                totalEpochLoss += model.TrainSequence(sequenceInputIndices, sequenceTargetIndices, learningRate);
                batchCount++;
                Console.Write($"\r  Lotes processados: {batchCount} ...");

                // Limpa o cache DEPOIS de processar o lote
                model._cacheManager.Reset();
            }

            _stopwatch.Stop();
            double avgLoss = batchCount > 0 ? totalEpochLoss / batchCount : double.PositiveInfinity;
            string elapsedFormatted = string.Format("{0:D2}:{1:D2}:{2:D2}", _stopwatch.Elapsed.Hours, _stopwatch.Elapsed.Minutes, _stopwatch.Elapsed.Seconds);
            
            Console.WriteLine($"\nÉpoca SFT concluída. Perda média: {avgLoss:F4} | Duração: {elapsedFormatted}");
            File.AppendAllText(logPath, $"Época SFT concluída. Perda: {avgLoss}. Duração: {elapsedFormatted}{Environment.NewLine}");

            // Validação
            double validationLoss = ValidateModel(model, datasetService);
            Console.WriteLine($"Perda Média de Validação: {validationLoss:F4}");
            File.AppendAllText(logPath, $"Perda Média de Validação: {validationLoss:F4}{Environment.NewLine}");
        }
    }

    private double ValidateModel(GenerativeNeuralNetworkLSTM modelToValidate, DatasetService datasetService)
    {
        Console.WriteLine("\n[Validação] Iniciando...");
        double totalLoss = 0;
        int batchCount = 0;
        datasetService.ResetValidation();
        var validationStopwatch = Stopwatch.StartNew();

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
        Console.WriteLine($"\r[Validação] Concluída em {validationStopwatch.Elapsed:mm\\:ss}.");
        return batchCount > 0 ? totalLoss / batchCount : double.PositiveInfinity;
    }
    
    // Métodos de monitoramento de memória removidos, pois o ciclo de vida agora é externo.
}