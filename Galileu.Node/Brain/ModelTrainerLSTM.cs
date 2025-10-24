using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Collections.Generic; // Necessário para List<>
using Galileu.Node.Core;
using Galileu.Node.Interfaces;
using Galileu.Node.Services;

namespace Galileu.Node.Brain;

/// <summary>
/// Orquestra o treinamento de uma única época. VERSÃO FINAL: Opera diretamente
/// sobre listas de amostras em memória, eliminando a dependência do DatasetService.
/// </summary>
public class ModelTrainerLSTM
{
    private readonly IMathEngine _mathEngine;
    private readonly Stopwatch _stopwatch = new Stopwatch();
    private readonly Process _currentProcess;
    private readonly string logPath = Path.Combine(Environment.CurrentDirectory, "Dayson", "training_log.txt");

    public ModelTrainerLSTM(IMathEngine mathEngine)
    {
        _mathEngine = mathEngine ?? throw new ArgumentNullException(nameof(mathEngine));
        _currentProcess = Process.GetCurrentProcess();
    }

    /// <summary>
    /// 🔥 CORREÇÃO DEFINITIVA: Executa o treinamento e validação para uma única época
    /// usando listas de amostras pré-processadas em memória.
    /// </summary>
    public void TrainSingleEpoch(
        GenerativeNeuralNetworkLSTM model,
        List<(int InputIndex, int TargetIndex)> trainingSamples,
        List<(int InputIndex, int TargetIndex)> validationSamples,
        double learningRate,
        int batchSize)
    {
        _stopwatch.Restart();
        Console.WriteLine($"\n{'═',60}");
        Console.WriteLine($"INICIANDO ÉPOCA DE TREINAMENTO (SFT) >> LR: {learningRate} >> {DateTime.UtcNow}");
        Console.WriteLine($"{'═',60}");

        double totalEpochLoss = 0;
        int batchCount = 0;
        
        if (trainingSamples.Count == 0)
        {
            Console.WriteLine("[AVISO] Nenhuma amostra de treinamento fornecida para esta época.");
        }

        // Loop de treinamento sobre a lista em memória
        for (int i = 0; i < trainingSamples.Count; i += batchSize)
        {
            var batch = trainingSamples.Skip(i).Take(batchSize).ToList();
            if (batch.Count == 0) break;

            var sequenceInputIndices = batch.Select(p => p.InputIndex).ToArray();
            var sequenceTargetIndices = batch.Select(p => p.TargetIndex).ToArray();

            model.ResetHiddenState();
            totalEpochLoss += model.TrainSequence(sequenceInputIndices, sequenceTargetIndices, learningRate);
            batchCount++;
            Console.Write($"\r  Lotes de treinamento processados: {batchCount} de {Math.Ceiling((double)trainingSamples.Count / batchSize)}...");

            model._cacheManager?.Reset();
        }

        _stopwatch.Stop();
        double avgLoss = batchCount > 0 ? totalEpochLoss / batchCount : double.PositiveInfinity;
        string elapsedFormatted = string.Format("{0:D2}:{1:D2}:{2:D2}", _stopwatch.Elapsed.Hours, _stopwatch.Elapsed.Minutes, _stopwatch.Elapsed.Seconds);
        
        Console.WriteLine($"\nÉpoca SFT concluída. Perda média: {avgLoss:F4} | Duração: {elapsedFormatted}");
        File.AppendAllText(logPath, $"Época SFT concluída. Perda: {avgLoss}. Duração: {elapsedFormatted}{Environment.NewLine}");

        // Validação
        double validationLoss = ValidateModel(model, validationSamples, batchSize);
        Console.WriteLine($"Perda Média de Validação: {validationLoss:F4}");
        File.AppendAllText(logPath, $"Perda Média de Validação: {validationLoss:F4}{Environment.NewLine}");
    }

    private double ValidateModel(GenerativeNeuralNetworkLSTM modelToValidate, List<(int InputIndex, int TargetIndex)> validationSamples, int batchSize)
    {
        Console.WriteLine("\n[Validação] Iniciando...");
        double totalLoss = 0;
        int batchCount = 0;
        var validationStopwatch = Stopwatch.StartNew();
        
        if (validationSamples.Count == 0)
        {
            Console.WriteLine("[AVISO] Nenhuma amostra de validação fornecida.");
        }

        for (int i = 0; i < validationSamples.Count; i += batchSize)
        {
            var batch = validationSamples.Skip(i).Take(batchSize).ToList();
            if (batch.Count == 0) break;

            var sequenceInputIndices = batch.Select(p => p.InputIndex).ToArray();
            var sequenceTargetIndices = batch.Select(p => p.TargetIndex).ToArray();

            totalLoss += modelToValidate.CalculateSequenceLoss(sequenceInputIndices, sequenceTargetIndices);
            batchCount++;
            Console.Write($"\r[Validação] Processando lote de validação {batchCount}...");
            
            modelToValidate._cacheManager?.Reset();
        }

        validationStopwatch.Stop();
        _currentProcess.Refresh();
        long currentMemory = _currentProcess.WorkingSet64 / (1024 * 1024);
        Console.WriteLine($"\r[Validação] Concluída em {validationStopwatch.Elapsed:mm\\:ss}. | RAM: {currentMemory}MB");
        
        return batchCount > 0 ? totalLoss / batchCount : double.PositiveInfinity;
    }
}