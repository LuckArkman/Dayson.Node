using Galileu.Node.Core;
using Galileu.Node.Interfaces;
using System;
using System.IO;

namespace Galileu.Node.Brain;

public class ModelSerializerLSTM
{
    public static void SaveModel(GenerativeNeuralNetworkLSTM model, string filePath)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }
        // A lógica de salvar agora está totalmente encapsulada no próprio modelo.
        model.SaveModel(filePath);
    }

    // --- CORREÇÃO PRINCIPAL: O método agora aceita IMathEngine, não OpenCLService ---
    public static GenerativeNeuralNetworkLSTM? LoadModel(string filePath, IMathEngine mathEngine)
    {
        try
        {
            // 1. Carrega o modelo base (os pesos) usando a engine de matemática fornecida.
            // Esta chamada está correta agora.
            var baseModel = NeuralNetworkLSTM.LoadModel(filePath, mathEngine);
            if (baseModel == null)
            {
                // A mensagem de erro já é impressa dentro do NeuralNetworkLSTM.LoadModel.
                return null;
            }

            // 2. Carrega o vocabulário, que é salvo separadamente.
            var vocabManager = new VocabularyManager();
            int vocabSize = vocabManager.LoadVocabulary();
            if (vocabSize == 0)
            {
                Console.WriteLine("Erro: Vocabulário vazio ou não encontrado em 'vocab.txt'.");
                return null;
            }

            // 3. Valida a consistência entre o modelo carregado e o vocabulário.
            if (vocabSize != baseModel.OutputSize)
            {
                Console.WriteLine(
                    $"Erro de Inconsistência: Tamanho do vocabulário ({vocabSize}) não corresponde ao OutputSize do modelo ({baseModel.OutputSize}).");
                return null;
            }

            // --- CORREÇÃO PRINCIPAL: Usa o construtor correto e limpo ---
            // Em vez de passar 20 parâmetros, passamos o modelo base já construído.
            // O compilador encontrará este construtor:
            // GenerativeNeuralNetworkLSTM(NeuralNetworkLSTM, VocabularyManager, ISearchService?)
            return new GenerativeNeuralNetworkLSTM(baseModel, vocabManager, new MockSearchService());
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Erro crítico ao carregar o modelo LSTM generativo: {ex.Message}");
            return null;
        }
    }
}