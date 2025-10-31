// --- START OF FILE ModelSerializerLSTM.cs ---

using Galileu.Node.Core;
using Galileu.Node.Interfaces;
using System;
using System.IO;

namespace Galileu.Node.Brain;

/// <summary>
/// Serializador para o modelo LSTM.
/// NOTA: Esta classe foi simplificada para atuar como um wrapper, delegando a lógica
/// de carregamento para o método de fábrica estático na própria classe do modelo,
/// que é a prática recomendada para a arquitetura atual.
/// </summary>
public class ModelSerializerLSTM
{
    /// <summary>
    /// Salva o modelo no caminho especificado, delegando a lógica para o próprio modelo.
    /// </summary>
    public static void SaveModel(GenerativeNeuralNetworkLSTM model, string filePath)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }
        
        // A lógica de salvamento está corretamente encapsulada no modelo.
        model.SaveModel(filePath);
    }

    /// <summary>
    /// Carrega um modelo generativo do disco.
    /// CORRIGIDO: Delega a chamada para o método de fábrica estático 'GenerativeNeuralNetworkLSTM.Load',
    /// que encapsula toda a lógica de carregamento e validação.
    /// </summary>
    public static GenerativeNeuralNetworkLSTM? LoadModel(string filePath, IMathEngine mathEngine)
    {
        try
        {
            // 1. Carrega o vocabulário, que é um componente externo ao modelo.
            var vocabManager = new VocabularyManager();
            int vocabSize = vocabManager.LoadVocabulary();
            if (vocabSize == 0)
            {
                Console.WriteLine("Erro: Vocabulário vazio ou não encontrado em 'vocab.txt'. O carregamento do modelo não pode continuar.");
                return null;
            }

            // 2. Delega a lógica de carregamento do modelo para o método de fábrica do próprio modelo.
            // Este método lida com o carregamento do modelo base, a validação de consistência e a
            // instanciação correta usando o construtor privado.
            var searchService = new MockSearchService(); // Pode ser injetado se necessário.
            var model = GenerativeNeuralNetworkLSTM.Load(filePath, mathEngine, vocabManager, searchService);

            if (model == null)
            {
                // A mensagem de erro específica já foi impressa pelo método Load/LoadModel.
                return null;
            }

            // 3. Validação final de consistência entre vocabulário e modelo carregado.
            if (vocabManager.VocabSize != model.OutputSize)
            {
                Console.WriteLine(
                    $"Erro de Inconsistência: Tamanho do vocabulário ({vocabManager.VocabSize}) não corresponde ao OutputSize do modelo ({model.OutputSize}).");
                return null;
            }
            
            return model;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Erro crítico ao carregar o modelo LSTM generativo: {ex.Message}");
            return null;
        }
    }
}