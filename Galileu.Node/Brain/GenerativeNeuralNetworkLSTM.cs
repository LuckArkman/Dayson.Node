// --- START OF FILE GenerativeNeuralNetworkLSTM.cs ---

using System;
using System.Collections.Generic;
using System.Linq;
using Galileu.Node.Core;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Brain;

/// <summary>
/// Estende a rede LSTM base (disk-backed) com a capacidade de gerar texto.
/// </summary>
public class GenerativeNeuralNetworkLSTM : NeuralNetworkLSTM
{
    public readonly VocabularyManager vocabularyManager;
    private readonly ISearchService searchService;
    private readonly int _embeddingSize;

    /// <summary>
    /// Construtor para criar um novo modelo generativo para treinamento.
    /// </summary>
    public GenerativeNeuralNetworkLSTM(int vocabSize, int embeddingSize, int hiddenSize, string datasetPath,
        ISearchService? searchService, IMathEngine mathEngine)
        : base(vocabSize, embeddingSize, hiddenSize, vocabSize, mathEngine)
    {
        this.vocabularyManager = new VocabularyManager();
        this.searchService = searchService ?? new MockSearchService();
        this._embeddingSize = embeddingSize;
        
        int loadedVocabSize = vocabularyManager.BuildVocabulary(datasetPath, maxVocabSize: vocabSize);
        if (loadedVocabSize == 0)
        {
            throw new InvalidOperationException("Vocabulário vazio. Verifique o arquivo de dataset.");
        }
    }

    /// <summary>
    /// Construtor privado para "envolver" um modelo base já carregado.
    /// Usa o construtor de cópia protegido da classe base para evitar a reinicialização dos pesos.
    /// </summary>
    private GenerativeNeuralNetworkLSTM(NeuralNetworkLSTM baseModel, VocabularyManager vocabManager, ISearchService? searchService)
        : base(baseModel) // Chama o construtor protegido de cópia.
    {
        this.vocabularyManager = vocabManager ?? throw new ArgumentNullException(nameof(vocabManager));
        this.searchService = searchService ?? new MockSearchService();
        
        // ===================================================================
        // == CORREÇÃO APLICADA AQUI                                        ==
        // ===================================================================
        // Acessa diretamente o _tensorManager (que é 'protected') em vez de
        // usar o método wrapper 'this.GetShape()', que causa problemas de 
        // resolução no construtor.
        this._embeddingSize = _tensorManager.GetShape(_weightsEmbeddingId)[1];
        // ===================================================================
    }
    
    /// <summary>
    /// Método de fábrica estático para carregar um modelo e envolvê-lo.
    /// </summary>
    public static GenerativeNeuralNetworkLSTM? Load(string modelPath, IMathEngine mathEngine, VocabularyManager vocabManager, ISearchService? searchService)
    {
        var baseModel = NeuralNetworkLSTM.LoadModel(modelPath, mathEngine);
        if (baseModel == null)
        {
            return null;
        }
        return new GenerativeNeuralNetworkLSTM(baseModel, vocabManager, searchService);
    }

    /// <summary>
    /// Gera uma continuação de texto a partir de um prompt, em conformidade com a arquitetura disk-backed.
    /// </summary>
    public string GenerateResponse(string inputText, int maxLength = 50)
    {
        if (string.IsNullOrEmpty(inputText)) return "Erro: Entrada vazia ou nula.";

        ResetHiddenState();
        var tokens = Tokenize(inputText);

        using (var embeddingMatrix = _tensorManager.LoadTensor(_weightsEmbeddingId))
        {
            foreach (var token in tokens.Take(tokens.Length - 1))
            {
                using var embeddingVector = _mathEngine.CreateTensor(new[] { 1, _embeddingSize });
                int tokenIndex = GetTokenIndex(token);
                _mathEngine.Lookup(embeddingMatrix, tokenIndex, embeddingVector);
                base.Forward(embeddingVector.ToCpuTensor());
            }
        }

        var responseTokens = new List<string>();
        string lastToken = tokens.LastOrDefault() ?? "<UNK>";

        using (var embeddingMatrix = _tensorManager.LoadTensor(_weightsEmbeddingId))
        {
            for (int i = 0; i < maxLength; i++)
            {
                using var embeddingVector = _mathEngine.CreateTensor(new[] { 1, _embeddingSize });
                int lastTokenIndex = GetTokenIndex(lastToken);
                _mathEngine.Lookup(embeddingMatrix, lastTokenIndex, embeddingVector);
                var output = base.Forward(embeddingVector.ToCpuTensor());

                int predictedTokenIndex = SampleToken(output);
                string predictedToken = vocabularyManager.ReverseVocab.ContainsKey(predictedTokenIndex)
                    ? vocabularyManager.ReverseVocab[predictedTokenIndex]
                    : "<UNK>";

                responseTokens.Add(predictedToken);
                if (predictedToken == "." || predictedToken == "!" || predictedToken == "?" || predictedToken == "<EOS>")
                {
                    break;
                }
                lastToken = predictedToken;
            }
        }

        string response = string.Join(" ", responseTokens).Trim();
        return response.Length > 0 ? response.Capitalize() : "Não foi possível gerar uma resposta.";
    }
    
    public float CalculateSequenceLoss(int[] inputIndices, int[] targetIndices)
    {
        var (predictionsId, loss) = base.ForwardPassGpuOptimized(inputIndices, targetIndices, inputIndices.Length);
        
        // Estas chamadas funcionam porque estão em métodos normais, não em um construtor.
        // O objeto 'this' já está totalmente inicializado.
        this.DeleteTensor(predictionsId);
        this.ClearForwardCache();

        return loss;
    }

    public void Reset()
    {
        base.ResetHiddenState();
        this.ClearForwardCache();
    }
    
    private int GetTokenIndex(string token)
    {
        return vocabularyManager.Vocab.TryGetValue(token.ToLower(), out int tokenIndex)
            ? tokenIndex
            : vocabularyManager.Vocab["<UNK>"];
    }

    private string[] Tokenize(string text)
    {
        return text.ToLower().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
    }

    private int SampleToken(Tensor output)
    {
        float[] probs = output.GetData();
        float r = (float)new Random().NextDouble();
        float cumulative = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            cumulative += probs[i];
            if (r <= cumulative) return i;
        }
        return probs.Length - 1;
    }
}