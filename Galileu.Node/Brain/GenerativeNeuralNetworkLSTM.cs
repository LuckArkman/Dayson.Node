using System;
using System.Collections.Generic;
using System.Linq;
using Galileu.Node.Core;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Brain;

/// <summary>
/// Estende a rede LSTM base com a capacidade de gerar texto.
/// Esta versão é otimizada para trabalhar com a arquitetura de Embedding,
/// gerenciando vocabulário, tokenização e a conversão de tokens em vetores de embedding.
/// </summary>
public class GenerativeNeuralNetworkLSTM : NeuralNetworkLSTM
{
    private readonly VocabularyManager vocabularyManager;
    private readonly ISearchService searchService;
    private readonly int _embeddingSize;

    /// <summary>
    /// Construtor para criar um novo modelo generativo para treinamento.
    /// </summary>
    public GenerativeNeuralNetworkLSTM(int vocabSize, int embeddingSize, int hiddenSize, string datasetPath,
            ISearchService? searchService, IMathEngine mathEngine)
        // Chama o construtor da classe base com os novos parâmetros arquiteturais
        : base(vocabSize, embeddingSize, hiddenSize, vocabSize, mathEngine)
    {
        this.vocabularyManager = new VocabularyManager();
        this.searchService = searchService ?? new MockSearchService();
        this._embeddingSize = embeddingSize;

        // Constrói o vocabulário a partir do dataset, garantindo que o tamanho corresponde ao esperado
        int loadedVocabSize = vocabularyManager.BuildVocabulary(datasetPath, maxVocabSize: vocabSize);
        if (loadedVocabSize == 0)
        {
            throw new InvalidOperationException("Vocabulário vazio. Verifique o arquivo de dataset.");
        }

        if (loadedVocabSize != vocabSize)
        {
            throw new ArgumentException(
                $"O tamanho do vocabulário construído ({loadedVocabSize}) não corresponde ao solicitado ({vocabSize}).");
        }
    }

    /// <summary>
    /// Novo construtor para "envolver" um modelo base (NeuralNetworkLSTM) que já foi carregado do disco.
    /// Este é o construtor que o ModelSerializer/GenerativeService usará após o carregamento.
    /// </summary>
    public GenerativeNeuralNetworkLSTM(NeuralNetworkLSTM baseModel, VocabularyManager vocabManager,
            ISearchService? searchService)
        // Chama o construtor protegido da classe base para transferir eficientemente
        // todos os pesos e a configuração que já foram carregados.
        : base(
            baseModel.InputSize, // vocabSize
            baseModel.weightsEmbedding!.Shape[1], // embeddingSize (derivado do tensor carregado)
            baseModel.HiddenSize, // hiddenSize
            baseModel.OutputSize, // outputSize
            baseModel.GetMathEngine(), // Passa a engine que ele já está usando
            // Passa todos os tensores de peso já carregados
            baseModel.weightsEmbedding!, baseModel.weightsInputForget!, baseModel.weightsHiddenForget!,
            baseModel.weightsInputInput!, baseModel.weightsHiddenInput!,
            baseModel.weightsInputCell!, baseModel.weightsHiddenCell!,
            baseModel.weightsInputOutput!, baseModel.weightsHiddenOutput!,
            baseModel.biasForget!, baseModel.biasInput!,
            baseModel.biasCell!, baseModel.biasOutput!,
            baseModel.weightsHiddenOutputFinal!, baseModel.biasOutputFinal!
        )
    {
        this.vocabularyManager = vocabManager ?? throw new ArgumentNullException(nameof(vocabManager));
        this.searchService = searchService ?? new MockSearchService();
        this._embeddingSize = baseModel.weightsEmbedding!.Shape[1];
    }

    /// <summary>
    /// Gera uma continuação de texto a partir de um prompt de entrada.
    /// </summary>
    public string GenerateResponse(string inputText, int maxLength = 50)
    {
        if (string.IsNullOrEmpty(inputText)) return "Erro: Entrada vazia ou nula.";

        ResetHiddenState();
        var tokens = Tokenize(inputText);
        using var embeddingVector = _tensorPool!.Rent(new[] { 1, _embeddingSize });

        // Aquece o estado da rede com o prompt, exceto o último token
        foreach (var token in tokens.Take(tokens.Length - 1))
        {
            int tokenIndex = GetTokenIndex(token);
            // Executa o lookup para obter o vetor de embedding
            GetMathEngine().Lookup(weightsEmbedding!, tokenIndex, embeddingVector);
            // Passa o vetor denso para o forward pass
            Forward(new Tensor(embeddingVector.ToCpuTensor().GetData(), new[] { _embeddingSize }));
        }

        var responseTokens = new List<string>();
        string lastToken = tokens.LastOrDefault() ?? "<UNK>";

        // Loop de geração de novos tokens
        for (int i = 0; i < maxLength; i++)
        {
            int lastTokenIndex = GetTokenIndex(lastToken);
            // Obtém o embedding para o último token
            GetMathEngine().Lookup(weightsEmbedding!, lastTokenIndex, embeddingVector);

            // Executa o forward pass com o vetor de embedding
            var output = Forward(new Tensor(embeddingVector.ToCpuTensor().GetData(), new[] { _embeddingSize }));

            // Amostra o próximo token a partir da distribuição de probabilidade de saída
            int predictedTokenIndex = SampleToken(output);
            string predictedToken = vocabularyManager.ReverseVocab.ContainsKey(predictedTokenIndex)
                ? vocabularyManager.ReverseVocab[predictedTokenIndex]
                : "<UNK>";

            // Condição de parada
            if (predictedToken == "." || predictedToken == "!" || predictedToken == "?")
            {
                responseTokens.Add(predictedToken);
                break;
            }

            responseTokens.Add(predictedToken);
            lastToken = predictedToken;
        }

        // Limpa o tensor reutilizado
        _tensorPool.Return(embeddingVector);

        string response = string.Join(" ", responseTokens).Trim();
        return response.Length > 0 ? response.Capitalize() : "Não foi possível gerar uma resposta.";
    }

    // --- Métodos Utilitários ---

    /// <summary>
    /// Obtém o índice de um token do vocabulário, com fallback para "<UNK>".
    /// </summary>
    private int GetTokenIndex(string token)
    {
        return vocabularyManager.Vocab.TryGetValue(token, out int tokenIndex)
            ? tokenIndex
            : vocabularyManager.Vocab["<UNK>"];
    }

    /// <summary>
    /// Tokenização simples baseada em espaços.
    /// </summary>
    private string[] Tokenize(string text)
    {
        // Nota: A tokenização com Regex do VocabularyManager é mais robusta para a construção do vocabulário.
        // Esta é suficiente para a inferência em tempo real.
        return text.ToLower().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
    }

    /// <summary>
    /// Amostra um índice de token da distribuição de probabilidade de saída do modelo.
    /// </summary>
    private int SampleToken(Tensor output)
    {
        double[] probs = output.GetData();
        double r = new Random().NextDouble();
        double cumulative = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            cumulative += probs[i];
            if (r <= cumulative) return i;
        }

        return probs.Length - 1; // Fallback para o último token
    }

    internal VocabularyManager VocabularyManager => vocabularyManager;

    /// <summary>
    /// Executa um único passo de "fine-tuning corretivo".
    /// Este método realiza um microajuste nos pesos do modelo para aproximá-lo de uma
    /// resposta de referência, usando uma taxa de aprendizado muito baixa para
    /// evitar a perda da capacidade generativa geral.
    /// </summary>
    /// <param name="inputText">O prompt de entrada original.</param>
    /// <param name="correctResponseText">A resposta de referência (considerada "correta").</param>
    /// <param name="microLearningRate">Uma taxa de aprendizado muito pequena para o ajuste fino.</param>
    /// <returns>A perda (loss) calculada durante este passo de correção.</returns>
    public double CorrectiveFineTuningStep(string inputText, string correctResponseText, double microLearningRate)
    {
        var inputTokens = Tokenize(inputText);
        var targetTokens = Tokenize(correctResponseText);

        if (inputTokens.Length == 0 || targetTokens.Length == 0) return 0.0;
    
        var combinedSequence = inputTokens.Concat(targetTokens).ToArray();
        var inputIndices = new List<int>();
        var targetIndices = new List<int>();

        for(int i = 0; i < combinedSequence.Length - 1; i++)
        {
            inputIndices.Add(GetTokenIndex(combinedSequence[i]));
            targetIndices.Add(GetTokenIndex(combinedSequence[i+1]));
        }
    
        if (inputIndices.Count == 0) return 0.0;

        ResetHiddenState();
        double loss = TrainSequence(inputIndices.ToArray(), targetIndices.ToArray(), microLearningRate);
        _cacheManager.Reset();

        return loss;
    }
}