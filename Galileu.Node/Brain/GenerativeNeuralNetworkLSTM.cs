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

        // Esta verificação pode ser flexibilizada dependendo da necessidade
        if (loadedVocabSize != vocabSize && vocabularyManager.Vocab.Count < vocabSize)
        {
             Console.WriteLine($"AVISO: O tamanho do vocabulário construído ({loadedVocabSize}) é menor que o solicitado ({vocabSize}). O modelo continuará, mas pode ser subótimo.");
        }
    }

    /// <summary>
    /// Novo construtor para "envolver" um modelo base (NeuralNetworkLSTM) que já foi carregado do disco.
    /// </summary>
    public GenerativeNeuralNetworkLSTM(NeuralNetworkLSTM baseModel, VocabularyManager vocabManager,
            ISearchService? searchService)
        // Chama o construtor protegido da classe base para transferir eficientemente
        // todos os pesos e a configuração que já foram carregados.
        : base(
            baseModel.InputSize,
            baseModel.weightsEmbedding!.Shape[1],
            baseModel.HiddenSize,
            baseModel.OutputSize,
            baseModel.GetMathEngine(),
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
        if (_tensorPool == null)
        {
            throw new InvalidOperationException("TensorPool não está inicializado. O modelo foi criado sem uma IMathEngine de GPU?");
        }
        
        using var embeddingVector = _tensorPool.Rent(new[] { 1, _embeddingSize });

        // Aquece o estado da rede com o prompt, exceto o último token
        foreach (var token in tokens.Take(tokens.Length - 1))
        {
            int tokenIndex = GetTokenIndex(token);
            GetMathEngine().Lookup(weightsEmbedding!, tokenIndex, embeddingVector);
            Forward(new Tensor(embeddingVector.ToCpuTensor().GetData(), new[] { 1, _embeddingSize }));
        }

        var responseTokens = new List<string>();
        string lastToken = tokens.LastOrDefault() ?? "<UNK>";

        // Loop de geração de novos tokens
        for (int i = 0; i < maxLength; i++)
        {
            int lastTokenIndex = GetTokenIndex(lastToken);
            GetMathEngine().Lookup(weightsEmbedding!, lastTokenIndex, embeddingVector);
            var output = Forward(new Tensor(embeddingVector.ToCpuTensor().GetData(), new[] { 1, _embeddingSize }));

            int predictedTokenIndex = SampleToken(output);
            string predictedToken = vocabularyManager.ReverseVocab.ContainsKey(predictedTokenIndex)
                ? vocabularyManager.ReverseVocab[predictedTokenIndex]
                : "<UNK>";

            if (predictedToken == "." || predictedToken == "!" || predictedToken == "?" || predictedToken == "<EOS>")
            {
                responseTokens.Add(predictedToken);
                break;
            }

            responseTokens.Add(predictedToken);
            lastToken = predictedToken;
        }
        
        // A devolução do tensor agora está dentro do using, o que é redundante mas seguro.
        // _tensorPool.Return(embeddingVector); // O 'using' já cuida disso.

        string response = string.Join(" ", responseTokens).Trim();
        // Capitalize extension method não é padrão, substituído por lógica equivalente.
        return response.Length > 0 ? char.ToUpper(response[0]) + response.Substring(1) : "Não foi possível gerar uma resposta.";
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
        double[] probs = output.GetData();
        double r = new Random().NextDouble();
        double cumulative = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            cumulative += probs[i];
            if (r <= cumulative) return i;
        }
        return probs.Length - 1;
    }

    internal VocabularyManager VocabularyManager => vocabularyManager;

    /// <summary>
    /// Executa um único passo de "fine-tuning corretivo".
    /// </summary>
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
        
        // 🔥 CORREÇÃO: Usa o operador de propagação nula (?.) para chamar Reset()
        // de forma segura, evitando NullReferenceException se o _cacheManager não estiver
        // injetado, o que previne o encerramento silencioso do processo.
        _cacheManager?.Reset();

        return loss;
    }
}