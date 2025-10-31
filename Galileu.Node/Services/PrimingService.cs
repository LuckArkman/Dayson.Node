using Galileu.Node.Brain;
using Galileu.Node.Core;

namespace Galileu.Node.Services;

public class PrimingService
{
    private readonly string _promptFilePath;

    public PrimingService()
    {
        _promptFilePath = Path.Combine(Environment.CurrentDirectory, "Dayson", "priming_prompt.txt");
    }

    /// <summary>
    /// Processa o prompt de diretiva para "aquecer" a memória do modelo.
    /// </summary>
    /// <param name="model">A instância do modelo a ser preparada.</param>
    public void PrimeModel(GenerativeNeuralNetworkLSTM model)
    {
        if (!File.Exists(_promptFilePath))
        {
            Console.WriteLine($"[PrimingService] Aviso: Arquivo de prompt não encontrado em '{_promptFilePath}'. O modelo não será inicializado com uma diretiva.");
            return;
        }

        Console.WriteLine("[PrimingService] Inicializando o modelo com a diretiva de comportamento...");

        var promptText = File.ReadAllText(_promptFilePath);
        var vocabManager = model.vocabularyManager;
        var tokens = promptText.ToLower().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
        foreach (var token in tokens)
        {
            var inputTensor = CreateOneHotTensorForToken(token, vocabManager);
            model.Forward(inputTensor);
        }

        Console.WriteLine("[PrimingService] Modelo inicializado com sucesso.");
    }
    
    
    private Tensor CreateOneHotTensorForToken(string token, VocabularyManager vocabManager)
    {
        int inputSize = vocabManager.VocabSize;
        float[] inputData = new float[inputSize];
        
        if (vocabManager.Vocab.TryGetValue(token, out int tokenIndex))
        {
            if (tokenIndex < inputData.Length)
                inputData[tokenIndex] = 1.0f;
        }
        else
        {
            if (vocabManager.Vocab.TryGetValue("<UNK>", out int unkIndex) && unkIndex < inputData.Length)
                inputData[unkIndex] = 1.0f;
        }

        return new Tensor(inputData, new int[] { inputSize });
    }
}