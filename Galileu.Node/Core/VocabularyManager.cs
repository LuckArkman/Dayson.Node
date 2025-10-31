using System.Text.RegularExpressions;

namespace Galileu.Node.Core;

public class VocabularyManager
{
    private readonly Dictionary<string, int> vocab;
    private readonly Dictionary<int, string> reverseVocab;
    public string VocabFilePath { get; } = Path.Combine(Environment.CurrentDirectory, "Dayson", "vocab.txt");

    public VocabularyManager()
    {
        vocab = new Dictionary<string, int>();
        reverseVocab = new Dictionary<int, string>();
    }

    public Dictionary<string, int> Vocab => vocab;
    public Dictionary<int, string> ReverseVocab => reverseVocab;
    public int VocabSize => vocab.Count;

    public int BuildVocabulary(string datasetPath, int maxVocabSize = 20000)
    {
        if (!File.Exists(datasetPath))
        {
            Console.WriteLine($"[VocabularyManager] ERRO: Arquivo de dataset não encontrado: {datasetPath}");
            return 0;
        }
        if (!File.Exists(VocabFilePath))
        {
            
            Console.WriteLine($"[VocabularyManager] Arquivo de vocabulario encontrado: {VocabFilePath}");
            return LoadVocabulary();
        }

        if (File.Exists(Path.Combine(Environment.CurrentDirectory, "Dayson", "vocab.txt")))
        {
            int vocabSize = LoadVocabulary();
            Console.WriteLine($"[VocabularyManager] Carregando vocabulario in Disco (máx: {VocabSize} tokens)...");
            return vocabSize;
        }

        Console.WriteLine($"[VocabularyManager] Construindo vocabulário (máx: {maxVocabSize} tokens)...");
        
        string text = File.ReadAllText(datasetPath);
        if (string.IsNullOrWhiteSpace(text))
        {
            Console.WriteLine("[VocabularyManager] ERRO: Dataset vazio.");
            return 0;
        }

        // FASE 1: Tokenização e contagem de frequências
        var tokenFrequency = new Dictionary<string, int>();
        string pattern = @"(\p{L}+|\p{N}+|[.,!?;:'""/\-])";
        var matches = Regex.Matches(text.ToLower(), pattern);

        Console.Write("[VocabularyManager] Analisando tokens...");
        int processedTokens = 0;
        
        foreach (Match match in matches)
        {
            string token = match.Value;
            
            if (tokenFrequency.ContainsKey(token))
                tokenFrequency[token]++;
            else
                tokenFrequency[token] = 1;

            processedTokens++;
            if (processedTokens % 100000 == 0)
            {
                Console.Write($"\r[VocabularyManager] Analisando tokens: {processedTokens:N0}");
            }
        }

        Console.WriteLine($"\r[VocabularyManager] Total de tokens processados: {processedTokens:N0}");
        Console.WriteLine($"[VocabularyManager] Tokens únicos encontrados: {tokenFrequency.Count:N0}");

        // FASE 2: Seleção dos tokens mais frequentes
        Vocab.Clear();
        ReverseVocab.Clear();

        // Tokens especiais sempre presentes
        Vocab["<PAD>"] = 0;
        Vocab["<UNK>"] = 1;
        ReverseVocab[0] = "<PAD>";
        ReverseVocab[1] = "<UNK>";

        // Ordena por frequência decrescente e seleciona os top N
        var topTokens = tokenFrequency
            .OrderByDescending(kvp => kvp.Value)
            .Take(maxVocabSize - 2) // -2 para reservar espaço de <PAD> e <UNK>
            .ToList();

        int index = 2;
        int minFrequency = topTokens.Last().Value; // Frequência do token menos comum selecionado

        foreach (var kvp in topTokens)
        {
            Vocab[kvp.Key] = index;
            ReverseVocab[index] = kvp.Key;
            index++;
        }

        Console.WriteLine($"[VocabularyManager] Vocabulário construído:");
        Console.WriteLine($"  - Tamanho final: {Vocab.Count:N0} tokens");
        Console.WriteLine($"  - Tokens descartados: {tokenFrequency.Count - topTokens.Count:N0}");
        Console.WriteLine($"  - Frequência mínima incluída: {minFrequency:N0}");
        
        if (topTokens.Count > 0)
        {
            var mostCommon = topTokens.First();
            Console.WriteLine($"  - Token mais frequente: '{mostCommon.Key}' ({mostCommon.Value:N0}x)");
        }

        // FASE 3: Salva vocabulário em disco
        SaveVocabulary();

        return Vocab.Count;
    }
    /// <summary>
    /// Salva o vocabulário atual em arquivo de texto.
    /// </summary>
    public void SaveVocabulary()
    {
        try
        {
            var directory = Path.GetDirectoryName(VocabFilePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            var sortedVocab = Vocab.OrderBy(kvp => kvp.Value);
            var lines = sortedVocab.Select(kvp => $"{kvp.Key}\t{kvp.Value}");
            File.WriteAllLines(VocabFilePath, lines);
            
            Console.WriteLine($"[VocabularyManager] Vocabulário salvo em: {VocabFilePath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[VocabularyManager] ERRO ao salvar vocabulário: {ex.Message}");
        }
    }

    public int LoadVocabulary()
    {
        if (!File.Exists(VocabFilePath))
        {
            Console.WriteLine($"[VocabularyManager] Arquivo de vocabulário não encontrado: {VocabFilePath}");
            return 0;
        }

        try
        {
            Vocab.Clear();
            ReverseVocab.Clear();

            var lines = File.ReadAllLines(VocabFilePath);
            foreach (var line in lines)
            {
                var parts = line.Split('\t');
                if (parts.Length == 2 && int.TryParse(parts[1], out int index))
                {
                    string token = parts[0];
                    Vocab[token] = index;
                    ReverseVocab[index] = token;
                }
            }

            Console.WriteLine($"[VocabularyManager] Vocabulário carregado: {Vocab.Count} tokens");
            return Vocab.Count;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[VocabularyManager] ERRO ao carregar vocabulário: {ex.Message}");
            return 0;
        }
    }
}