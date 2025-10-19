using System.Text.RegularExpressions;
using System.IO;
using System;
using System.Collections.Generic;
using System.Linq;

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

    // ====================================================================
    // MÉTODO CORRIGIDO
    // ====================================================================
    public int BuildVocabulary(string datasetPath, int maxVocabSize = 20000)
    {
        // --- LÓGICA CORRIGIDA ---
        // 1. Tenta carregar o vocabulário primeiro.
        // Se o arquivo existir E contiver tokens, o trabalho está feito.
        if (File.Exists(VocabFilePath))
        {
            int loadedSize = LoadVocabulary();
            if (loadedSize > 0)
            {
                Console.WriteLine($"[VocabularyManager] Vocabulário válido encontrado e carregado de '{Path.GetFileName(VocabFilePath)}'. Pulando construção.");
                return loadedSize;
            }
        }

        // 2. Se o vocabulário não pôde ser carregado (arquivo não existe ou está vazio),
        // então prossegue para a construção a partir do dataset.
        Console.WriteLine($"[VocabularyManager] Vocabulário não encontrado ou vazio. Construindo um novo a partir do dataset...");
        
        if (!File.Exists(datasetPath))
        {
            Console.WriteLine($"[VocabularyManager] ERRO: Arquivo de dataset não encontrado para construir vocabulário: {datasetPath}");
            return 0;
        }

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

        Console.Write("[VocabularyManager] Analisando tokens do dataset...");
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

        Vocab["<PAD>"] = 0;
        Vocab["<UNK>"] = 1;
        ReverseVocab[0] = "<PAD>";
        ReverseVocab[1] = "<UNK>";

        var topTokens = tokenFrequency
            .OrderByDescending(kvp => kvp.Value)
            .Take(maxVocabSize - 2)
            .ToList();

        if (topTokens.Count == 0)
        {
            Console.WriteLine("[VocabularyManager] ERRO: Nenhum token válido encontrado no dataset para construir o vocabulário.");
            return 0;
        }

        int index = 2;
        int minFrequency = topTokens.Last().Value;

        foreach (var kvp in topTokens)
        {
            Vocab[kvp.Key] = index;
            ReverseVocab[index] = kvp.Key;
            index++;
        }

        Console.WriteLine($"[VocabularyManager] Vocabulário construído:");
        Console.WriteLine($"  - Tamanho final: {Vocab.Count:N0} tokens");
        Console.WriteLine($"  - Frequência mínima incluída: {minFrequency:N0}");
        
        // FASE 3: Salva o novo vocabulário em disco
        SaveVocabulary();

        return Vocab.Count;
    }

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
            Console.WriteLine($"[VocabularyManager] Vocabulário salvo em: {Path.GetFileName(VocabFilePath)}");
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
            // Não é um erro, apenas informa que não há nada para carregar.
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
                    Vocab[parts[0]] = index;
                    ReverseVocab[index] = parts[0];
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