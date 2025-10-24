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

    /// <summary>
    /// 🔥 CORREÇÃO DEFINITIVA: Método público que expõe a lógica de tokenização
    /// consistente com a usada para construir o vocabulário.
    /// </summary>
    /// <param name="text">O texto a ser tokenizado.</param>
    /// <returns>Um array de tokens em minúsculas.</returns>
    public string[] Tokenize(string text)
    {
        // Usa a mesma Regex que BuildVocabulary para garantir consistência.
        string pattern = @"(\p{L}+|\p{N}+|[.,!?;:'""/\-])";
        var matches = Regex.Matches(text.ToLower(), pattern);
        return matches.Cast<Match>().Select(m => m.Value).ToArray();
    }

    public int BuildVocabulary(string datasetPath, int maxVocabSize = 20000)
    {
        if (File.Exists(VocabFilePath))
        {
            int loadedSize = LoadVocabulary();
            if (loadedSize > 0)
            {
                Console.WriteLine($"[VocabularyManager] Vocabulário válido encontrado e carregado de '{Path.GetFileName(VocabFilePath)}'. Pulando construção.");
                return loadedSize;
            }
        }

        Console.WriteLine($"[VocabularyManager] Vocabulário não encontrado. Construindo um novo a partir do dataset...");
        if (!File.Exists(datasetPath))
        {
            Console.WriteLine($"[VocabularyManager] ERRO: Arquivo de dataset não encontrado: {datasetPath}");
            return 0;
        }

        string text = File.ReadAllText(datasetPath);
        if (string.IsNullOrWhiteSpace(text)) return 0;

        var tokenFrequency = new Dictionary<string, int>();
        var tokens = Tokenize(text); // Agora usa o método centralizado

        foreach (string token in tokens)
        {
            if (tokenFrequency.ContainsKey(token))
                tokenFrequency[token]++;
            else
                tokenFrequency[token] = 1;
        }

        Vocab.Clear();
        ReverseVocab.Clear();
        Vocab["<PAD>"] = 0; Vocab["<UNK>"] = 1;
        ReverseVocab[0] = "<PAD>"; ReverseVocab[1] = "<UNK>";

        var topTokens = tokenFrequency.OrderByDescending(kvp => kvp.Value).Take(maxVocabSize - 2);

        int index = 2;
        foreach (var kvp in topTokens)
        {
            Vocab[kvp.Key] = index;
            ReverseVocab[index] = kvp.Key;
            index++;
        }

        SaveVocabulary();
        return Vocab.Count;
    }

    public void SaveVocabulary()
    {
        try
        {
            var directory = Path.GetDirectoryName(VocabFilePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory)) Directory.CreateDirectory(directory);
            var sortedVocab = Vocab.OrderBy(kvp => kvp.Value);
            File.WriteAllLines(VocabFilePath, sortedVocab.Select(kvp => $"{kvp.Key}\t{kvp.Value}"));
            Console.WriteLine($"[VocabularyManager] Vocabulário salvo em: {Path.GetFileName(VocabFilePath)}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[VocabularyManager] ERRO ao salvar vocabulário: {ex.Message}");
        }
    }

    public int LoadVocabulary()
    {
        if (!File.Exists(VocabFilePath)) return 0;
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