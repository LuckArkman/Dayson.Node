using Galileu.Node.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Galileu.Node.Brain;

public static class TextMetricsCalculator
{
    #region BLEU Score
    public static double CalculateBleuScore(string candidate, string reference)
    {
        var candTokens = Tokenize(candidate);
        var refTokens = Tokenize(reference);
        if (candTokens.Length == 0 || refTokens.Length == 0) return 0.0;

        double score = 0.0;
        double[] weights = { 0.25, 0.25, 0.25, 0.25 }; 

        for (int n = 1; n <= 4; n++)
        {
            if (candTokens.Length < n) break;
            var candNgrams = GetNgrams(candTokens, n);
            var refNgrams = GetNgrams(refTokens, n);
            if (candNgrams.Count == 0) continue;

            int clippedCount = 0;
            var refCounts = refNgrams.GroupBy(g => g).ToDictionary(g => g.Key, g => g.Count());
            foreach (var group in candNgrams.GroupBy(g => g))
            {
                refCounts.TryGetValue(group.Key, out int refCount);
                clippedCount += Math.Min(group.Count(), refCount);
            }

            double precision = (double)clippedCount / candNgrams.Count;
            if (precision > 0) score += weights[n - 1] * Math.Log(precision);
        }

        double brevityPenalty = candTokens.Length < refTokens.Length ? Math.Exp(1.0 - (double)refTokens.Length / candTokens.Length) : 1.0;
        return brevityPenalty * Math.Exp(score);
    }

    private static List<string> GetNgrams(string[] tokens, int n)
    {
        var ngrams = new List<string>();
        for (int i = 0; i <= tokens.Length - n; i++) ngrams.Add(string.Join(" ", tokens.Skip(i).Take(n)));
        return ngrams;
    }
    #endregion

    #region ROUGE-L Score
    public static double CalculateRougeLScore(string candidate, string reference)
    {
        var candTokens = Tokenize(candidate);
        var refTokens = Tokenize(reference);
        if (candTokens.Length == 0 || refTokens.Length == 0) return 0.0;

        int lcsLength = LongestCommonSubsequence(candTokens, refTokens);
        double recall = (double)lcsLength / refTokens.Length;
        double precision = (double)lcsLength / candTokens.Length;

        if (precision + recall == 0) return 0.0;
        return (2 * precision * recall) / (precision + recall);
    }

    private static int LongestCommonSubsequence(string[] a, string[] b)
    {
        var lengths = new int[a.Length + 1, b.Length + 1];
        for (int i = 0; i < a.Length; i++)
            for (int j = 0; j < b.Length; j++)
                if (a[i] == b[j]) lengths[i + 1, j + 1] = lengths[i, j] + 1;
                else lengths[i + 1, j + 1] = Math.Max(lengths[i + 1, j], lengths[i, j + 1]);
        return lengths[a.Length, b.Length];
    }
    #endregion

    #region Cosine Similarity
    public static double CalculateCosineSimilarity(string candidate, string reference, Dictionary<string, int> vocab, IMathTensor embeddingMatrix)
    {
        var embeddingData = embeddingMatrix.ToCpuTensor().GetData();
        int embeddingSize = embeddingMatrix.Shape[1];

        var vecA = GetSentenceVector(candidate, vocab, embeddingData, embeddingSize);
        var vecB = GetSentenceVector(reference, vocab, embeddingData, embeddingSize);

        if (vecA == null || vecB == null) return 0.0;
        double dotProduct = 0.0, magnitudeA = 0.0, magnitudeB = 0.0;

        for (int i = 0; i < vecA.Length; i++)
        {
            dotProduct += vecA[i] * vecB[i];
            magnitudeA += vecA[i] * vecA[i];
            magnitudeB += vecB[i] * vecB[i];
        }

        magnitudeA = Math.Sqrt(magnitudeA);
        magnitudeB = Math.Sqrt(magnitudeB);

        if (magnitudeA == 0 || magnitudeB == 0) return 0.0;
        return dotProduct / (magnitudeA * magnitudeB);
    }

    private static double[] GetSentenceVector(string sentence, Dictionary<string, int> vocab, double[] embeddingData, int embeddingSize)
    {
        var tokens = Tokenize(sentence);
        var vector = new double[embeddingSize];
        int tokenCount = 0;
        foreach (var token in tokens)
        {
            if (vocab.TryGetValue(token, out int index))
            {
                int offset = index * embeddingSize;
                for (int i = 0; i < embeddingSize; i++) vector[i] += embeddingData[offset + i];
                tokenCount++;
            }
        }
        if (tokenCount == 0) return null;
        for (int i = 0; i < embeddingSize; i++) vector[i] /= tokenCount;
        return vector;
    }
    #endregion

    private static string[] Tokenize(string text) => text.ToLower().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
}