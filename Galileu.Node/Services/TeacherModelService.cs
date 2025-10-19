using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Galileu.Node.Services;

public class TeacherModelService
{
    private readonly HttpClient _httpClient;
    
    // IMPORTANTE: Mova esta chave para variáveis de ambiente em produção!
    private const string ApiKey = "AIzaSyDEsWciYO_Zyi58pE9nXOH_C_Coe88FJ4Q";

    // Endpoint corrigido - usando gemini-1.5-flash (estável)
    private const string ApiEndpoint =
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent";

    private const int VOCAB_CONTEXT_SIZE = 4000;

    public TeacherModelService()
    {
        _httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(120) };
        _httpClient.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
    }

    // ====================================================================
    // MÉTODO OTIMIZADO COM PROCESSAMENTO EM LOTE
    // ====================================================================
    public async Task<List<(string input, string output)>> GenerateSyntheticDataAsync(IEnumerable<string> vocabulary)
    {
        var syntheticDataset = new List<(string, string)>();
        var fullVocabularySet = new HashSet<string>(vocabulary);
        var vocabListForContext = vocabulary.Where(v => !v.StartsWith("<")).Take(VOCAB_CONTEXT_SIZE).ToList();
        string vocabContext = string.Join(", ", vocabListForContext);

        string systemPrompt =
            $"Você é um tutor de IA gerando um dataset. Sua tarefa é criar pares de 'pergunta' e 'resposta'. " +
            $"É CRÍTICO que suas respostas contenham APENAS palavras da seguinte lista de vocabulário: [{vocabContext}]. " +
            $"Responda em formato JSON com uma lista de objetos, cada um com as chaves 'pergunta' e 'resposta'.";

        var tokensToProcess = vocabulary.Where(v => !v.StartsWith("<") && v.Length > 2).ToList();
        int batchSize = 20; // Processa 20 tokens por chamada de API
        int discardedCount = 0;

        for (int i = 0; i < tokensToProcess.Count; i += batchSize)
        {
            var batchTokens = tokensToProcess.Skip(i).Take(batchSize).ToList();
            Console.Write($"\r[Teacher] Gerando dados: Processando lote {i / batchSize + 1}/{(tokensToProcess.Count / batchSize) + 1}. Descartados: {discardedCount}...");

            var instructions = new StringBuilder();
            instructions.AppendLine(
                "Gere um exemplo de 'definição' e um de 'uso em frase' para cada uma das seguintes palavras:");
            foreach (var token in batchTokens)
            {
                instructions.AppendLine($"- {token}");
            }

            var userPrompt = instructions.ToString();
            var fullPrompt = $"{systemPrompt}\n\n{userPrompt}";

            var jsonResponse = await CallApiAsync(fullPrompt);

            if (!string.IsNullOrEmpty(jsonResponse))
            {
                try
                {
                    // Limpa a resposta para garantir que seja um JSON válido
                    var cleanedJson = jsonResponse.Trim().Replace("```json", "").Replace("```", "");
                    var generatedPairs = JsonSerializer.Deserialize<List<JsonElement>>(cleanedJson);

                    foreach (var pairElement in generatedPairs)
                    {
                        if (pairElement.TryGetProperty("pergunta", out var q) &&
                            pairElement.TryGetProperty("resposta", out var a))
                        {
                            string pergunta = q.GetString() ?? "";
                            string resposta = a.GetString() ?? "";

                            if (!string.IsNullOrEmpty(pergunta) && !string.IsNullOrEmpty(resposta) &&
                                IsResponseValid(resposta, fullVocabularySet))
                            {
                                syntheticDataset.Add((pergunta, resposta));
                            }
                            else
                            {
                                discardedCount++;
                            }
                        }
                    }
                }
                catch (JsonException ex)
                {
                    Console.WriteLine(
                        $"\n[Teacher] AVISO: Falha ao parsear JSON da API. Resposta: {jsonResponse.Substring(0, Math.Min(100, jsonResponse.Length))}... Erro: {ex.Message}");
                    discardedCount += batchTokens.Count * 2; // Assume que todo o lote falhou
                }
            }
            else
            {
                discardedCount += batchTokens.Count * 2; // Se a API não respondeu, descarta o lote
            }
        }

        Console.WriteLine(
            $"\n[Teacher] Geração de dataset sintético concluída. Exemplos válidos: {syntheticDataset.Count}. Exemplos descartados: {discardedCount}.");
        return syntheticDataset;
    }

    private bool IsResponseValid(string response, HashSet<string> vocabularySet)
    {
        // Pattern para extrair palavras e números (ignora pontuação)
        string pattern = @"(\p{L}+|\p{N}+)";
        var matches = Regex.Matches(response.ToLower(), pattern);
        
        foreach (Match match in matches)
        {
            if (!vocabularySet.Contains(match.Value))
            {
                return false;
            }
        }

        return true;
    }

    public async Task<string> CallApiAsync(string prompt)
    {
        try
        {
            var requestBody = new { contents = new[] { new { parts = new[] { new { text = prompt } } } } };
            var content = new StringContent(JsonSerializer.Serialize(requestBody), Encoding.UTF8, "application/json");

            // Adiciona a chave de API como parâmetro de query
            var urlWithKey = $"{ApiEndpoint}?key={ApiKey}";

            //Console.WriteLine("\n\r==================== INÍCIO DA CHAMADA DE API ====================");
            //Console.WriteLine("[PROMPT ENVIADO PARA O MODELO PROFESSOR]:");
            //Console.WriteLine(prompt); // Descomente para debug
            //Console.WriteLine("-----------------------------------------------------------------");

            var response = await _httpClient.PostAsync(urlWithKey, content);
            var jsonResponse = await response.Content.ReadAsStringAsync();

            //Console.WriteLine("[RESPOSTA BRUTA RECEBIDA DA API]:");
            //Console.WriteLine(jsonResponse); // Descomente para debug
            //Console.WriteLine("===================== FIM DA CHAMADA DE API =====================\n");

            if (!response.IsSuccessStatusCode)
            {
                Console.WriteLine($"\r[Teacher] ERRO na API: Status {response.StatusCode}. Verifique a resposta bruta impressa acima para detalhes.");
                Console.WriteLine($"[Teacher] Resposta de erro: {jsonResponse}");
                return "";
            }

            using var doc = JsonDocument.Parse(jsonResponse);

            if (doc.RootElement.TryGetProperty("candidates", out var candidates) && candidates.GetArrayLength() > 0)
            {
                var firstCandidate = candidates[0];
                if (firstCandidate.TryGetProperty("content", out var contentProp) &&
                    contentProp.TryGetProperty("parts", out var parts) && parts.GetArrayLength() > 0)
                {
                    return parts[0].GetProperty("text").GetString()?.Trim() ?? "";
                }
            }

            Console.WriteLine(
                "\n[Teacher] AVISO: Resposta da API recebida, mas em formato inesperado (pode ter sido bloqueada).");
            return "";
        }
        catch (TaskCanceledException ex)
        {
            Console.WriteLine($"\n[Teacher] ERRO: Timeout na chamada da API. A requisição demorou mais de 120 segundos.");
            Console.WriteLine($"--> Detalhes: {ex.Message}");
            return "";
        }
        catch (HttpRequestException ex)
        {
            Console.WriteLine($"\n[Teacher] ERRO: Falha na comunicação HTTP com a API.");
            Console.WriteLine($"--> Detalhes: {ex.Message}");
            if (ex.InnerException != null)
            {
                Console.WriteLine($"--> Erro Interno: {ex.InnerException.Message}");
            }
            return "";
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n[Teacher] ERRO CRÍTICO na chamada da API: {ex.Message}");
            if (ex.InnerException != null)
            {
                Console.WriteLine($"--> Erro Interno: {ex.InnerException.Message}");
            }
            return "";
        }
    }
}