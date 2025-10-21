using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Galileu.Node.Data;
using MongoDB.Bson;

namespace Galileu.Node.Services;

/// <summary>
/// Serviço responsável por interagir com o modelo "Professor" (Modelo B, ex: Gemini),
/// gerar dados de treinamento supervisionado (SFT), e fornecer respostas de referência
/// com fallback para um cache no MongoDB para garantir resiliência.
/// </summary>
public class TeacherModelService
{
    private readonly HttpClient _httpClient;
    private readonly MongoDbService _mongoDbService;

    // ====================================================================
    // ⚠️ AÇÃO NECESSÁRIA ⚠️
    // SUBSTITUA O TEXTO ABAIXO PELA SUA CHAVE DE API VÁLIDA DO GOOGLE AI STUDIO.
    // Lembre-se de NUNCA compartilhar esta chave em repositórios públicos.
    // ====================================================================
    private const string ApiKey = "AIzaSyDEsWciYO_Zyi58pE9nXOH_C_Coe88FJ4Q";

    // Endpoint para um modelo rápido e estável do Gemini.
    private const string ApiEndpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent";

    private const int VOCAB_CONTEXT_SIZE = 4000;
    
    // ====================================================================
    // INÍCIO DA IMPLEMENTAÇÃO DO CONTROLE DE TAXA (RATE LIMITING)
    // ====================================================================
    
    // Define o número máximo de requisições permitidas por minuto.
    private const int MaxRequestsPerMinute = 15;
    
    // Define a janela de tempo para o controle de taxa (1 minuto).
    private static readonly TimeSpan RateLimitTimeWindow = TimeSpan.FromMinutes(1);
    
    // Fila thread-safe para armazenar os timestamps das requisições recentes.
    private readonly ConcurrentQueue<DateTime> _requestTimestamps = new();
    
    // Semáforo para garantir que a verificação do limite seja atômica (evita race conditions).
    private readonly SemaphoreSlim _rateLimitLock = new(1, 1);

    // ====================================================================
    // FIM DA IMPLEMENTAÇÃO DO CONTROLE DE TAXA
    // ====================================================================

    public TeacherModelService(MongoDbService mongoDbService)
    {
        _httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(220) };
        _httpClient.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
        _mongoDbService = mongoDbService;
    }

    /// <summary>
    /// Gera um dataset sintético em lotes para a fase de SFT.
    /// Salva os exemplos válidos no MongoDB para uso futuro como cache.
    /// </summary>
    public async Task<List<(string input, string output)>> GenerateSyntheticDataAsync(IEnumerable<string> vocabulary)
    {
        var syntheticDataset = new List<(string, string)>();
        var fullVocabularySet = new HashSet<string>(vocabulary);
        var vocabListForContext = vocabulary.Where(v => !v.StartsWith("<")).Take(VOCAB_CONTEXT_SIZE).ToList();
        string vocabContext = string.Join(", ", vocabListForContext);

        string systemPrompt =
            $"Você é um tutor de IA gerando um dataset. Sua tarefa é criar pares de 'pergunta' e 'resposta'. " +
            $"É CRÍTICO que suas respostas contenham APENAS palavras da seguinte lista de vocabulário: [{vocabContext}]. " +
            $"Não use nenhuma palavra que não esteja nesta lista. Se não puder formar uma resposta, omita o par. " +
            $"Responda em formato JSON com uma lista de objetos, cada um com as chaves 'pergunta' e 'resposta'.";

        var tokensToProcess = vocabulary.Where(v => !v.StartsWith("<") && v.Length > 2).ToList();
        int batchSize = 20;
        int discardedCount = 0;

        for (int i = 0; i < tokensToProcess.Count; i += batchSize)
        {
            var batchTokens = tokensToProcess.Skip(i).Take(batchSize).ToList();
            Console.Write($"\r[Teacher] Gerando dados: Processando lote {i / batchSize + 1}/{(tokensToProcess.Count / batchSize) + 1}. Descartados: {discardedCount}...");

            var instructions = new StringBuilder();
            instructions.AppendLine("Gere um exemplo de 'definição' e um de 'uso em frase' para cada uma das seguintes palavras:");
            foreach (var token in batchTokens)
            {
                instructions.AppendLine($"- {token}");
            }

            var userPrompt = instructions.ToString();
            
            var fullPrompt = $"{systemPrompt}\n\n{userPrompt}";
            //Console.WriteLine(fullPrompt);
            var jsonResponse = await CallApiAsync(fullPrompt);

            if (!string.IsNullOrEmpty(jsonResponse))
            {
                try
                {
                    var cleanedJson = jsonResponse.Trim().Replace("```json", "").Replace("```", "");
                    var generatedPairs = JsonSerializer.Deserialize<List<JsonElement>>(cleanedJson);

                    foreach (var pairElement in generatedPairs)
                    {
                        if (pairElement.TryGetProperty("pergunta", out var q) && pairElement.TryGetProperty("resposta", out var a))
                        {
                            string pergunta = q.GetString() ?? "";
                            string resposta = a.GetString() ?? "";

                            if (!string.IsNullOrEmpty(pergunta) && !string.IsNullOrEmpty(resposta) && IsResponseValid(resposta, fullVocabularySet))
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
                    Console.WriteLine($"\n[Teacher] AVISO: Falha ao parsear JSON. Erro: {ex.Message}");
                    discardedCount += batchTokens.Count * 2;
                }
            }
            else
            {
                discardedCount += batchTokens.Count * 2;
            }
        }

        if (syntheticDataset.Any())
        {
            var examplesToSave = syntheticDataset
                .Select(p => new GeneratedExample(ObjectId.GenerateNewId(), p.Item1, p.Item2, DateTime.UtcNow))
                .ToList();
            await _mongoDbService.InsertManyAsync(examplesToSave);
            Console.WriteLine($"\n[Teacher] {examplesToSave.Count} novos exemplos salvos no MongoDB.");
        }

        Console.WriteLine($"\n[Teacher] Geração de dataset sintético concluída. Exemplos válidos: {syntheticDataset.Count}. Exemplos descartados: {discardedCount}.");
        return syntheticDataset;
    }
    
    /// <summary>
    /// Método principal para a fase de RLAIF. Tenta obter uma resposta ideal da API.
    /// Se falhar, busca um exemplo aleatório no cache do MongoDB.
    /// </summary>
    public async Task<string> GetReferenceResponseAsync(string prompt, HashSet<string> vocabularySet)
    {
        try
        {
            string response = await CallApiAsync(prompt);
            
            if (!string.IsNullOrEmpty(response) && IsResponseValid(response, vocabularySet))
            {
                var example = new GeneratedExample(ObjectId.GenerateNewId(), prompt, response, DateTime.UtcNow);
                await _mongoDbService.InsertManyAsync(new List<GeneratedExample> { example });
                return response;
            }
            
            throw new Exception("Resposta da API inválida ou fora do vocabulário.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n[Teacher] Falha na API ({ex.Message}). Usando fallback do MongoDB...");
            var cachedExample = await _mongoDbService.GetRandomExampleAsync();
            if (cachedExample != null)
            {
                Console.WriteLine("[Teacher] Exemplo do cache do MongoDB recuperado com sucesso.");
                return cachedExample.Output;
            }
            
            Console.WriteLine("[Teacher] AVISO: Cache do MongoDB vazio ou indisponível. Retornando resposta padrão.");
            return "não sei"; 
        }
    }

    /// <summary>
    /// Valida se todos os tokens de uma resposta existem no vocabulário fornecido.
    /// </summary>
    private bool IsResponseValid(string response, HashSet<string> vocabularySet)
    {
        string pattern = @"(\p{L}+|\p{N}+|[.,!?;:'""/\-])";
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

    /// <summary>
    /// Aguarda, se necessário, para garantir que o limite de requisições por minuto não seja excedido.
    /// </summary>
    private async Task WaitForRateLimitAsync()
    {
        await _rateLimitLock.WaitAsync();
        try
        {
            // Loop para garantir que, após a espera, a condição seja reavaliada.
            while (true)
            {
                var now = DateTime.UtcNow;
                var boundary = now.Subtract(RateLimitTimeWindow);

                // Remove timestamps da fila que são mais antigos que a janela de tempo.
                while (_requestTimestamps.TryPeek(out var oldest) && oldest < boundary)
                {
                    _requestTimestamps.TryDequeue(out _);
                }

                // Se a contagem de requisições na janela de tempo for menor que o limite, permite prosseguir.
                if (_requestTimestamps.Count < MaxRequestsPerMinute)
                {
                    _requestTimestamps.Enqueue(now);
                    break; // Sai do loop e permite a execução da chamada API.
                }

                // Se o limite foi atingido, calcula o tempo de espera necessário.
                _requestTimestamps.TryPeek(out var nextAvailableSlot);
                var waitTime = (nextAvailableSlot + RateLimitTimeWindow) - now;

                if (waitTime > TimeSpan.Zero)
                {
                    await Task.Delay(waitTime);
                }
            }
        }
        finally
        {
            _rateLimitLock.Release();
        }
    }

    /// <summary>
    /// Lógica de comunicação com a API do Gemini, com tratamento de erros robusto.
    /// Esta função agora respeita o limite de 15 requisições por minuto.
    /// </summary>
    public async Task<string> CallApiAsync(string prompt)
    {
        // Garante que o limite de taxa seja respeitado antes de fazer a chamada.
        await WaitForRateLimitAsync();

        try
        {
            var requestBody = new { contents = new[] { new { parts = new[] { new { text = prompt } } } } };
            var content = new StringContent(JsonSerializer.Serialize(requestBody), Encoding.UTF8, "application/json");

            var urlWithKey = $"{ApiEndpoint}?key={ApiKey}";
            var response = await _httpClient.PostAsync(urlWithKey, content);
            var jsonResponse = await response.Content.ReadAsStringAsync();

            if (!response.IsSuccessStatusCode)
            {
                Console.WriteLine($"\n[Teacher] ERRO na API: Status {response.StatusCode}.");
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
            
            Console.WriteLine("\n[Teacher] AVISO: Resposta da API recebida, mas em formato inesperado (pode ter sido bloqueada por segurança).");
            Console.WriteLine($"--> Resposta: {jsonResponse}");
            return "";
        }
        catch (TaskCanceledException ex)
        {
            Console.WriteLine($"\n[Teacher] ERRO: Timeout na chamada da API. A requisição demorou mais de 220 segundos. Detalhes: {ex.Message}");
            return "";
        }
        catch (HttpRequestException ex)
        {
            Console.WriteLine($"\n[Teacher] ERRO: Falha na comunicação HTTP com a API. Verifique sua conexão com a internet ou se o endpoint está correto.");
            if (ex.InnerException != null) Console.WriteLine($"--> Erro Interno: {ex.InnerException.Message}");
            return "";
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n[Teacher] ERRO CRÍTICO na chamada da API: {ex.Message}");
            if (ex.InnerException != null) Console.WriteLine($"--> Erro Interno: {ex.InnerException.Message}");
            return "";
        }
    }
}