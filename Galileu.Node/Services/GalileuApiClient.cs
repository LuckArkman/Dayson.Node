using System.Net.Http.Json;
using Galileu.Node.Models;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging; // Onde os DTOs estão (ou Galileu.API.Controllers)

namespace Services;

/// <summary>
/// Cliente HTTP para que os nós trabalhadores possam interagir
/// com os serviços centralizados da Galileu.API (ex: WalletService).
/// </summary>
public class GalileuApiClient
{
    private readonly HttpClient _httpClient;
    private readonly ILogger<GalileuApiClient> _logger;

    public GalileuApiClient(IConfiguration configuration, ILogger<GalileuApiClient> logger)
    {
        _logger = logger;
        _httpClient = new HttpClient
        {
            // O endereço da API central é lido da configuração.
            BaseAddress = new Uri(configuration["GalileuApi:BaseUrl"]
                                  ?? "http://localhost:5001")
        };
    }

    public async Task<BalanceResponseDto?> GetBalanceAsync(string walletAddress)
    {
        try
        {
            // A chave precisa ser codificada para URL para evitar problemas com caracteres especiais.
            var encodedAddress = System.Web.HttpUtility.UrlEncode(walletAddress);

            // A rota deve ser consistente com a definida no WalletController.
            // Assumindo a rota /api/wallet/balance?address=...
            var response = await _httpClient.GetAsync($"/api/wallet/balance?address={encodedAddress}");

            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<BalanceResponseDto>();
        }
        catch (HttpRequestException ex)
        {
            _logger.LogError(ex, "Falha ao consultar o saldo para o endereço {Address}", walletAddress);
            return null;
        }
    }

    public async Task<IEnumerable<TransactionDocument>?> GetHistoryAsync(string walletAddress)
    {
        try
        {
            var encodedAddress = System.Web.HttpUtility.UrlEncode(walletAddress);
            var response = await _httpClient.GetAsync($"/api/wallet/history?address={encodedAddress}");

            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<IEnumerable<TransactionDocument>>();
        }
        catch (HttpRequestException ex)
        {
            _logger.LogError(ex, "Falha ao consultar o histórico para o endereço {Address}", walletAddress);
            return null;
        }
    }
}

// DTOs necessários para o cliente. É uma boa prática tê-los em um projeto compartilhado.
public record BalanceResponseDto(string Address, decimal Balance);