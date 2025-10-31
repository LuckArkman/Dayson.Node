using Galileu.Node.Models;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using MongoDB.Driver;
using MongoDB.Driver.Linq;
using Services;

namespace Galileu.Node.Services;

public class WalletService
{
    private readonly IMongoCollection<TransactionDocument> _transactions;
    private readonly IMongoCollection<WalletDocument> _wallets;
    private readonly ILogger<WalletService> _logger;

    public WalletService(IOptions<object> mongoSettings, ILogger<WalletService> logger)
    {
    }

    // Método para criar e armazenar uma nova carteira
    public async Task CreateWalletAsync(string address)
    {
        var wallet = new WalletDocument
        {
            Address = address,
            CreatedAt = DateTime.UtcNow
        };
        await _wallets.InsertOneAsync(wallet);
        _logger.LogInformation("Nova carteira criada e armazenada no DB para o endereço: {Address}", address);
    }

    public async Task CreateTransactionAsync(string fromAddress, string toAddress, decimal amount, string? notes = null)
    {
        await Task.CompletedTask;
    }

    private string NormalizeWalletAddress(string walletAddress)
    {
        if (string.IsNullOrWhiteSpace(walletAddress))
            return string.Empty;

        // Corrige espaços que vieram de '+'
        string corrected = walletAddress.Replace(" ", "+");

        // Decodifica caracteres URL (%2F -> /, %2B -> +)
        string decoded = System.Web.HttpUtility.UrlDecode(corrected);

        return decoded;
    }

    public async Task<decimal> GetBalanceAsync(string walletAddress)
    {
        try
        {
            var receivedTask = _transactions.AsQueryable()
                .Where(t => t.toAddress == walletAddress)
                .SumAsync(t => t.amount);

            var sentTask = _transactions.AsQueryable()
                .Where(t => t.fromAddress == walletAddress)
                .SumAsync(t => t.amount);

            await Task.WhenAll(receivedTask, sentTask);

            var balance = receivedTask.Result - sentTask.Result;

            _logger.LogDebug(
                "Saldo calculado para carteira (hash: {Hash}): Recebido={Received}, Enviado={Sent}, Saldo={Balance}",
                walletAddress.GetHashCode(), receivedTask.Result, sentTask.Result, balance);

            return balance;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Erro ao calcular saldo da carteira (hash: {Hash})", walletAddress.GetHashCode());
            throw;
        }
    }

    public async Task<IEnumerable<TransactionDocument>> GetHistoryAsync(string walletAddress)
    {
        try
        {
            var filter = Builders<TransactionDocument>.Filter.Or(
                Builders<TransactionDocument>.Filter.Eq(t => t.fromAddress, walletAddress),
                Builders<TransactionDocument>.Filter.Eq(t => t.toAddress, walletAddress)
            );

            var transactions = await _transactions.Find(filter)
                .SortByDescending(t => t.timestamp)
                .ToListAsync();

            _logger.LogDebug("Histórico obtido para carteira (hash: {Hash}): {Count} transações",
                walletAddress.GetHashCode(), transactions.Count);

            return transactions;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Erro ao obter histórico da carteira (hash: {Hash})", walletAddress.GetHashCode());
            throw;
        }
    }

    /// <summary>
    /// Obtém o número total de transações de uma carteira
    /// </summary>
    public async Task<int> GetTransactionCountAsync(string walletAddress)
    {
        try
        {
            var filter = Builders<TransactionDocument>.Filter.Or(
                Builders<TransactionDocument>.Filter.Eq(t => t.fromAddress, walletAddress),
                Builders<TransactionDocument>.Filter.Eq(t => t.toAddress, walletAddress)
            );

            var count = (int)await _transactions.CountDocumentsAsync(filter);

            _logger.LogDebug("Contagem de transações para carteira (hash: {Hash}): {Count}",
                walletAddress.GetHashCode(), count);

            return count;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Erro ao contar transações da carteira (hash: {Hash})", walletAddress.GetHashCode());
            throw;
        }
    }

    /// <summary>
    /// Verifica se uma carteira existe
    /// </summary>
    public async Task<bool> WalletExistsAsync(string walletAddress)
    {
        try
        {
            var wallet = await _wallets.Find(w => w.Address == walletAddress).FirstOrDefaultAsync();
            var exists = wallet != null;

            _logger.LogDebug("Verificação de existência da carteira (hash: {Hash}): {Exists}",
                walletAddress.GetHashCode(), exists);

            return exists;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Erro ao verificar existência da carteira (hash: {Hash})",
                walletAddress.GetHashCode());
            throw;
        }
    }

    /// <summary>
    /// Obtém informações básicas da carteira
    /// </summary>
    public async Task<WalletDocument?> GetWalletInfoAsync(string walletAddress)
    {
        try
        {
            var wallet = await _wallets.Find(w => w.Address == walletAddress).FirstOrDefaultAsync();

            _logger.LogDebug("Informações da carteira obtidas (hash: {Hash}): {Found}",
                walletAddress.GetHashCode(), wallet != null);

            return wallet;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Erro ao obter informações da carteira (hash: {Hash})", walletAddress.GetHashCode());
            throw;
        }
    }

    public async Task<IEnumerable<TransactionDocument>> GetFullLedgerAsync()
    {
        try
        {
            // Retorna todas as transações, ordenadas pela mais recente primeiro
            var transactions = await _transactions.Find(_ => true)
                .SortByDescending(t => t.timestamp)
                .ToListAsync();

            _logger.LogDebug("Ledger completo obtido: {Count} transações", transactions.Count);

            return transactions;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Erro ao obter ledger completo");
            throw;
        }
    }
}