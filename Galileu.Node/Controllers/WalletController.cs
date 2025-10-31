using Galileu.Node.Models;
using Galileu.Node.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;

namespace Galileu.Node.Controllers;

[ApiController]
[Route("api/[controller]")]
public class WalletController : ControllerBase
{
    private readonly WalletService _walletService;
    private readonly ILogger<WalletController> _logger;

    public WalletController(WalletService walletService, ILogger<WalletController> logger)
    {
        _walletService = walletService;
        _logger = logger;
    }

    [HttpPost("balance")]
    public async Task<IActionResult> GetBalance([FromBody] WalletRequest? request)
    {
        try
        {
            if (request == null || string.IsNullOrWhiteSpace(request.WalletAddress))
                return BadRequest(new { Message = "Endereço da carteira é obrigatório." });

            var balance = await _walletService.GetBalanceAsync(request.WalletAddress);
            return Ok(new
            {
                address = request.WalletAddress,
                balance
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Erro ao processar saldo da carteira.");
            return StatusCode(500, new { Message = "Erro inesperado ao processar sua solicitação." });
        }
    }

    /// <summary>
    /// Consulta o histÃ³rico de transaÃ§Ãµes de uma carteira. O endereÃ§o deve ser codificado em Base64Url.
    /// </summary>
    [HttpPost("history")]
    public async Task<IActionResult> GetHistory([FromBody] WalletRequest request)
    {
        try
        {
            // Use diretamente, sem decodificação
            if (string.IsNullOrWhiteSpace(request.WalletAddress))
                return BadRequest(new { Message = "O endereço da carteira não pode ser vazio." });

            var history = await _walletService.GetHistoryAsync(request.WalletAddress);
            return Ok(history);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Erro ao processar histórico da carteira.");
            return StatusCode(500, new { Message = "Erro inesperado ao processar sua solicitação." });
        }
    }
}