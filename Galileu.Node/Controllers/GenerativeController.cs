using Galileu.Node.Models;
using Galileu.Node.Services;
using Microsoft.AspNetCore.Mvc;

namespace Galileu.Node.Controllers;

[ApiController]
[Route("api/[controller]")]
public class GenerativeController : ControllerBase
{
    private readonly GenerativeService _generativeService;

    // O construtor agora apenas recebe a injeção de dependência.
    public GenerativeController(GenerativeService generativeService)
    {
        _generativeService = generativeService;
    }

    [HttpGet("status")]
    public IActionResult GetStatus()
    {
        if (_generativeService.IsModelLoaded)
        {
            return Ok("Serviço generativo pronto e modelo carregado.");
        }
        return Ok("Serviço generativo em execução, mas nenhum modelo foi carregado. Use o endpoint /trainer.");
    }

    [HttpPost("trainer")]
    // É mais comum usar [FromBody] para objetos complexos, a menos que esteja fazendo upload de arquivo.
    public async Task<IActionResult> Trainer([FromBody] Trainer trainerOptions)
    {
        try
        {
            // A validação e toda a lógica agora estão encapsuladas no serviço.
            //await _generativeService.TrainModelAsync(trainerOptions.);
            return Ok("O treinamento do modelo foi iniciado. Acompanhe o status no console.");
        }
        catch (FileNotFoundException ex)
        {
            return BadRequest(new { message = ex.Message });
        }
        catch (Exception ex)
        {
            // Captura outras exceções que podem ocorrer durante a configuração do treinamento.
            return StatusCode(500, new { message = $"Um erro ocorreu: {ex.Message}" });
        }
    }

    [HttpPost("generate")]
    public async Task<IActionResult> Generate([FromBody] GenerateResponse generateResponse)
    {
        if (!_generativeService.IsModelLoaded)
        {
            return BadRequest("O modelo não está carregado. Treine um modelo primeiro usando o endpoint /trainer.");
        }

        var response = await _generativeService.GenerateAsync(generateResponse);
        return Ok(new { response });
    }
}