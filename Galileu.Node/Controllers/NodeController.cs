using Microsoft.AspNetCore.Mvc;
using Galileu.Node.Models; // Adicionado se NodeInfo ou outras classes do Models forem usadas.
using Services; // Adicionado se NodeRegistryService ou outros serviços forem injetados.

namespace Galileu.Node.Controllers;

[ApiController]
[Route("api/[controller]")]
public class NodeController : ControllerBase
{
    private readonly NodeRegistryService _nodeRegistryService; // Exemplo de injeção se for o caso

    // Exemplo de construtor se NodeRegistryService for usado
    public NodeController(NodeRegistryService nodeRegistryService)
    {
        _nodeRegistryService = nodeRegistryService;
    }

    [HttpGet]
    public IActionResult Get()
    {
        return Ok("Controlador funcionando!");
    }

    [HttpPost("register")]
    public IActionResult RegisterNode([FromBody] NodeRegistrationRequest request)
    {
        // Exemplo de uso do serviço
        _nodeRegistryService.RegisterNode(
            null,
            request.WalletAddress,
            request.NodeAddress,
            request.Specializations
        );
        Console.WriteLine(
            $"[NodeController] Requisição de registro de nó recebida: ID={request.NodeId}, Endereço={request.NodeAddress}");
        return Ok($"Nó {request.NodeId} registrado com sucesso.");
    }

    // Você precisaria de um DTO para a requisição de registro:
    public class NodeRegistrationRequest
    {
        public string NodeId { get; set; } = null!;
        public string NodeAddress { get; set; } = null!;
        public string WalletAddress { get; set; } = null!;
        public List<string> Specializations { get; set; } = new List<string>();
    }
}