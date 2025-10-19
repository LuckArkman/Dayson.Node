using Microsoft.AspNetCore.Mvc;
using Services;

namespace Galileu.Node.Controllers;

[ApiController]
[Route("api/[controller]")]
public class CommandController : ControllerBase
{
    private readonly NodeRegistryService _nodeRegistryService;

    public CommandController(NodeRegistryService nodeRegistryService)
    {
        _nodeRegistryService = nodeRegistryService;
    }

    [HttpGet]
    public IActionResult Get()
    {
        return Ok("Controlador funcionando!");
    }
}