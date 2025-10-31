using System;
using System.IO;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using Services;

namespace Galileu.Node.Controllers;

[ApiController]
[Route("api/[controller]")]
public class SettingsController : ControllerBase
{
    private readonly NodeRegistryService _nodeRegistryService; // Exemplo de injeção se for o caso

    public SettingsController(NodeRegistryService nodeRegistryService)
    {
        _nodeRegistryService = nodeRegistryService;
    }

    [HttpGet]
    public IActionResult Get()
    {
        return Ok("Controlador funcionando!");
    }
}