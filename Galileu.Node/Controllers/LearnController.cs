using Microsoft.AspNetCore.Mvc;

namespace Galileu.Node.Controllers;

[ApiController]
[Route("api/[controller]")]
public class LearnController : ControllerBase
{
    public LearnController()
    {
    }

    [HttpGet]
    public IActionResult Get()
    {
        return Ok("LearnController funcionando!");
    }
}