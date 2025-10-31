using Akka.Actor;
using Akka.Configuration;
using Galileu.Node.Models;
using Galileu.Node.Services;
using Galileu.Services;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Services;

public class AkkaHostedService : IHostedService
{
    private readonly IServiceProvider _serviceProvider;
    private ActorSystem? _actorSystem;

    public AkkaHostedService(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        var nodeState = _serviceProvider.GetRequiredService<NodeState>();
        var akkaPort = new Uri(nodeState.Address).Port + 10000; // Garante uma porta única

        var config = ConfigurationFactory.ParseString(@$"
             akka {{
                 actor {{
                     provider = remote
                 }}
                 remote {{
                     dot-netty.tcp {{
                         port = {akkaPort}
                         hostname = localhost
                     }}
                 }}
             }}");

        _actorSystem = ActorSystem.Create("Dyson-System", config);
        var actorSystemSingleton = _serviceProvider.GetRequiredService<ActorSystemSingleton>();
        actorSystemSingleton.ActorSystem = _actorSystem;
        using (var scope = _serviceProvider.CreateScope())
        {
            var services = scope.ServiceProvider;
            var registry = services.GetRequiredService<NodeRegistryService>();
            var specializations = new[] { "Tradução", "Geração de Código" };

            _actorSystem.ActorOf(
                Props.Create(() => new SpecializedNodeActor(specializations, registry, nodeState)),
                $"specialized-node-{nodeState.Id.Substring(0, 8)}"
            );
        }

        Console.WriteLine($"[Akka] Sistema de Atores iniciado em akka.tcp://Dyson-System@localhost:{akkaPort}");
        return Task.CompletedTask;
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        if (_actorSystem != null)
        {
            await _actorSystem.Terminate();
        }
    }
}