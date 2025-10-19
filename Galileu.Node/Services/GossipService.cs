using Galileu.Node.Models;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace Services;

public class GossipService : IHostedService, IDisposable
{
    private readonly NodeState _nodeState;
    private readonly NodeClient _nodeClient;
    private readonly ILogger<GossipService> _logger;
    private Timer? _timer;
    private readonly Random _random = new();

    public GossipService(NodeState nodeState, NodeClient nodeClient, ILogger<GossipService> logger)
    {
        _nodeState = nodeState;
        _nodeClient = nodeClient;
        _logger = logger;
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Gossip Service iniciado. Fofocando a cada 15 segundos.");
        _timer = new Timer(DoGossip, null, TimeSpan.FromSeconds(10), TimeSpan.FromSeconds(15));
        return Task.CompletedTask;
    }

    private async void DoGossip(object? state)
    {
        if (string.IsNullOrEmpty(_nodeState.NodeJwt))
        {
            _logger.LogTrace("Aguardando registro e JWT para iniciar o gossip.");
            return;
        }

        var otherPeers = _nodeState.GetKnownPeers().Where(p => p != _nodeState.Address).ToList();
        if (!otherPeers.Any()) return;

        var targetPeerAddress = otherPeers[_random.Next(otherPeers.Count)];

        try
        {
            _logger.LogInformation("Tentando fofocar com o par: {TargetPeer}", targetPeerAddress);
            var request = new GossipSyncRequest(Guid.NewGuid(), _nodeState.GetKnownPeers(), _nodeState.NodeJwt);

            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            var response =
                await _nodeClient.SendRequestAsync<GossipSyncResponse>(targetPeerAddress, request, cts.Token);

            _nodeState.MergePeers(response.KnownPeers);
            _logger.LogInformation("Fofoca com {TargetPeer} bem-sucedida. Total de pares conhecidos: {PeerCount}",
                targetPeerAddress, _nodeState.GetKnownPeers().Count);
        }
        catch (Exception ex)
        {
            _logger.LogWarning("Falha ao fofocar com o par {TargetPeer}. Razão: {ErrorType}", targetPeerAddress,
                ex.GetType().Name);
            _nodeState.RemovePeer(targetPeerAddress);
        }
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _timer?.Change(Timeout.Infinite, 0);
        return Task.CompletedTask;
    }

    public void Dispose() => _timer?.Dispose();
}