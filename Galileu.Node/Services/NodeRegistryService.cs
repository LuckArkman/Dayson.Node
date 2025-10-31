using System.Collections.Concurrent;
using Akka.Actor;
using Galileu.Node.Models;

namespace Services;

public class NodeRegistryService
{
    // O dicionário armazena o objeto NodeInfo completo.
    private readonly ConcurrentDictionary<IActorRef, NodeInfo> _registry = new();

    // CORRIGIDO: A assinatura do método deve aceitar 4 parâmetros
    public void RegisterNode(IActorRef nodeActor, string walletAddress, string networkAddress,
        IEnumerable<string> specializations)
    {
        var nodeInfo = new NodeInfo(
            nodeActor,
            walletAddress,
            networkAddress,
            new HashSet<string>(specializations, StringComparer.OrdinalIgnoreCase)
        );
        _registry[nodeActor] = nodeInfo;
        Console.WriteLine($"[Registry] Nó {nodeActor.Path.Name} em {networkAddress} registrado.");
    }

    public string? GetWalletAddress(IActorRef nodeActor)
    {
        _registry.TryGetValue(nodeActor, out var info);
        return info?.walletAddress;
    }

    public void UnregisterNode(IActorRef nodeActor)
    {
        _registry.TryRemove(nodeActor, out _);
        Console.WriteLine($"[Registry] Nó {nodeActor.Path.Name} removido do registro.");
    }

    /// <summary>
    /// Consulta o registro e retorna os nós que possuem uma especialização específica.
    /// Esta é a função-chave para o Worker.
    /// </summary>
    public IEnumerable<NodeInfo> FindNodesBySpecialization(string specialization)
    {
        return _registry.Values
            .Where(node => node.Specializations.Contains(specialization));
    }

    /// <summary>
    /// Retorna uma quantidade de nós "genéricos" para tarefas que não têm uma especialização clara.
    /// </summary>
    public IEnumerable<NodeInfo> GetGeneralPurposeNodes(int count)
    {
        // Uma estratégia simples é pegar os primeiros 'count' nós.
        // Uma estratégia melhor poderia ser selecionar nós com menos carga ou com uma especialização "geral".
        return _registry.Values.Take(count);
    }
}