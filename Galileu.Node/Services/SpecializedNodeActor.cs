using Akka.Actor;
using Galileu.Node.Models; // Usar o namespace corrigido
using Galileu.Services;
using Services; // Usar o namespace corrigido

namespace Galileu.Services;

public class SpecializedNodeActor : ReceiveActor
{
    private readonly NodeRegistryService _registryService;
    private readonly NodeState _nodeState;
    private readonly IEnumerable<string> _specializations;
    private readonly string _myWalletAddress;
    private readonly string _nodeId;

    public SpecializedNodeActor(
        IEnumerable<string> specializations,
        NodeRegistryService registryService,
        NodeState nodeState)
    {
        _specializations = specializations;
        _registryService = registryService;
        _nodeState = nodeState;
        _nodeId = $"node-{_nodeState.Id.Substring(0, 8)}";

        var (publicKey, _) = CryptoUtils.GenerateKeyPair();
        _myWalletAddress = CryptoUtils.NormalizePublicKey(publicKey);

        // Agora o compilador encontrará estes tipos porque eles não estão mais aninhados
        Receive<ProcessSubtask>(msg =>
        {
            var specsString = string.Join(", ", _specializations);
            Console.WriteLine($"[Actor {_nodeId}/{specsString}] Recebeu subtarefa '{msg.Content}'.");

            var fragment = $"<fragment_from:{_nodeId}>";

            Sender.Tell(new SubtaskResult(msg.TaskId, msg.SubtaskIndex, fragment, Self));
        });

        Receive<RequestValidation>(msg =>
        {
            Console.WriteLine(
                $"[Actor {_nodeId}] Recebeu pedido para validar o fragmento: '{msg.FragmentToValidate.Fragment}'");

            bool isValid = !string.IsNullOrWhiteSpace(msg.FragmentToValidate.Fragment);
            Console.WriteLine($"[Actor {_nodeId}] Votando. Voto: {(isValid ? "Válido" : "Inválido")}");

            Sender.Tell(new ValidationVote(msg.TaskId, msg.FragmentToValidate.SubtaskIndex, isValid, Self));
        });
    }

    protected override void PreStart()
    {
        // Esta chamada agora corresponde à assinatura correta de 4 parâmetros
        _registryService.RegisterNode(Self, _myWalletAddress, _nodeState.Address, _specializations);
        base.PreStart();
    }

    protected override void PostStop()
    {
        _registryService.UnregisterNode(Self);
        base.PostStop();
    }
}