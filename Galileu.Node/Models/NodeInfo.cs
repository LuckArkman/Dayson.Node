using Akka.Actor;

namespace Galileu.Node.Models;

public record NodeInfo(
    IActorRef ActorRef,
    string walletAddress,
    string netAddress,
    HashSet<string> Specializations
    // As áreas de expertise do nó
);