using Akka.Actor;

namespace Galileu.Node.Services;

public class ActorSystemSingleton
{
    public ActorSystem? ActorSystem { get; set; }
}