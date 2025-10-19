namespace Galileu.Node.Models;

public record NodeRegistrationResponse(string NodeJwt, IEnumerable<string> InitialPeers);