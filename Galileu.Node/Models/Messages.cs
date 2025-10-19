// Localização: Galileu.Node.Models/Messages.cs

using System.Text.Json.Serialization;
using Akka.Actor;

namespace Galileu.Node.Models;

// ####################################################################
// ## MENSAGENS PARA A REDE P2P (WebSocket)                          ##
// ####################################################################

[JsonDerivedType(typeof(AuthRequest), typeDiscriminator: "auth_request")]
[JsonDerivedType(typeof(AuthResponse), typeDiscriminator: "auth_response")]
[JsonDerivedType(typeof(GossipSyncRequest), typeDiscriminator: "gossip_sync_request")]
[JsonDerivedType(typeof(GossipSyncResponse), typeDiscriminator: "gossip_sync_response")]
[JsonDerivedType(typeof(PingRequest), typeDiscriminator: "ping_request")]
[JsonDerivedType(typeof(PongResponse), typeDiscriminator: "pong_response")]
public abstract record Message(Guid CorrelationId);

// --- Requisições P2P ---
public record AuthRequest(Guid CorrelationId, string NodeJwt) : Message(CorrelationId);

public record GossipSyncRequest(Guid CorrelationId, List<string> KnownPeers, string AuthToken) : Message(CorrelationId);

public record PingRequest(Guid CorrelationId, string FromNodeId, string AuthToken) : Message(CorrelationId);

// --- Respostas P2P ---
public record AuthResponse(Guid CorrelationId, bool Success, string Message) : Message(CorrelationId);

public record GossipSyncResponse(Guid CorrelationId, List<string> KnownPeers) : Message(CorrelationId);

public record PongResponse(Guid CorrelationId, string Message) : Message(CorrelationId);

// ####################################################################
// ## MENSAGENS PARA O CONSENSO DE ATORES (Akka.NET)                 ##
// ## Estes não precisam herdar de Message, pois são usados apenas   ##
// ## internamente pelo Akka.NET e não são serializados para a rede P2P. ##
// ####################################################################

public record ProcessTaskRequest(string Prompt, Guid TaskId);

public record StartConsensusRound(Guid TaskId, string[] Subtasks, IActorRef Worker, List<IActorRef> SelectedNodes);

public record ProcessSubtask(Guid TaskId, int SubtaskIndex, string Content);

public record SubtaskResult(Guid TaskId, int SubtaskIndex, string Fragment, IActorRef Node);

public record RequestValidation(Guid TaskId, SubtaskResult FragmentToValidate);

public record ValidationVote(Guid TaskId, int SubtaskIndex, bool IsValid, IActorRef Voter);

public record ConsensusReached(Guid TaskId, List<SubtaskResult> ValidatedFragments);

public record ConsensusFailed(Guid TaskId, string Reason);