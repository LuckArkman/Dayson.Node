using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using Galileu.Node.Models;

namespace Galileu.Node.Models;

public class NodeClient
{
    private readonly PolymorphicTypeResolver _typeResolver;
    private readonly JsonSerializerOptions _jsonOptions;

    public NodeClient(PolymorphicTypeResolver typeResolver)
    {
        _typeResolver = typeResolver;
        _jsonOptions = new JsonSerializerOptions { TypeInfoResolver = _typeResolver };
    }

    public async Task<TResponse> SendRequestAsync<TResponse>(string targetAddress, Message request,
        CancellationToken token)
        where TResponse : Message
    {
        using var ws = new ClientWebSocket();
        var wsUrl = new Uri(targetAddress.Replace("http://", "ws://").Replace("https://", "wss://") + "/ws");

        await ws.ConnectAsync(wsUrl, token);

        var jsonMessage = JsonSerializer.Serialize(request, _jsonOptions);
        var sendBuffer = new ArraySegment<byte>(Encoding.UTF8.GetBytes(jsonMessage));
        await ws.SendAsync(sendBuffer, WebSocketMessageType.Text, true, token);

        var receiveBuffer = new ArraySegment<byte>(new byte[4096]);
        var result = await ws.ReceiveAsync(receiveBuffer, token);

        if (receiveBuffer.Array == null)
        {
            throw new InvalidOperationException("O buffer de recebimento está vazio.");
        }

        var jsonResponse = Encoding.UTF8.GetString(receiveBuffer.Array, 0, result.Count);
        var responseMessage = JsonSerializer.Deserialize<Message>(jsonResponse, _jsonOptions);

        return (TResponse)responseMessage!;
    }
}