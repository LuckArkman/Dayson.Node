﻿using System.Net;
using Galileu.Node.Models;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using System.Net.Http.Json;
using System.Net.Sockets;
using Galileu.Node.Gpu;
using Galileu.Node.Services;
using Microsoft.AspNetCore.Http;
using Services;
using System.Globalization;
using Galileu.Node.Brain;
using Galileu.Node.Core;

var builder = WebApplication.CreateBuilder(args);
var port = GetAvailablePort();

int GetAvailablePort()
{
    using (var listener = new TcpListener(IPAddress.Loopback, 0))
    {
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }
}

var myAddress = $"http://localhost:{port}";
Console.WriteLine($" === Swagger Acess http://localhost:{port}/swagger/index.html ===");
builder.WebHost.UseUrls(myAddress);
builder.Services.AddSingleton<PrimingService>();
builder.Services.AddSingleton(new NodeState(myAddress));
builder.Services.AddSingleton<PolymorphicTypeResolver>();
builder.Services.AddSingleton(provider => new NodeClient(provider.GetRequiredService<PolymorphicTypeResolver>()));
builder.Services.AddHostedService<GossipService>();
builder.Services.AddSingleton<NodeRegistryService>();
builder.Services.AddSingleton<GenerativeService>();
builder.Services.AddSingleton<WalletService>();
builder.Services.AddSingleton<ActorSystemSingleton>();
builder.Services.AddHostedService<AkkaHostedService>();

builder.Services.AddControllers();

builder.Services.AddEndpointsApiExplorer();

builder.Services.AddSwaggerGen(options =>
{
    options.SwaggerDoc("v1", new Microsoft.OpenApi.Models.OpenApiInfo
    {
        Title = "Dyson Node API",
        Version = "v1"
    });
});

var app = builder.Build();

app.UseSwagger();
app.UseSwaggerUI(options =>
{
    options.SwaggerEndpoint("/swagger/v1/swagger.json", "Dyson Node API V1");
    options.RoutePrefix = "swagger";
});

app.UseWebSockets();
app.MapControllers();

app.MapGet("/ws", async (HttpContext context) => { });

async Task<Message?> HandleMessage(Message? receivedMessage, NodeState state, NodeClient client)
{
    switch (receivedMessage)
    {
        case GossipSyncRequest gossipRequest:
            state.MergePeers(gossipRequest.KnownPeers);
            return new GossipSyncResponse(gossipRequest.CorrelationId, state.GetKnownPeers());
        default: return null;
    }
}

_ = BootstrapNodeAsync(app.Services, args);

app.Run();


async Task BootstrapNodeAsync(IServiceProvider services, string[] args)
{
    var nodeState = services.GetRequiredService<NodeState>();
    await Task.Delay(2000);

    var bootstrapApiAddress = args.Length > 0 ? args[0] : "http://localhost:5001";
    Console.WriteLine($"Iniciando registro com o Orquestrador em {bootstrapApiAddress}...");

    var (publicKey, _) = CryptoUtils.GenerateKeyPair();
    var normalizedPublicKey = CryptoUtils.NormalizePublicKey(publicKey);

    using var apiClient = new HttpClient { BaseAddress = new Uri(bootstrapApiAddress) };
    try
    {
        var registrationRequest = new NodeRegistrationRequest(normalizedPublicKey, nodeState.Address);
        var response = await apiClient.PostAsJsonAsync("/api/auth/register-node", registrationRequest);
        response.EnsureSuccessStatusCode();

        var regResponse = await response.Content.ReadFromJsonAsync<NodeRegistrationResponse>();
        if (string.IsNullOrEmpty(regResponse?.NodeJwt)) throw new Exception("Orquestrador não retornou um JWT.");

        nodeState.NodeJwt = regResponse.NodeJwt;
        nodeState.MergePeers(regResponse.InitialPeers);

        Console.WriteLine("Nó validado pelo Orquestrador. JWT recebido e rede P2P iniciada.");
        nodeState.PrintStatus();
    }
    catch (Exception ex)
    {
        Console.WriteLine(
            $"ERRO CRÍTICO no registro: {ex.Message}. O nó não pode se juntar à rede e permanecerá em modo de espera.");
    }

    var _generativeService = services.GetRequiredService<GenerativeService>();
    string modelPath = Path.Combine(Environment.CurrentDirectory, "Dayson", "Dayson.json");
    var datasetPath = Path.Combine(Environment.CurrentDirectory, "Dayson", "pt_0.txt");

    Console.WriteLine($"Arquivo de dataset existe >> {File.Exists(datasetPath)}");
    if (!File.Exists(modelPath))
    {
        if (!File.Exists(datasetPath))
        {
            Console.WriteLine($"Arquivo de dataset não encontrado em: {datasetPath}");
            return;
        }

        Console.WriteLine("\n========================================");
        Console.WriteLine("CONFIGURAÇÃO DE TREINAMENTO OTIMIZADA");
        Console.WriteLine("MODO: LONGO PRAZO (100 ÉPOCAS / 20 DIAS)");
        Console.WriteLine("========================================");

        var vocabManager = new VocabularyManager();
        int vocabSize = vocabManager.BuildVocabulary(datasetPath, maxVocabSize: 20000);

        if (vocabSize == 0)
        {
            Console.WriteLine("Falha ao construir ou carregar o vocabulário.");
            return;
        }

        // === NOVO: Configuração para treinamento longo ===
        var trainer = new Trainer(
            datasetPath,
            epochs: 25, // 100 épocas conforme requisito
            learningRate: 0.002f,
            validationSplit: 0.1f,
            batchSize: 16 // Reduzido de 32 para 24 (menos memória)
        );
        const int HIDDEN_SIZE = 192;
        const int EMBEDDING_SIZE = 96;
        const int CONTEXT_WINDOW = 6;

        Console.WriteLine($"\n[Config] Vocabulário: {vocabSize} tokens");
        Console.WriteLine($"[Config] Hidden Size: {HIDDEN_SIZE}");
        Console.WriteLine($"[Config] Embedding Size: {EMBEDDING_SIZE}");
        Console.WriteLine($"[Config] Épocas: {trainer.epochs}");
        Console.WriteLine($"[Config] Learning Rate: {trainer.learningRate}");
        Console.WriteLine($"[Config] Batch Size: {trainer.batchSize} (otimizado para memória)");
        Console.WriteLine($"[Config] Validação: {trainer.validationSplit * 100}%");
        Console.WriteLine($"[Config] META: RAM < 10GB durante 20 dias\n");

        // === NOVO: Inicia MemoryMonitor ===
        using var memoryMonitor = new MemoryMonitor();

        // Configura callbacks automáticos
        memoryMonitor.OnWarning = () =>
        {
            Console.WriteLine("[MemoryMonitor] Callback: Executando GC otimizado...");
            GC.Collect(1, GCCollectionMode.Optimized, false);
        };

        memoryMonitor.OnCritical = () =>
        {
            Console.WriteLine("[MemoryMonitor] Callback: Executando GC agressivo...");
            GC.Collect(2, GCCollectionMode.Forced, true, true);
            GC.WaitForPendingFinalizers();
        };

        memoryMonitor.OnEmergency = () =>
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("[MemoryMonitor] EMERGÊNCIA: Considere reduzir batch size ou pausar treinamento!");
            Console.ResetColor();
        };

        // Inicia monitoramento a cada 30 segundos
        memoryMonitor.Start(intervalSeconds: 30);

        Console.WriteLine("========================================");
        Console.WriteLine("INICIANDO TREINAMENTO");
        Console.WriteLine("Duração estimada: 20 dias");
        Console.WriteLine("Monitoramento de memória: ATIVO");
        Console.WriteLine("========================================\n");

        try
        {
            await _generativeService.TrainModelAsync(trainer, vocabSize, EMBEDDING_SIZE, HIDDEN_SIZE, CONTEXT_WINDOW);

            Console.WriteLine("\n========================================");
            Console.WriteLine("TREINAMENTO CONCLUÍDO COM SUCESSO!");
            Console.WriteLine("========================================");
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"\n[ERRO] Treinamento falhou: {ex.Message}");
            Console.WriteLine($"Stack Trace: {ex.StackTrace}");
            Console.ResetColor();
        }
        finally
        {
            // Para monitor e mostra relatório
            memoryMonitor.Stop();
        }
    }
    else
    {
        Console.WriteLine($"[Bootstrap] Modelo encontrado em {modelPath}. Pulando treinamento.");
        _generativeService.InitializeFromDisk();
    }
}