// --- START OF FILE Program.cs (VERSÃO FINAL E ATUALIZADA) ---

using System.Net;
using Galileu.Node.Models;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using System.Net.Http.Json;
using System.Net.Sockets;
using Galileu.Node.Services;
using Microsoft.AspNetCore.Http;
using Services;
using Galileu.Node.Brain;

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
builder.Services.AddSingleton<MongoDbService>(); 
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

    if (!File.Exists(modelPath))
    {
        if (!File.Exists(datasetPath))
        {
            Console.WriteLine($"Arquivo de dataset não encontrado em: {datasetPath}");
            return;
        }

        Console.WriteLine("\n=============================================");
        Console.WriteLine("CONFIGURAÇÃO DE TREINAMENTO HÍBRIDO CONTÍNUO");
        Console.WriteLine("MODO: LONGO PRAZO (SUPERVISIONADO + REFORÇO)");
        Console.WriteLine("=============================================");

        // A construção do vocabulário agora é feita dentro do GenerativeService
        // para garantir que o modelo seja inicializado corretamente.

        // === CORREÇÃO: Configuração para treinamento híbrido longo ===
        var trainerOptions = new Trainer(
            datasetPath,
            epochs: 100,          // 100 épocas no total
            learningRate: 0.001,  // Taxa de aprendizado inicial para SFT
            validationSplit: 0.1, // Menos validação, mais dados para treino
            batchSize: 24         // Otimizado para memória
        );

        Console.WriteLine($"\n[Config] Dataset: {Path.GetFileName(datasetPath)}");
        Console.WriteLine($"[Config] Total de Épocas: {trainerOptions.epochs}");
        Console.WriteLine($"[Config] Learning Rate (SFT): {trainerOptions.learningRate}");
        Console.WriteLine($"[Config] Batch Size: {trainerOptions.batchSize}");
        Console.WriteLine($"[Config] META: RAM < 10GB e aprendizado contínuo.\n");
        
        using var memoryMonitor = new MemoryMonitor();
        memoryMonitor.OnWarning = () => GC.Collect(1, GCCollectionMode.Optimized, false);
        memoryMonitor.OnCritical = () => { GC.Collect(2, GCCollectionMode.Forced, true, true); GC.WaitForPendingFinalizers(); };
        memoryMonitor.OnEmergency = () => {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("[MemoryMonitor] EMERGÊNCIA: Considere reduzir batch size ou pausar treinamento!");
            Console.ResetColor();
        };
        memoryMonitor.Start(intervalSeconds: 30);

        Console.WriteLine("========================================");
        Console.WriteLine("INICIANDO TREINAMENTO HÍBRIDO");
        Console.WriteLine("Monitoramento de memória: ATIVO");
        Console.WriteLine("========================================\n");
        
        Console.WriteLine("AVISO IMPORTANTE: Certifique-se de que sua chave de API e endpoint do modelo professor");
        Console.WriteLine("estejam configurados corretamente no arquivo 'TeacherModelService.cs'.\n");

        try
        {
            // === CORREÇÃO PRINCIPAL: Chama o método de treinamento híbrido ===
            await _generativeService.TrainWithTeacherAsync(trainerOptions);

            Console.WriteLine("\n========================================");
            Console.WriteLine("TREINAMENTO HÍBRIDO CONCLUÍDO COM SUCESSO!");
            Console.WriteLine("========================================");
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"\n[ERRO] Treinamento híbrido falhou: {ex.Message}");
            Console.WriteLine($"Stack Trace: {ex.StackTrace}");
            Console.ResetColor();
        }
        finally
        {
            memoryMonitor.Stop();
        }
    }
    else
    {
        Console.WriteLine($"[Bootstrap] Modelo encontrado em {modelPath}. Pulando treinamento.");
        _generativeService.InitializeFromDisk();
    }
}