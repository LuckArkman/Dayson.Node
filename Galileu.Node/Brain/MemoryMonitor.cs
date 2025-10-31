using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

namespace Galileu.Node.Brain;

/// <summary>
/// Monitor contínuo de memória para treinamentos de longo prazo.
/// Alerta quando RAM ultrapassa limites críticos.
/// </summary>
public class MemoryMonitor : IDisposable
{
    private readonly Process _process;
    private readonly CancellationTokenSource _cts;
    private Task? _monitoringTask;
    private bool _disposed = false;
    
    // Limites configuráveis
    private const long WARNING_THRESHOLD_MB = 100000;  // 8GB
    private const long CRITICAL_THRESHOLD_MB = 150000; // 9.5GB
    private const long EMERGENCY_THRESHOLD_MB = 250000; // 10GB
    
    // Estatísticas
    private long _peakMemoryMB = 0;
    private int _warningCount = 0;
    private int _criticalCount = 0;
    private DateTime _startTime;
    
    // Callbacks para ações automáticas
    public Action? OnWarning { get; set; }
    public Action? OnCritical { get; set; }
    public Action? OnEmergency { get; set; }

    public MemoryMonitor()
    {
        _process = Process.GetCurrentProcess();
        _cts = new CancellationTokenSource();
        _startTime = DateTime.UtcNow;
    }

    /// <summary>
    /// Inicia monitoramento em background.
    /// </summary>
    public void Start(int intervalSeconds = 30)
    {
        if (_monitoringTask != null)
            throw new InvalidOperationException("Monitor já está rodando.");
        
        //Console.WriteLine($"[MemoryMonitor] Iniciado (intervalo: {intervalSeconds}s)");
        //Console.WriteLine($"[MemoryMonitor] Limites: Aviso={WARNING_THRESHOLD_MB}MB, Crítico={CRITICAL_THRESHOLD_MB}MB, Emergência={EMERGENCY_THRESHOLD_MB}MB");
        
        _monitoringTask = Task.Run(async () =>
        {
            while (!_cts.Token.IsCancellationRequested)
            {
                CheckMemory();
                
                try
                {
                    await Task.Delay(TimeSpan.FromSeconds(intervalSeconds), _cts.Token);
                }
                catch (TaskCanceledException)
                {
                    break;
                }
            }
        });
    }

    /// <summary>
    /// Para monitoramento e imprime relatório final.
    /// </summary>
    public void Stop()
    {
        if (_monitoringTask == null)
            return;
        
        _cts.Cancel();
        _monitoringTask.Wait(TimeSpan.FromSeconds(5));
        
        PrintFinalReport();
    }

    /// <summary>
    /// Verifica memória atual e dispara alertas.
    /// </summary>
    private void CheckMemory()
    {
        _process.Refresh();
        long currentMemoryMB = _process.WorkingSet64 / (1024 * 1024);
        
        if (currentMemoryMB > _peakMemoryMB)
            _peakMemoryMB = currentMemoryMB;
        
        // Log normal
        //Console.WriteLine($"[MemoryMonitor] RAM: {currentMemoryMB}MB | Pico: {_peakMemoryMB}MB | Uptime: {GetUptime()}");
        
        // Verifica thresholds
        if (currentMemoryMB >= EMERGENCY_THRESHOLD_MB)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"[EMERGÊNCIA] RAM atingiu {currentMemoryMB}MB! LIMITE CRÍTICO!");
            Console.ResetColor();
            
            OnEmergency?.Invoke();
            
            // Ação drástica: GC forçado
            Console.WriteLine("[MemoryMonitor] Executando GC de emergência...");
            GC.Collect(2, GCCollectionMode.Forced, true, true);
            GC.WaitForPendingFinalizers();
            GC.Collect(2, GCCollectionMode.Forced, true, true);
        }
        else if (currentMemoryMB >= CRITICAL_THRESHOLD_MB)
        {
            _criticalCount++;
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"[CRÍTICO] RAM em {currentMemoryMB}MB - Próximo do limite!");
            Console.ResetColor();
            
            OnCritical?.Invoke();
        }
        else if (currentMemoryMB >= WARNING_THRESHOLD_MB)
        {
            _warningCount++;
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"[AVISO] RAM em {currentMemoryMB}MB - Aproximando do limite");
            Console.ResetColor();
            
            OnWarning?.Invoke();
        }
    }

    /// <summary>
    /// Força verificação imediata (chamável externamente).
    /// </summary>
    public long CheckNow()
    {
        _process.Refresh();
        return _process.WorkingSet64 / (1024 * 1024);
    }

    /// <summary>
    /// Retorna tempo desde início do monitoramento.
    /// </summary>
    private string GetUptime()
    {
        var elapsed = DateTime.UtcNow - _startTime;
        return $"{elapsed.Days}d {elapsed.Hours:D2}h {elapsed.Minutes:D2}m";
    }

    /// <summary>
    /// Imprime relatório final de execução.
    /// </summary>
    private void PrintFinalReport()
    {
        Console.WriteLine("\n" + new string('═', 60));
        Console.WriteLine("RELATÓRIO FINAL - MEMORY MONITOR");
        Console.WriteLine(new string('═', 60));
        Console.WriteLine($"Duração total: {GetUptime()}");
        Console.WriteLine($"Pico de RAM: {_peakMemoryMB}MB");
        Console.WriteLine($"Avisos: {_warningCount}");
        Console.WriteLine($"Críticos: {_criticalCount}");
        
        if (_peakMemoryMB < WARNING_THRESHOLD_MB)
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("✓ Memória mantida abaixo de 8GB - EXCELENTE");
        }
        else if (_peakMemoryMB < CRITICAL_THRESHOLD_MB)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("⚠ Memória entre 8-9.5GB - ACEITÁVEL");
        }
        else
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("✗ Memória excedeu 9.5GB - REQUER OTIMIZAÇÃO");
        }
        Console.ResetColor();
        Console.WriteLine(new string('═', 60) + "\n");
    }

    public void Dispose()
    {
        if (_disposed) return;
        
        Stop();
        _cts?.Dispose();
        _disposed = true;
    }
}