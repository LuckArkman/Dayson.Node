using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Gpu;

/// <summary>
/// Rastreador de alocações de memória GPU para detectar vazamentos.
/// Monitora criação e liberação de tensores, identificando leaks em tempo real.
/// </summary>
public class GpuMemoryTracker : IDisposable
{
    private readonly ConcurrentDictionary<int, TensorAllocation> _activeTensors;
    private readonly ReaderWriterLockSlim _lock;
    private int _nextId = 0;
    private long _totalAllocatedBytes = 0;
    private long _peakMemoryBytes = 0;
    private int _totalAllocations = 0;
    private int _totalDeallocations = 0;
    private bool _disposed = false;
    
    // Configuração de alertas
    private const long WARNING_THRESHOLD_MB = 2048;  // 2GB
    private const long CRITICAL_THRESHOLD_MB = 3072; // 3GB
    private const int LEAK_DETECTION_INTERVAL_MS = 30000; // 30 segundos
    
    private Timer? _leakDetectionTimer;
    
    public GpuMemoryTracker()
    {
        _activeTensors = new ConcurrentDictionary<int, TensorAllocation>();
        _lock = new ReaderWriterLockSlim();
        
        // Inicia detector de vazamentos em background
        _leakDetectionTimer = new Timer(DetectLeaks, null, 
            LEAK_DETECTION_INTERVAL_MS, LEAK_DETECTION_INTERVAL_MS);
    }
    
    /// <summary>
    /// Registra alocação de um novo tensor.
    /// </summary>
    public int TrackAllocation(IMathTensor tensor, string location, int[] shape)
    {
        _lock.EnterWriteLock();
        try
        {
            int id = Interlocked.Increment(ref _nextId);
            long sizeBytes = CalculateTensorSize(shape);
            
            var allocation = new TensorAllocation
            {
                Id = id,
                Tensor = new WeakReference<IMathTensor>(tensor),
                Shape = shape,
                SizeBytes = sizeBytes,
                AllocationTime = DateTime.UtcNow,
                AllocationLocation = location,
                StackTrace = Environment.StackTrace
            };
            
            _activeTensors[id] = allocation;
            _totalAllocatedBytes += sizeBytes;
            _totalAllocations++;
            
            if (_totalAllocatedBytes > _peakMemoryBytes)
                _peakMemoryBytes = _totalAllocatedBytes;
            
            // Verifica thresholds
            CheckMemoryThresholds();
            
            return id;
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }
    
    /// <summary>
    /// Registra liberação de um tensor.
    /// </summary>
    public void TrackDeallocation(int tensorId)
    {
        _lock.EnterWriteLock();
        try
        {
            if (_activeTensors.TryRemove(tensorId, out var allocation))
            {
                _totalAllocatedBytes -= allocation.SizeBytes;
                _totalDeallocations++;
            }
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }
    
    /// <summary>
    /// Verifica se há tensores não liberados há muito tempo.
    /// </summary>
    private void DetectLeaks(object? state)
    {
        _lock.EnterReadLock();
        try
        {
            var now = DateTime.UtcNow;
            var suspectedLeaks = _activeTensors.Values
                .Where(a => (now - a.AllocationTime).TotalMinutes > 5)
                .OrderByDescending(a => a.SizeBytes)
                .Take(10)
                .ToList();
            
            if (suspectedLeaks.Any())
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"\n⚠️  [GPU Memory] POSSÍVEL VAZAMENTO DETECTADO!");
                Console.WriteLine($"    Tensores ativos: {_activeTensors.Count}");
                Console.WriteLine($"    Memória alocada: {_totalAllocatedBytes / (1024.0 * 1024.0):F2} MB");
                Console.WriteLine($"\n    Top 10 tensores suspeitos:");
                
                foreach (var leak in suspectedLeaks)
                {
                    var age = (now - leak.AllocationTime).TotalMinutes;
                    Console.WriteLine($"      • ID {leak.Id}: {leak.SizeBytes / (1024.0 * 1024.0):F2} MB " +
                                      $"há {age:F1} min - {leak.AllocationLocation}");
                }
                Console.ResetColor();
            }
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }
    
    /// <summary>
    /// Verifica thresholds de memória e emite alertas.
    /// </summary>
    private void CheckMemoryThresholds()
    {
        long currentMB = _totalAllocatedBytes / (1024 * 1024);
        
        if (currentMB >= CRITICAL_THRESHOLD_MB)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"\n🔴 [GPU Memory] CRÍTICO: {currentMB} MB alocados!");
            Console.WriteLine($"    Tensores ativos: {_activeTensors.Count}");
            PrintTopAllocations(5);
            Console.ResetColor();
        }
        else if (currentMB >= WARNING_THRESHOLD_MB)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"\n⚠️  [GPU Memory] AVISO: {currentMB} MB alocados");
            Console.ResetColor();
        }
    }
    
    /// <summary>
    /// Imprime maiores alocações ativas.
    /// </summary>
    private void PrintTopAllocations(int count)
    {
        var top = _activeTensors.Values
            .OrderByDescending(a => a.SizeBytes)
            .Take(count)
            .ToList();
        
        Console.WriteLine($"    Top {count} maiores tensores:");
        foreach (var alloc in top)
        {
            var shape = string.Join("x", alloc.Shape);
            var age = (DateTime.UtcNow - alloc.AllocationTime).TotalSeconds;
            Console.WriteLine($"      • {alloc.SizeBytes / (1024.0 * 1024.0):F2} MB " +
                              $"[{shape}] há {age:F1}s - {alloc.AllocationLocation}");
        }
    }
    
    /// <summary>
    /// Força liberação de tensores órfãos (GC coletados mas não dispostos).
    /// </summary>
    public int CleanupOrphans()
    {
        _lock.EnterWriteLock();
        try
        {
            var orphans = _activeTensors
                .Where(kvp => !kvp.Value.Tensor.TryGetTarget(out _))
                .Select(kvp => kvp.Key)
                .ToList();
            
            foreach (var id in orphans)
            {
                if (_activeTensors.TryRemove(id, out var alloc))
                {
                    _totalAllocatedBytes -= alloc.SizeBytes;
                }
            }
            
            if (orphans.Count > 0)
            {
                Console.WriteLine($"[GPU Memory] Limpou {orphans.Count} tensores órfãos");
            }
            
            return orphans.Count;
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }
    
    /// <summary>
    /// Retorna estatísticas de memória.
    /// </summary>
    public MemoryStatistics GetStatistics()
    {
        _lock.EnterReadLock();
        try
        {
            return new MemoryStatistics
            {
                ActiveTensors = _activeTensors.Count,
                AllocatedBytes = _totalAllocatedBytes,
                PeakBytes = _peakMemoryBytes,
                TotalAllocations = _totalAllocations,
                TotalDeallocations = _totalDeallocations,
                LeakedTensors = _totalAllocations - _totalDeallocations
            };
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }
    
    /// <summary>
    /// Imprime relatório detalhado.
    /// </summary>
    public void PrintReport()
    {
        var stats = GetStatistics();
        
        Console.WriteLine("\n╔════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║           RELATÓRIO DE MEMÓRIA GPU                         ║");
        Console.WriteLine("╠════════════════════════════════════════════════════════════╣");
        Console.WriteLine($"║ Tensores ativos:        {stats.ActiveTensors,10:N0}                   ║");
        Console.WriteLine($"║ Memória alocada:        {stats.AllocatedBytes / (1024.0 * 1024.0),10:F2} MB              ║");
        Console.WriteLine($"║ Pico de memória:        {stats.PeakBytes / (1024.0 * 1024.0),10:F2} MB              ║");
        Console.WriteLine("╠════════════════════════════════════════════════════════════╣");
        Console.WriteLine($"║ Total de alocações:     {stats.TotalAllocations,10:N0}                   ║");
        Console.WriteLine($"║ Total de liberações:    {stats.TotalDeallocations,10:N0}                   ║");
        Console.WriteLine($"║ Tensores vazados:       {stats.LeakedTensors,10:N0}                   ║");
        Console.WriteLine("╚════════════════════════════════════════════════════════════╝\n");
        
        if (stats.ActiveTensors > 0)
        {
            PrintTopAllocations(10);
        }
    }
    
    private long CalculateTensorSize(int[] shape)
    {
        long elements = 1;
        foreach (var dim in shape)
            elements *= dim;
        return elements * sizeof(float);
    }
    
    public void Dispose()
    {
        if (_disposed) return;
        
        _leakDetectionTimer?.Dispose();
        _lock?.Dispose();
        _disposed = true;
    }
}

/// <summary>
/// Informações sobre uma alocação de tensor.
/// </summary>
public class TensorAllocation
{
    public int Id { get; set; }
    public WeakReference<IMathTensor> Tensor { get; set; } = null!;
    public int[] Shape { get; set; } = Array.Empty<int>();
    public long SizeBytes { get; set; }
    public DateTime AllocationTime { get; set; }
    public string AllocationLocation { get; set; } = string.Empty;
    public string StackTrace { get; set; } = string.Empty;
}

/// <summary>
/// Estatísticas de memória GPU.
/// </summary>
public class MemoryStatistics
{
    public int ActiveTensors { get; set; }
    public long AllocatedBytes { get; set; }
    public long PeakBytes { get; set; }
    public int TotalAllocations { get; set; }
    public int TotalDeallocations { get; set; }
    public int LeakedTensors { get; set; }
}