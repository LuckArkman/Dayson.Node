using System;
using System.Diagnostics;
using System.Threading;

namespace Galileu.Node.Brain;

/// <summary>
/// Monitora a carga da GPU e ajusta dinamicamente parâmetros de treinamento.
/// Previne sobrecarga (>100%) através de throttling adaptativo, com foco em
/// manter a estabilidade térmica.
/// </summary>
public class GpuLoadMonitor
{
    private int _currentBatchSize;
    private int _throttleDelayMs;
    private double _averageUtilization;
    private readonly object _lock = new object();
    
    // --- AJUSTE FINO: Thresholds mais conservadores para priorizar o resfriamento ---
    private const double HIGH_LOAD_THRESHOLD = 90.0;     // Age mais cedo, a partir de 90%
    private const double CRITICAL_LOAD_THRESHOLD = 105.0;  // Mantido para emergências
    private const double OPTIMAL_LOAD_TARGET = 80.0;     // Tenta manter a GPU em uma carga mais confortável de 80%
    
    // Limites de ajuste
    private const int MIN_BATCH_SIZE = 8;
    private const int MAX_BATCH_SIZE = 64;
    private const int MIN_THROTTLE_MS = 0;
    private const int MAX_THROTTLE_MS = 100; // Aumentado para permitir pausas maiores se necessário

    public GpuLoadMonitor(int initialBatchSize)
    {
        _currentBatchSize = initialBatchSize;
        _throttleDelayMs = 1; // Padrão
        _averageUtilization = 0;
    }

    public int CurrentBatchSize
    {
        get { lock (_lock) return _currentBatchSize; }
    }

    public int ThrottleDelayMs
    {
        get { lock (_lock) return _throttleDelayMs; }
    }

    /// <summary>
    /// Atualiza estatísticas de utilização e ajusta parâmetros se necessário.
    /// Deve ser chamado após cada lote (batch) de treinamento.
    /// </summary>
    public void RecordUtilization(double utilizationPercent, double batchTimeSeconds)
    {
        lock (_lock)
        {
            const double ALPHA = 0.3;
            _averageUtilization = ALPHA * utilizationPercent + (1 - ALPHA) * _averageUtilization;

            if (_averageUtilization > CRITICAL_LOAD_THRESHOLD)
            {
                // CRISE: Reduz batch size agressivamente e aumenta muito a pausa.
                _currentBatchSize = Math.Max(MIN_BATCH_SIZE, _currentBatchSize - 8);
                _throttleDelayMs = Math.Min(MAX_THROTTLE_MS, _throttleDelayMs + 20);
                
                Console.ForegroundColor = ConsoleColor.Red;
                //Console.WriteLine($"\n[GPU Monitor] CRÍTICO: {_averageUtilization:F1}% → Batch: {_currentBatchSize}, Throttle: {_throttleDelayMs}ms");
                Console.ResetColor();
            }
            else if (_averageUtilization > HIGH_LOAD_THRESHOLD)
            {
                // ALTA CARGA: Ajuste moderado, priorizando o aumento do throttle (pausa).
                _throttleDelayMs = Math.Min(MAX_THROTTLE_MS, _throttleDelayMs + 10);
                _currentBatchSize = Math.Max(MIN_BATCH_SIZE, _currentBatchSize - 4);
                
                Console.ForegroundColor = ConsoleColor.Yellow;
                //Console.WriteLine($"\n[GPU Monitor] ALTA CARGA: {_averageUtilization:F1}% → Batch: {_currentBatchSize}, Throttle: {_throttleDelayMs}ms");
                Console.ResetColor();
            }
            else if (_averageUtilization < OPTIMAL_LOAD_TARGET && batchTimeSeconds < 0.5)
            {
                // SUB-UTILIZAÇÃO: Pode aumentar carga gradualmente, priorizando diminuir a pausa.
                if (_throttleDelayMs > MIN_THROTTLE_MS)
                {
                    _throttleDelayMs = Math.Max(MIN_THROTTLE_MS, _throttleDelayMs - 5);
                }
                else if (_currentBatchSize < MAX_BATCH_SIZE)
                {
                    _currentBatchSize = Math.Min(MAX_BATCH_SIZE, _currentBatchSize + 2);
                }
            }
        }
    }

    /// <summary>
    /// Aplica a pausa (throttle) calculada dinamicamente.
    /// </summary>
    public void ApplyThrottle()
    {
        int delay;
        lock (_lock) { delay = _throttleDelayMs; }
        
        if (delay > 0)
        {
            Thread.Sleep(delay);
        }
    }

    /// <summary>
    /// Simula medição de utilização da GPU.
    /// Em produção, substitua por chamadas reais ao driver OpenCL.
    /// </summary>
    public static double MeasureGpuUtilization()
    {
        try
        {
            using (var process = Process.GetCurrentProcess())
            {
                double memoryMB = process.WorkingSet64 / (1024.0 * 1024.0);
                // Heurística simples: assume que >2GB indica alta utilização
                return Math.Min(150.0, (memoryMB / 2000.0) * 100.0);
            }
        }
        catch
        {
            return 50.0; // Fallback
        }
    }

    public void PrintStatus()
    {
        lock (_lock)
        {
            //Console.WriteLine($"[GPU Monitor] Utilização Média: {_averageUtilization:F1}% | Batch Size: {_currentBatchSize} | Throttle: {_throttleDelayMs}ms");
        }
    }
}