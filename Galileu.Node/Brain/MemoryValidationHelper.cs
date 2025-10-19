using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace Galileu.Node.Brain;

/// <summary>
/// Helper para validar se a memória está sendo liberada corretamente entre épocas.
/// </summary>
public class MemoryValidationHelper
{
    private readonly Process _process;
    private readonly string _logPath;
    private long _baselineMemoryMB;
    private long _previousEpochMemoryMB;
    private int _consecutiveLeaks = 0;

    public MemoryValidationHelper(string logPath)
    {
        _process = Process.GetCurrentProcess();
        _logPath = logPath;
        _baselineMemoryMB = GetCurrentMemoryMB();
    }

    /// <summary>
    /// Chama este método ANTES de iniciar cada época.
    /// </summary>
    public void RecordEpochStart(int epochNumber)
    {
        _previousEpochMemoryMB = GetCurrentMemoryMB();

        string message = $"\n[Época {epochNumber}] MEMÓRIA NO INÍCIO: {_previousEpochMemoryMB}MB";
        Console.WriteLine(message);
        File.AppendAllText(_logPath, message + Environment.NewLine);
    }

    /// <summary>
    /// Chama este método DEPOIS de liberar memória (após GC).
    /// </summary>
    public ValidationResult ValidateMemoryRelease(int epochNumber)
    {
        long currentMemoryMB = GetCurrentMemoryMB();
        long memoryDelta = currentMemoryMB - _previousEpochMemoryMB;
        double leakPercentage = ((double)memoryDelta / _previousEpochMemoryMB) * 100;

        var result = new ValidationResult
        {
            EpochNumber = epochNumber,
            MemoryBefore = _previousEpochMemoryMB,
            MemoryAfter = currentMemoryMB,
            Delta = memoryDelta,
            LeakPercentage = leakPercentage,
            BaselineMemory = _baselineMemoryMB
        };

        // Critérios de validação
        if (memoryDelta > 100) // Mais de 100MB de diferença
        {
            result.Status = ValidationStatus.Critical;
            result.Message = $"VAZAMENTO CRÍTICO: +{memoryDelta}MB retidos após GC!";
            _consecutiveLeaks++;
        }
        else if (memoryDelta > 50) // Entre 50-100MB
        {
            result.Status = ValidationStatus.Warning;
            result.Message = $"ALERTA: +{memoryDelta}MB não liberados completamente.";
            _consecutiveLeaks++;
        }
        else if (memoryDelta > -10 && memoryDelta < 10) // Variação normal (±10MB)
        {
            result.Status = ValidationStatus.Success;
            result.Message = $"OK: Memória estável ({memoryDelta:+#;-#;0}MB)";
            _consecutiveLeaks = 0;
        }
        else // Memória diminuiu (ideal!)
        {
            result.Status = ValidationStatus.Excellent;
            result.Message = $"EXCELENTE: {Math.Abs(memoryDelta)}MB liberados!";
            _consecutiveLeaks = 0;
        }

        // Log detalhado
        PrintValidationReport(result);
        LogToFile(result);

        // Alerta de vazamento contínuo
        if (_consecutiveLeaks >= 3)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"\n⚠️  ALERTA: {_consecutiveLeaks} épocas consecutivas com vazamento de memória!");
            Console.WriteLine("   Possível causa: Recursos não estão sendo descartados corretamente.");
            Console.ResetColor();
        }

        return result;
    }

    private void PrintValidationReport(ValidationResult result)
    {
        Console.WriteLine("\n╔════════════════════════════════════════════════════════════╗");
        Console.WriteLine($"║  VALIDAÇÃO DE MEMÓRIA - ÉPOCA {result.EpochNumber,3}                      ║");
        Console.WriteLine("╠════════════════════════════════════════════════════════════╣");

        Console.Write("║  Status: ");
        switch (result.Status)
        {
            case ValidationStatus.Excellent:
                Console.ForegroundColor = ConsoleColor.Green;
                Console.Write("✓ EXCELENTE");
                break;
            case ValidationStatus.Success:
                Console.ForegroundColor = ConsoleColor.Green;
                Console.Write("✓ APROVADO");
                break;
            case ValidationStatus.Warning:
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.Write("⚠ ALERTA");
                break;
            case ValidationStatus.Critical:
                Console.ForegroundColor = ConsoleColor.Red;
                Console.Write("✗ CRÍTICO");
                break;
        }

        Console.ResetColor();
        Console.WriteLine(new string(' ', 47 - result.Status.ToString().Length) + "║");

        Console.WriteLine($"║  Memória ANTES do GC: {result.MemoryBefore,6}MB{new string(' ', 27)}║");
        Console.WriteLine($"║  Memória APÓS o GC:   {result.MemoryAfter,6}MB{new string(' ', 27)}║");
        Console.WriteLine(
            $"║  Delta:               {result.Delta,6:+#;-#;0}MB ({result.LeakPercentage,5:+0.0;-0.0;0.0}%){new string(' ', 16)}║");
        Console.WriteLine($"║  Baseline (Época 1):  {result.BaselineMemory,6}MB{new string(' ', 27)}║");
        Console.WriteLine("╠════════════════════════════════════════════════════════════╣");
        Console.WriteLine($"║  {result.Message.PadRight(58)}║");
        Console.WriteLine("╚════════════════════════════════════════════════════════════╝\n");
    }

    private void LogToFile(ValidationResult result)
    {
        string logEntry = $"[Época {result.EpochNumber}] Status: {result.Status} | " +
                          $"Antes: {result.MemoryBefore}MB | Após: {result.MemoryAfter}MB | " +
                          $"Delta: {result.Delta:+#;-#;0}MB | {result.Message}";
        File.AppendAllText(_logPath, logEntry + Environment.NewLine);
    }

    private long GetCurrentMemoryMB()
    {
        _process.Refresh();
        return _process.WorkingSet64 / (1024 * 1024);
    }

    /// <summary>
    /// NOVO: Valida se a liberação de memória atingiu o objetivo de 95%+
    /// </summary>
    public ValidationResult ValidateMemoryReleaseStrict(int epochNumber)
    {
        long currentMemoryMB = GetCurrentMemoryMB();
        long memoryDelta = currentMemoryMB - _previousEpochMemoryMB;
        double leakPercentage = ((double)memoryDelta / _previousEpochMemoryMB) * 100;
        double releasePercentage = 100.0 - leakPercentage;

        var result = new ValidationResult
        {
            EpochNumber = epochNumber,
            MemoryBefore = _previousEpochMemoryMB,
            MemoryAfter = currentMemoryMB,
            Delta = memoryDelta,
            LeakPercentage = leakPercentage,
            BaselineMemory = _baselineMemoryMB,
            ReleasePercentage = releasePercentage
        };

        // ========================================
        // CRITÉRIOS ESTRITOS DE VALIDAÇÃO
        // ========================================

        if (releasePercentage >= 95.0)
        {
            result.Status = ValidationStatus.Excellent;
            result.Message = $"EXCELENTE: {releasePercentage:F1}% da memória foi liberada!";
            _consecutiveLeaks = 0;
        }
        else if (releasePercentage >= 85.0)
        {
            result.Status = ValidationStatus.Success;
            result.Message = $"BOM: {releasePercentage:F1}% liberados (objetivo: 95%+)";
            _consecutiveLeaks = 0;
        }
        else if (releasePercentage >= 70.0)
        {
            result.Status = ValidationStatus.Warning;
            result.Message = $"ALERTA: Apenas {releasePercentage:F1}% liberados. Possível vazamento.";
            _consecutiveLeaks++;
        }
        else
        {
            result.Status = ValidationStatus.Critical;
            result.Message = $"CRÍTICO: Apenas {releasePercentage:F1}% liberados! Vazamento detectado!";
            _consecutiveLeaks++;
        }

        // Log detalhado
        PrintValidationReportStrict(result);
        LogToFile(result);

        // Alerta de vazamento contínuo
        if (_consecutiveLeaks >= 2)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"\n⚠️ ALERTA CRÍTICO: {_consecutiveLeaks} épocas consecutivas com vazamento!");
            Console.WriteLine("   CAUSAS POSSÍVEIS:");
            Console.WriteLine("   1. Tensores não estão sendo descartados (Dispose não chamado)");
            Console.WriteLine("   2. Cache Manager não está liberando arquivos");
            Console.WriteLine("   3. TensorPool retém tensores órfãos");
            Console.WriteLine("   4. Referências circulares impedem o GC");
            Console.WriteLine("   5. Driver GPU não está liberando buffers");
            Console.ResetColor();
        }

        return result;
    }

    /// <summary>
    /// NOVO: Relatório visual aprimorado
    /// </summary>
    private void PrintValidationReportStrict(ValidationResult result)
    {
        Console.WriteLine("\n╔═══════════════════════════════════════════════════════════╗");
        Console.WriteLine($"║  VALIDAÇÃO DE MEMÓRIA - ÉPOCA {result.EpochNumber,3}                        ║");
        Console.WriteLine("╠═══════════════════════════════════════════════════════════╣");

        // Status com cor
        Console.Write("║  Status: ");
        switch (result.Status)
        {
            case ValidationStatus.Excellent:
                Console.ForegroundColor = ConsoleColor.Green;
                Console.Write("✓ EXCELENTE");
                break;
            case ValidationStatus.Success:
                Console.ForegroundColor = ConsoleColor.Green;
                Console.Write("✓ BOM");
                break;
            case ValidationStatus.Warning:
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.Write("⚠ ALERTA");
                break;
            case ValidationStatus.Critical:
                Console.ForegroundColor = ConsoleColor.Red;
                Console.Write("✗ CRÍTICO");
                break;
        }

        Console.ResetColor();
        Console.WriteLine(new string(' ', 47 - result.Status.ToString().Length) + "║");

        Console.WriteLine("╠═══════════════════════════════════════════════════════════╣");
        Console.WriteLine($"║  Memória ANTES:        {result.MemoryBefore,6}MB                          ║");
        Console.WriteLine($"║  Memória DEPOIS:       {result.MemoryAfter,6}MB                          ║");
        Console.WriteLine($"║  Delta:                {result.Delta,6:+#;-#;0}MB                          ║");
        Console.WriteLine("╠═══════════════════════════════════════════════════════════╣");

        // Barra de progresso de liberação
        double releasePercent = result.ReleasePercentage;
        int barLength = 40;
        int filledLength = (int)(barLength * (releasePercent / 100.0));
        string bar = new string('█', filledLength) + new string('░', barLength - filledLength);

        Console.Write("║  Liberação: [");
        if (releasePercent >= 95.0)
            Console.ForegroundColor = ConsoleColor.Green;
        else if (releasePercent >= 85.0)
            Console.ForegroundColor = ConsoleColor.Yellow;
        else
            Console.ForegroundColor = ConsoleColor.Red;

        Console.Write(bar);
        Console.ResetColor();
        Console.WriteLine($"] {releasePercent,5:F1}%  ║");

        Console.WriteLine("╠═══════════════════════════════════════════════════════════╣");
        Console.WriteLine($"║  {result.Message.PadRight(57)}║");
        Console.WriteLine("╚═══════════════════════════════════════════════════════════╝\n");
    }

    /// <summary>
    /// NOVO: Análise de componentes (onde está a memória)
    /// </summary>
    public void AnalyzeMemoryComponents(GenerativeNeuralNetworkLSTM model)
    {
        Console.WriteLine("\n┌───────────────────────────────────────────────────────┐");
        Console.WriteLine("│  ANÁLISE DE COMPONENTES DE MEMÓRIA                   │");
        Console.WriteLine("├───────────────────────────────────────────────────────┤");

        try
        {
            // Pesos do modelo
            long weightsMemory = 0;
            if (model.weightsEmbedding != null)
                weightsMemory += model.weightsEmbedding.Length * sizeof(float);
            if (model.weightsInputForget != null)
                weightsMemory += model.weightsInputForget.Length * sizeof(float);
            // ... (adicionar outros pesos)

            long weightsMB = weightsMemory / (1024 * 1024);
            Console.WriteLine($"│  Pesos do Modelo:          {weightsMB,6}MB                  │");

            // TensorPool
            if (model._tensorPool != null)
            {
                // Assumindo que TensorPool tem um método GetMemoryUsage()
                Console.WriteLine($"│  TensorPool:               [uso não medido]        │");
            }

            // Cache Manager
            Console.WriteLine($"│  Cache Manager:            [arquivos em disco]     │");

            // Adam Optimizer
            Console.WriteLine($"│  Adam Optimizer:           [estados na GPU]        │");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"│  ERRO na análise: {ex.Message.PadRight(30)}│");
        }

        Console.WriteLine("└───────────────────────────────────────────────────────┘\n");
    }

    /// <summary>
    /// NOVO: Gera relatório detalhado de todas as épocas com recomendações
    /// </summary>
    public void GenerateFinalReportEnhanced()
    {
        Console.WriteLine("\n╔═══════════════════════════════════════════════════════════╗");
        Console.WriteLine("║       RELATÓRIO FINAL DE VALIDAÇÃO DE MEMÓRIA            ║");
        Console.WriteLine("╚═══════════════════════════════════════════════════════════╝");

        long finalMemory = GetCurrentMemoryMB();
        long totalLeak = finalMemory - _baselineMemoryMB;
        double leakPercentage = ((double)totalLeak / _baselineMemoryMB) * 100;

        Console.WriteLine($"  Memória Inicial (Época 1):  {_baselineMemoryMB,6}MB");
        Console.WriteLine($"  Memória Final:              {finalMemory,6}MB");
        Console.WriteLine($"  Vazamento Total:            {totalLeak,6:+#;-#;0}MB ({leakPercentage:+0.0;-0.0;0.0}%)");
        Console.WriteLine();

        // Classificação
        if (totalLeak <= 50)
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("  ✓ APROVADO: Liberação de memória EXCELENTE!");
            Console.WriteLine("    Vazamento acumulado está dentro do limite tolerável.");
        }
        else if (totalLeak <= 150)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("  ⚠ ACEITÁVEL: Pequeno vazamento acumulado detectado.");
            Console.WriteLine("    Recomenda-se investigar os componentes que retêm memória.");
        }
        else
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("  ✗ REPROVADO: Vazamento significativo de memória!");
            Console.WriteLine();
            Console.WriteLine("  AÇÕES RECOMENDADAS:");
            Console.WriteLine("  1. Verificar se Dispose() está sendo chamado em todos os tensores");
            Console.WriteLine("  2. Confirmar que Cache Manager deleta arquivos temporários");
            Console.WriteLine("  3. Validar que TensorPool libera tensores não usados");
            Console.WriteLine("  4. Verificar logs do driver GPU para vazamentos no device");
            Console.WriteLine("  5. Usar profiler de memória (dotMemory, PerfView) para análise");
        }

        Console.ResetColor();

        Console.WriteLine("\n" + new string('═', 63) + "\n");
    }


    /// <summary>
    /// Gera relatório final de todas as épocas.
    /// </summary>
    public void GenerateFinalReport()
    {
        Console.WriteLine("\n╔════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║         RELATÓRIO FINAL DE VALIDAÇÃO DE MEMÓRIA           ║");
        Console.WriteLine("╚════════════════════════════════════════════════════════════╝");

        long finalMemory = GetCurrentMemoryMB();
        long totalLeak = finalMemory - _baselineMemoryMB;

        Console.WriteLine($"  Memória Inicial (Época 1):  {_baselineMemoryMB}MB");
        Console.WriteLine($"  Memória Final:              {finalMemory}MB");
        Console.WriteLine($"  Vazamento Total:            {totalLeak:+#;-#;0}MB");

        if (totalLeak <= 50)
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("\n  ✓ APROVADO: Liberação de memória dentro do esperado!");
        }
        else if (totalLeak <= 200)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("\n  ⚠ ACEITÁVEL: Pequeno vazamento acumulado detectado.");
        }
        else
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("\n  ✗ REPROVADO: Vazamento significativo de memória detectado!");
            Console.WriteLine("     Recomenda-se revisar o código de Dispose e GC.");
        }

        Console.ResetColor();
        Console.WriteLine("\n" + new string('═', 62) + "\n");
    }
}

public enum ValidationStatus
{
    Excellent, // Memória diminuiu
    Success, // Memória estável (±10MB)
    Warning, // Leak de 50-100MB
    Critical // Leak > 100MB
}

public class ValidationResult
{
    public int EpochNumber { get; set; }
    public long MemoryBefore { get; set; }
    public long MemoryAfter { get; set; }
    public long Delta { get; set; }
    public double LeakPercentage { get; set; }
    
    // NOVO: Percentual de liberação (inverso do leak)
    public double ReleasePercentage { get; set; }
    
    public long BaselineMemory { get; set; }
    public ValidationStatus Status { get; set; }
    public string Message { get; set; } = "";
    
    // NOVO: Timestamp da validação
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    // NOVO: Detalhes de componentes (se disponível)
    public Dictionary<string, long>? ComponentMemory { get; set; }
}
