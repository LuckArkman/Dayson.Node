using Galileu.Node.Interfaces;
using System.Collections.Generic;

namespace Galileu.Node.Brain;

/// <summary>
/// Pool de tensores reutilizáveis com gerenciamento agressivo de memória.
/// OTIMIZADO PARA TREINAMENTOS LONGOS (100+ épocas).
/// </summary>
public class TensorPool : IDisposable
{
    private readonly IMathEngine _mathEngine;
    private readonly Dictionary<string, Queue<IMathTensor>> _pools;
    private readonly HashSet<IMathTensor> _inUse;
    private bool _disposed = false;

    // === NOVOS PARÂMETROS DE CONTROLE DE MEMÓRIA ===
    private const int MAX_POOL_SIZE_PER_SHAPE = 64;  // Máximo de tensores por shape
    private const long MAX_TOTAL_MEMORY_MB = 2048;    // 2GB limite total do pool
    private int _operationsSinceLastTrim = 0;
    private const int TRIM_INTERVAL = 500;            // Auto-trim a cada 500 operações

    public TensorPool(IMathEngine mathEngine)
    {
        _mathEngine = mathEngine;
        _pools = new Dictionary<string, Queue<IMathTensor>>();
        _inUse = new HashSet<IMathTensor>();
    }

    /// <summary>
    /// Obtém ou cria um tensor com as dimensões especificadas.
    /// NOVO: Implementa limite por shape e auto-trim periódico.
    /// </summary>
    public IMathTensor Rent(int[] shape)
    {
        string key = GetKey(shape);
        
        if (!_pools.ContainsKey(key))
            _pools[key] = new Queue<IMathTensor>();
        
        var pool = _pools[key];
        
        IMathTensor tensor;
        if (pool.Count > 0)
        {
            tensor = pool.Dequeue();
        }
        else
        {
            // NOVO: Verifica limite de memória antes de alocar
            if (GetTotalMemoryUsageMB() > MAX_TOTAL_MEMORY_MB)
            {
                TrimExcessMemory();
            }
            tensor = _mathEngine.CreateTensor(shape);
        }
        
        _inUse.Add(tensor);
        
        // NOVO: Auto-trim periódico
        _operationsSinceLastTrim++;
        if (_operationsSinceLastTrim >= TRIM_INTERVAL)
        {
            TrimExcessMemory();
            _operationsSinceLastTrim = 0;
        }
        
        return tensor;
    }

    /// <summary>
    /// Devolve o tensor ao pool para reutilização.
    /// NOVO: Implementa limite de tamanho do pool.
    /// </summary>
    public void Return(IMathTensor tensor)
    {
        if (tensor == null || !_inUse.Contains(tensor))
            return;
        
        _inUse.Remove(tensor);
        string key = GetKey(tensor.Shape);
        
        if (!_pools.ContainsKey(key))
            _pools[key] = new Queue<IMathTensor>();
        
        var pool = _pools[key];
        
        // NOVO: Limite por shape - libera imediatamente se exceder
        if (pool.Count >= MAX_POOL_SIZE_PER_SHAPE)
        {
            tensor.Dispose();
            return;
        }
        
        pool.Enqueue(tensor);
    }

    /// <summary>
    /// NOVO: Remove tensores em excesso quando limite de memória é atingido.
    /// </summary>
    private void TrimExcessMemory()
    {
        long currentMemoryMB = GetTotalMemoryUsageMB();
        long targetMemoryMB = MAX_TOTAL_MEMORY_MB / 2; // Reduz para 50% do limite
        
        if (currentMemoryMB <= targetMemoryMB)
            return;
        
        //Console.WriteLine($"[TensorPool] Trim iniciado: {currentMemoryMB}MB -> alvo {targetMemoryMB}MB");
        
        // Libera pools com mais tensores primeiro
        var sortedPools = _pools
            .OrderByDescending(kvp => kvp.Value.Count * GetShapeMemoryMB(kvp.Key))
            .ToList();
        
        foreach (var (shape, pool) in sortedPools)
        {
            int released = 0;
            // Mantém apenas 25% dos tensores em cada pool
            int keepCount = Math.Max(4, pool.Count / 4);
            
            while (pool.Count > keepCount)
            {
                var tensor = pool.Dequeue();
                tensor.Dispose();
                released++;
            }
            
            if (released > 0)
            {
                //Console.WriteLine($"[TensorPool] Shape {shape}: liberados {released} tensors");
            }
            
            // Verifica se já atingiu o alvo
            currentMemoryMB = GetTotalMemoryUsageMB();
            if (currentMemoryMB <= targetMemoryMB)
                break;
        }
        
        // Força coleta de lixo após trim agressivo
        GC.Collect(2, GCCollectionMode.Forced, true, true);
        GC.WaitForPendingFinalizers();
        
        //Console.WriteLine($"[TensorPool] Trim concluído: {GetTotalMemoryUsageMB()}MB");
    }

    /// <summary>
    /// Libera todos os tensores que não estão em uso.
    /// Chame isso entre épocas para limpar memória.
    /// </summary>
    public void Trim()
    {
        foreach (var pool in _pools.Values)
        {
            while (pool.Count > 0)
            {
                var tensor = pool.Dequeue();
                tensor.Dispose();
            }
        }
        
        // NOVO: Força GC após trim manual
        GC.Collect();
        GC.WaitForPendingFinalizers();
    }

    /// <summary>
    /// NOVO: Calcula uso total de memória do pool (aproximado).
    /// </summary>
    private long GetTotalMemoryUsageMB()
    {
        long totalBytes = 0;
        
        foreach (var (shapeKey, pool) in _pools)
        {
            long shapeBytes = GetShapeMemoryBytes(shapeKey);
            totalBytes += shapeBytes * pool.Count;
        }
        
        // Adiciona tensores em uso
        foreach (var tensor in _inUse)
        {
            totalBytes += tensor.Length * sizeof(float);
        }
        
        return totalBytes / (1024 * 1024);
    }

    /// <summary>
    /// NOVO: Calcula memória de um shape específico.
    /// </summary>
    private long GetShapeMemoryMB(string shapeKey)
    {
        return GetShapeMemoryBytes(shapeKey) / (1024 * 1024);
    }

    private long GetShapeMemoryBytes(string shapeKey)
    {
        var dimensions = shapeKey.Split('x').Select(int.Parse).ToArray();
        long elements = dimensions.Aggregate(1L, (a, b) => a * b);
        return elements * sizeof(float);
    }

    /// <summary>
    /// NOVO: Imprime estatísticas detalhadas de uso de memória.
    /// </summary>
    public void PrintStats()
    {
        //Console.WriteLine("\n[TensorPool] === ESTATÍSTICAS DE MEMÓRIA ===");
        //Console.WriteLine($"Tensores em uso: {_inUse.Count}");
        //Console.WriteLine($"Pools ativos: {_pools.Count}");
        
        long totalPooledMB = 0;
        foreach (var (shape, pool) in _pools.OrderByDescending(kvp => kvp.Value.Count))
        {
            long shapeMB = GetShapeMemoryMB(shape);
            long poolMB = shapeMB * pool.Count;
            totalPooledMB += poolMB;
            
            if (pool.Count > 0)
            {
                //Console.WriteLine($"  Shape {shape}: {pool.Count} tensors × {shapeMB}MB = {poolMB}MB");
            }
        }
        
        long inUseMB = _inUse.Sum(t => t.Length * sizeof(float)) / (1024 * 1024);
        
        //Console.WriteLine($"\nTotal pooled: {totalPooledMB}MB");
        //Console.WriteLine($"Total em uso: {inUseMB}MB");
        //Console.WriteLine($"TOTAL POOL: {totalPooledMB + inUseMB}MB / {MAX_TOTAL_MEMORY_MB}MB");
        //Console.WriteLine("=========================================\n");
    }

    private string GetKey(int[] shape)
    {
        return string.Join("x", shape);
    }

    public void Dispose()
    {
        if (_disposed) return;
        
        // Libera tensores em uso
        foreach (var tensor in _inUse)
            tensor.Dispose();
        
        // Libera pools
        foreach (var pool in _pools.Values)
        {
            while (pool.Count > 0)
                pool.Dequeue().Dispose();
        }
        
        _inUse.Clear();
        _pools.Clear();
        _disposed = true;
    }
}