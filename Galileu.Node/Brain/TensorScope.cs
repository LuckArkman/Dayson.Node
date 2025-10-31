using System;
using System.Collections.Generic;
using System.Diagnostics;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Brain;

/// <summary>
/// Gerencia o ciclo de vida de tensores temporários dentro de um escopo.
/// Garante liberação automática ao sair do escopo (using pattern).
/// Previne vazamentos de memória VRAM causados por tensores intermediários.
/// </summary>
public class TensorScope : IDisposable
{
    private readonly List<IMathTensor> _managedTensors;
    private readonly TensorPool? _pool;
    private readonly string _scopeName;
    private readonly Stopwatch _stopwatch;
    private bool _disposed = false;
    
    // Estatísticas
    private int _tensorCount = 0;
    private long _totalBytesManaged = 0;
    
    public TensorScope(string scopeName = "Anonymous", TensorPool? pool = null)
    {
        _scopeName = scopeName;
        _pool = pool;
        _managedTensors = new List<IMathTensor>();
        _stopwatch = Stopwatch.StartNew();
    }
    
    /// <summary>
    /// Registra um tensor para gerenciamento automático.
    /// O tensor será liberado automaticamente ao fim do escopo.
    /// </summary>
    public T Track<T>(T tensor) where T : IMathTensor
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(TensorScope));
        
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));
        
        _managedTensors.Add(tensor);
        _tensorCount++;
        _totalBytesManaged += tensor.Length * sizeof(float);
        
        return tensor;
    }
    
    /// <summary>
    /// Registra múltiplos tensores de uma só vez.
    /// </summary>
    public void TrackAll(params IMathTensor[] tensors)
    {
        foreach (var tensor in tensors)
        {
            if (tensor != null)
                Track(tensor);
        }
    }
    
    /// <summary>
    /// Remove um tensor do gerenciamento (caso precise mantê-lo vivo).
    /// </summary>
    public T Untrack<T>(T tensor) where T : IMathTensor
    {
        _managedTensors.Remove(tensor);
        return tensor;
    }
    
    /// <summary>
    /// Cria um sub-escopo aninhado.
    /// </summary>
    public TensorScope CreateSubScope(string name)
    {
        return new TensorScope($"{_scopeName}.{name}", _pool);
    }
    
    public void Dispose()
    {
        if (_disposed) return;
        
        _stopwatch.Stop();
        
        int freedCount = 0;
        long freedBytes = 0;
        
        // Libera tensores em ordem reversa (LIFO)
        for (int i = _managedTensors.Count - 1; i >= 0; i--)
        {
            var tensor = _managedTensors[i];
            
            try
            {
                long tensorBytes = tensor.Length * sizeof(float);
                
                // Se tem pool, retorna ao pool; senão, dispose direto
                if (_pool != null && tensor.IsGpu)
                {
                    _pool.Return(tensor);
                }
                else
                {
                    tensor.Dispose();
                }
                
                freedCount++;
                freedBytes += tensorBytes;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[TensorScope:{_scopeName}] Erro ao liberar tensor: {ex.Message}");
            }
        }
        
        _managedTensors.Clear();
        
        // Log de performance (apenas se houver tensores)
        if (_tensorCount > 0 && _stopwatch.ElapsedMilliseconds > 100)
        {
            /*
            Console.WriteLine($"[TensorScope:{_scopeName}] " +
                              $"Liberou {freedCount}/{_tensorCount} tensors " +
                              $"({freedBytes / (1024.0 * 1024.0):F2} MB) " +
                              $"em {_stopwatch.ElapsedMilliseconds}ms");
            */
        }
        
        _disposed = true;
    }
}

/// <summary>
/// Extensões para simplificar uso de TensorScope.
/// </summary>
public static class TensorScopeExtensions
{
    /// <summary>
    /// Executa ação dentro de um escopo gerenciado.
    /// Exemplo: TensorScope.Run("MyOperation", scope => { ... });
    /// </summary>
    public static void Run(string name, Action<TensorScope> action, TensorPool? pool = null)
    {
        using var scope = new TensorScope(name, pool);
        action(scope);
    }
    
    /// <summary>
    /// Executa função dentro de um escopo gerenciado e retorna resultado.
    /// Exemplo: var result = TensorScope.Run("MyOperation", scope => { return tensor; });
    /// </summary>
    public static T Run<T>(string name, Func<TensorScope, T> func, TensorPool? pool = null)
    {
        using var scope = new TensorScope(name, pool);
        return func(scope);
    }
}