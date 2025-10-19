using System;
using System.Collections.Generic;
using System.Linq;
using Galileu.Node.Brain;
using Galileu.Node.Core;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Brain;

/// <summary>
/// Implementa o otimizador Adam para atualização de pesos da rede neural.
/// Esta versão é otimizada para GPU, gerenciando os estados de momento (m, v)
/// como IMathTensors para evitar transferências de dados entre CPU e GPU.
/// Também suporta a serialização de seu estado para treinamentos de longa duração.
/// </summary>
public class AdamOptimizer
{
    private readonly double _learningRate;
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _epsilon;

    // Dicionários guardam IMathTensor. Estes tensores residirão na GPU (se aplicável),
    // eliminando a necessidade de cópias para a RAM.
    private readonly Dictionary<int, IMathTensor> _m;
    private readonly Dictionary<int, IMathTensor> _v;
    private readonly Dictionary<int, int> _t;

    public AdamOptimizer(double learningRate = 0.001, double beta1 = 0.9,
        double beta2 = 0.999, double epsilon = 1e-8)
    {
        _learningRate = learningRate;
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        _m = new Dictionary<int, IMathTensor>();
        _v = new Dictionary<int, IMathTensor>();
        _t = new Dictionary<int, int>();
    }

    /// <summary>
    /// Atualiza os parâmetros de uma camada diretamente no dispositivo de computação (CPU ou GPU).
    /// Delega a lógica matemática para a IMathEngine, que executará um kernel otimizado.
    /// </summary>
    /// <param name="layerId">Um identificador único para a camada de pesos.</param>
    /// <param name="parameters">O tensor de pesos/parâmetros a ser atualizado.</param>
    /// <param name="gradients">O tensor de gradientes correspondente aos parâmetros.</param>
    /// <param name="mathEngine">A engine de matemática a ser usada para as operações.</param>
    public void UpdateParametersGpu(int layerId, IMathTensor parameters, IMathTensor gradients, IMathEngine mathEngine)
    {
        // Inicializa os tensores de momento (m e v) no dispositivo se for a primeira vez.
        if (!_m.ContainsKey(layerId))
        {
            _m[layerId] = mathEngine.CreateTensor(parameters.Shape);
            _v[layerId] = mathEngine.CreateTensor(parameters.Shape);
            _t[layerId] = 0;
        }

        // Incrementa o timestep para esta camada específica.
        _t[layerId]++;
        int t = _t[layerId];

        var m = _m[layerId];
        var v = _v[layerId];

        // Delega a lógica complexa de atualização para um único método na IMathEngine.
        // Isso executa um kernel que realiza todas as operações em paralelo na GPU,
        // sem nenhuma cópia de dados para a memória principal.
        mathEngine.AdamUpdate(parameters, gradients, m, v, _learningRate, _beta1, _beta2, _epsilon, t);
    }

    /// <summary>
    /// Extrai o estado atual do otimizador (momentos e timesteps) para serialização.
    /// Os tensores são convertidos para o formato de CPU (TensorData) para serem salvos em JSON.
    /// </summary>
    /// <returns>Uma tupla contendo os dicionários de estado para m, v, e t.</returns>
    public (Dictionary<int, TensorData> m, Dictionary<int, TensorData> v, Dictionary<int, int> t) GetState()
    {
        var mState = _m.ToDictionary(kvp => kvp.Key, kvp => kvp.Value.ToCpuTensor().ToTensorData());
        var vState = _v.ToDictionary(kvp => kvp.Key, kvp => kvp.Value.ToCpuTensor().ToTensorData());
        var tState = new Dictionary<int, int>(_t); // Cria uma cópia para evitar modificações externas

        return (mState, vState, tState);
    }

    /// <summary>
    /// Restaura o estado do otimizador a partir de dados desserializados.
    /// Limpa qualquer estado anterior e recria os tensores de momento no dispositivo de computação.
    /// </summary>
    /// <param name="mState">O dicionário de estado para o momento m.</param>
    /// <param name="vState">O dicionário de estado para o momento v.</param>
    /// <param name="tState">O dicionário de estado para os timesteps t.</param>
    /// <param name="mathEngine">A engine de matemática usada para criar os novos tensores.</param>
    public void SetState(Dictionary<int, TensorData> mState, Dictionary<int, TensorData> vState,
        Dictionary<int, int> tState, IMathEngine mathEngine)
    {
        Reset(); // Limpa completamente qualquer estado interno existente.

        foreach (var kvp in mState)
        {
            _m[kvp.Key] = mathEngine.CreateTensor(kvp.Value.data, kvp.Value.shape);
        }

        foreach (var kvp in vState)
        {
            _v[kvp.Key] = mathEngine.CreateTensor(kvp.Value.data, kvp.Value.shape);
        }

        foreach (var kvp in tState)
        {
            _t[kvp.Key] = kvp.Value;
        }
    }

    /// <summary>
    /// CRÍTICO: Limpa os estados internos do otimizador para liberar memória.
    /// Libera a memória de cada tensor de momento que foi alocado (especialmente na GPU).
    /// </summary>
    public void Reset()
    {
        Console.WriteLine("[AdamOptimizer] Iniciando reset de estados...");

        int tensorCount = 0;
        long totalMemoryFreed = 0;

        // ========================================
        // FASE 1: LIBERAR MOMENTOS M
        // ========================================
        foreach (var tensor in _m.Values)
        {
            if (tensor != null)
            {
                long tensorSize = tensor.Length * sizeof(float);
                totalMemoryFreed += tensorSize;
                tensor.Dispose();
                tensorCount++;
            }
        }

        // ========================================
        // FASE 2: LIBERAR MOMENTOS V
        // ========================================
        foreach (var tensor in _v.Values)
        {
            if (tensor != null)
            {
                long tensorSize = tensor.Length * sizeof(float);
                totalMemoryFreed += tensorSize;
                tensor.Dispose();
                tensorCount++;
            }
        }

        // ========================================
        // FASE 3: LIMPAR DICIONÁRIOS
        // ========================================
        _m.Clear();
        _v.Clear();
        _t.Clear();

        // Relatório
        double memoryFreedMB = totalMemoryFreed / (1024.0 * 1024.0);
        Console.WriteLine($"[AdamOptimizer] ✓ Reset concluído:");
        Console.WriteLine($"  └─ {tensorCount} tensores descartados");
        Console.WriteLine($"  └─ ~{memoryFreedMB:F2}MB liberados");
    }

    /// <summary>
    /// NOVO: Valida se o otimizador foi resetado corretamente.
    /// Útil para debug de vazamentos de memória.
    /// </summary>
    public bool IsReset()
    {
        return _m.Count == 0 && _v.Count == 0 && _t.Count == 0;
    }

    /// <summary>
    /// NOVO: Retorna estatísticas de uso de memória.
    /// </summary>
    public (int layerCount, long memoryMB) GetMemoryStats()
    {
        long totalBytes = 0;

        foreach (var tensor in _m.Values)
        {
            totalBytes += tensor.Length * sizeof(float);
        }

        foreach (var tensor in _v.Values)
        {
            totalBytes += tensor.Length * sizeof(float);
        }

        long memoryMB = totalBytes / (1024 * 1024);
        return (_m.Count, memoryMB);
    }
}