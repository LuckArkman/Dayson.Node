using System;
using System.Collections.Generic;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Brain;

/// <summary>
/// Implementa o otimizador Adam para atualização de pesos da rede neural.
/// Esta versão é otimizada para GPU, gerenciando os estados de momento (m, v)
/// como IMathTensors para evitar transferências de dados entre CPU e GPU.
/// </summary>
public class AdamOptimizer
{
    private readonly float _learningRate;
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _epsilon;

    // MODIFICADO: Dicionários agora guardam IMathTensor.
    // Estes tensores residirão na GPU, eliminando a necessidade de cópias para a RAM.
    private readonly Dictionary<int, IMathTensor> _m;
    private readonly Dictionary<int, IMathTensor> _v;
    private readonly Dictionary<int, int> _t;

    public AdamOptimizer(float learningRate = 0.001f, float beta1 = 0.9f,
        float beta2 = 0.999f, float epsilon = 1e-8f)
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
    /// Delega a lógica matemática para a IMathEngine, que executará um kernel otimizado na GPU.
    /// </summary>
    /// <param name="layerId">Um identificador único para a camada de pesos.</param>
    /// <param name="parameters">O tensor de pesos/parâmetros a ser atualizado.</param>
    /// <param name="gradients">O tensor de gradientes correspondente aos parâmetros.</param>
    /// <param name="mathEngine">A engine de matemática a ser usada para as operações.</param>
    public void UpdateParametersGpu(int layerId, IMathTensor parameters, IMathTensor gradients, IMathEngine mathEngine)
    {
        // Inicializa os tensores de momento (m e v) no dispositivo (GPU) se for a primeira vez.
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
        // Isso executa um kernel OpenCL que realiza todas as operações em paralelo na GPU,
        // sem nenhuma cópia de dados para a memória principal.
        mathEngine.AdamUpdate(parameters, gradients, m, v, _learningRate, _beta1, _beta2, _epsilon, t);
    }

    /// <summary>
    /// CRÍTICO: Limpa os estados internos do otimizador para liberar memória.
    /// Este método deve ser chamado entre as épocas de treinamento para prevenir
    /// o acúmulo de memória pelos tensores de momento.
    /// </summary>
    public void Reset()
    {
        // Libera a memória de cada tensor de momento que foi alocado na GPU.
        foreach (var tensor in _m.Values)
        {
            tensor.Dispose();
        }

        foreach (var tensor in _v.Values)
        {
            tensor.Dispose();
        }

        // Limpa os dicionários, liberando as referências aos tensores.
        _m.Clear();
        _v.Clear();
        _t.Clear();

        // Sugere ao Garbage Collector que este é um bom momento para uma coleta.
        GC.Collect(0, GCCollectionMode.Optimized);
    }
}