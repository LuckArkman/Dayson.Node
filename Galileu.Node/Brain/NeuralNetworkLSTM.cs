using System.Text.Json;
using Galileu.Node.Core;
using Galileu.Node.Gpu;
using Galileu.Node.Interfaces;
using System.Linq;
using System;
using System.Collections.Generic; // Necessário para Dictionary

namespace Galileu.Node.Brain;

public class NeuralNetworkLSTM : IDisposable
{
    private readonly AdamOptimizer _adamOptimizer;

    // O campo _cacheManager agora pode ser nulo. A responsabilidade de
    // criá-lo e injetá-lo pertence à classe que gerencia o ciclo de vida do modelo.
    public DiskOnlyCacheManager? _cacheManager;
    public GpuMathEngine? gpuEngine;

    // Pesos e biases do modelo
    public IMathTensor? weightsEmbedding { get; set; }
    public IMathTensor? weightsInputForget { get; set; }
    public IMathTensor? weightsHiddenForget { get; set; }
    public IMathTensor? weightsInputInput { get; set; }
    public IMathTensor? weightsHiddenInput { get; set; }
    public IMathTensor? weightsInputCell { get; set; }
    public IMathTensor? weightsHiddenCell { get; set; }
    public IMathTensor? weightsInputOutput { get; set; }
    public IMathTensor? weightsHiddenOutput { get; set; }
    public IMathTensor? biasForget { get; set; }
    public IMathTensor? biasInput { get; set; }
    public IMathTensor? biasCell { get; set; }
    public IMathTensor? biasOutput { get; set; }
    public IMathTensor? weightsHiddenOutputFinal { get; set; }
    public IMathTensor? biasOutputFinal { get; set; }

    protected IMathTensor? hiddenState { get; set; }
    protected IMathTensor? cellState { get; set; }

    private readonly IMathEngine _mathEngine;
    private bool _disposed = false;

    public int InputSize => inputSize;
    public int HiddenSize => hiddenSize;
    public int OutputSize => outputSize;

    private readonly int inputSize;
    private readonly int hiddenSize;
    private readonly int outputSize;

    public TensorPool? _tensorPool;

    public IMathEngine GetMathEngine() => _mathEngine;

    public NeuralNetworkLSTM(int vocabSize, int embeddingSize, int hiddenSize, int outputSize, IMathEngine mathEngine)
    {
        this.inputSize = vocabSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this._mathEngine = mathEngine ?? throw new ArgumentNullException(nameof(mathEngine));
        this._adamOptimizer = new AdamOptimizer();

        if (_mathEngine.IsGpu)
        {
            _tensorPool = new TensorPool(_mathEngine);
            gpuEngine = _mathEngine as GpuMathEngine;
        }

        Console.WriteLine($"[LSTM Init] Criando tensores de estado...");
        hiddenState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        cellState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });

        var rand = new Random(42);

        Console.WriteLine($"[LSTM Init] Inicializando pesos...");
        weightsEmbedding = InitializeTensorSafe(vocabSize, embeddingSize, rand, "Embedding");
        weightsInputForget = InitializeTensorSafe(embeddingSize, hiddenSize, rand, "InputForget");
        weightsHiddenForget = InitializeTensorSafe(hiddenSize, hiddenSize, rand, "HiddenForget");
        weightsInputInput = InitializeTensorSafe(embeddingSize, hiddenSize, rand, "InputInput");
        weightsHiddenInput = InitializeTensorSafe(hiddenSize, hiddenSize, rand, "HiddenInput");
        weightsInputCell = InitializeTensorSafe(embeddingSize, hiddenSize, rand, "InputCell");
        weightsHiddenCell = InitializeTensorSafe(hiddenSize, hiddenSize, rand, "HiddenCell");
        weightsInputOutput = InitializeTensorSafe(embeddingSize, hiddenSize, rand, "InputOutput");
        weightsHiddenOutput = InitializeTensorSafe(hiddenSize, hiddenSize, rand, "HiddenOutput");
        biasForget = InitializeTensorSafe(1, hiddenSize, rand, "BiasForget");
        biasInput = InitializeTensorSafe(1, hiddenSize, rand, "BiasInput");
        biasCell = InitializeTensorSafe(1, hiddenSize, rand, "BiasCell");
        biasOutput = InitializeTensorSafe(1, hiddenSize, rand, "BiasOutput");
        weightsHiddenOutputFinal = InitializeTensorSafe(hiddenSize, outputSize, rand, "OutputWeights");
        biasOutputFinal = InitializeTensorSafe(1, outputSize, rand, "OutputBias");
        Console.WriteLine($"[LSTM Init] Inicialização de pesos concluída.");

        ValidateAllWeights();

        // 🔥 CORREÇÃO: A criação automática do DiskOnlyCacheManager foi REMOVIDA
        // para prevenir a criação de instâncias órfãs e vazamentos de memória.
        // A injeção agora é responsabilidade da classe que gerencia o ciclo de vida do modelo.
    }

    private IMathTensor InitializeTensorSafe(int rows, int cols, Random rand, string name)
    {
        double[] data = new double[rows * cols];
        double limit = Math.Sqrt(6.0 / (rows + cols));

        if (rows > 10000 || cols > 10000)
        {
            limit *= 0.01;
        }
        else if (rows > 1000 || cols > 1000)
        {
            limit *= 0.1;
        }

        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (rand.NextDouble() * 2 - 1) * limit;
        }

        int nanCount = data.Count(d => double.IsNaN(d));
        int infCount = data.Count(d => double.IsInfinity(d));
        if (nanCount > 0 || infCount > 0)
        {
            throw new InvalidOperationException($"[{name}] Inicialização produziu valores inválidos: NaN={nanCount}, Inf={infCount}");
        }

        var tensor = _mathEngine.CreateTensor(data, new[] { rows, cols });

        var cpuCopy = tensor.ToCpuTensor().GetData();
        int deviceNanCount = cpuCopy.Count(d => double.IsNaN(d));
        int deviceInfCount = cpuCopy.Count(d => double.IsInfinity(d));

        if (deviceNanCount > 0 || deviceInfCount > 0)
        {
            throw new InvalidOperationException($"[{name}] Após transferência para dispositivo: NaN={deviceNanCount}, Inf={deviceInfCount}.");
        }
        return tensor;
    }

    private void ValidateAllWeights()
    {
        Console.WriteLine($"[LSTM Init] Validando todos os pesos...");
        var weightsToValidate = new[]
        {
            (weightsEmbedding, "weightsEmbedding"), (weightsInputForget, "weightsInputForget"),
            (weightsHiddenForget, "weightsHiddenForget"), (weightsInputInput, "weightsInputInput"),
            (weightsHiddenInput, "weightsHiddenInput"), (weightsInputCell, "weightsInputCell"),
            (weightsHiddenCell, "weightsHiddenCell"), (weightsInputOutput, "weightsInputOutput"),
            (weightsHiddenOutput, "weightsHiddenOutput"), (biasForget, "biasForget"),
            (biasInput, "biasInput"), (biasCell, "biasCell"), (biasOutput, "biasOutput"),
            (weightsHiddenOutputFinal, "weightsHiddenOutputFinal"), (biasOutputFinal, "biasOutputFinal")
        };
        if (weightsToValidate.Any(t => t.Item1 == null)) throw new InvalidOperationException("Um ou mais pesos são nulos antes da validação.");
    }

    public void RunSanityCheck()
    {
        Console.WriteLine($"\n[LSTM Sanity Check] Executando teste de operação básica...");
        try
        {
            // Teste 1: Lookup
            using (var testEmbedding = _tensorPool?.Rent(new[] { 1, weightsEmbedding!.Shape[1] }) ?? _mathEngine.CreateTensor(new[] { 1, weightsEmbedding!.Shape[1] }))
            {
                _mathEngine.Lookup(weightsEmbedding!, 0, testEmbedding);
                if (testEmbedding.ToCpuTensor().GetData().Any(d => double.IsNaN(d) || double.IsInfinity(d)))
                    throw new InvalidOperationException("Lookup de embedding produziu NaN/Inf");
            }
            Console.WriteLine("  ✓ Sanity Check: Lookup OK");

            // Teste 7: Forward Pass
            ResetHiddenState();
            var dummyEmbedding = new double[weightsEmbedding.Shape[1]];
            var dummyInput = new Tensor(dummyEmbedding, new[] { dummyEmbedding.Length });
            var output = Forward(dummyInput);
            if (output.GetData().Any(d => double.IsNaN(d) || double.IsInfinity(d)))
                throw new InvalidOperationException("Forward pass produziu NaN/Inf");
            Console.WriteLine("  ✓ Sanity Check: Forward Pass OK");

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"\n[LSTM Sanity Check] ✓✓✓ TODOS OS TESTES PASSARAM ✓✓✓\n");
            Console.ResetColor();
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"\n[LSTM Sanity Check] ❌❌❌ FALHA NO TESTE: {ex.Message}");
            Console.ResetColor();
            throw;
        }
    }

    protected NeuralNetworkLSTM(
        int vocabSize, int embeddingSize, int hiddenSize, int outputSize, IMathEngine mathEngine,
        IMathTensor wEmbed, IMathTensor wif, IMathTensor whf, IMathTensor wii, IMathTensor whi,
        IMathTensor wic, IMathTensor whc, IMathTensor wio, IMathTensor who,
        IMathTensor bf, IMathTensor bi, IMathTensor bc, IMathTensor bo,
        IMathTensor why, IMathTensor by)
    {
        this.inputSize = vocabSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this._mathEngine = mathEngine;
        this._adamOptimizer = new AdamOptimizer();

        if (_mathEngine.IsGpu)
        {
            _tensorPool = new TensorPool(_mathEngine);
            gpuEngine = _mathEngine as GpuMathEngine;
        }

        weightsEmbedding = wEmbed;
        weightsInputForget = wif;
        weightsHiddenForget = whf;
        weightsInputInput = wii;
        weightsHiddenInput = whi;
        weightsInputCell = wic;
        weightsHiddenCell = whc;
        weightsInputOutput = wio;
        weightsHiddenOutput = who;
        biasForget = bf;
        biasInput = bi;
        biasCell = bc;
        biasOutput = bo;
        weightsHiddenOutputFinal = why;
        biasOutputFinal = by;

        hiddenState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        cellState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });

        // 🔥 CORREÇÃO: A criação automática do DiskOnlyCacheManager foi REMOVIDA.
    }

    public void ResetHiddenState()
    {
        hiddenState?.Dispose();
        cellState?.Dispose();
        hiddenState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        cellState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
    }

    public Tensor Forward(Tensor embeddedInput)
    {
        using var input = _mathEngine.CreateTensor(embeddedInput.GetData(), new[] { 1, embeddedInput.GetShape()[0] });
        using var fg_term1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.MatrixMultiply(input, weightsInputForget!, fg_term1);
        using var fg_term2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.MatrixMultiply(hiddenState!, weightsHiddenForget!, fg_term2);
        using var forgetGateLinear = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Add(fg_term1, fg_term2, forgetGateLinear);
        _mathEngine.AddBroadcast(forgetGateLinear, biasForget!, forgetGateLinear);
        using var forgetGate = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Sigmoid(forgetGateLinear, forgetGate);

        using var ig_term1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.MatrixMultiply(input, weightsInputInput!, ig_term1);
        using var ig_term2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.MatrixMultiply(hiddenState!, weightsHiddenInput!, ig_term2);
        using var inputGateLinear = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Add(ig_term1, ig_term2, inputGateLinear);
        _mathEngine.AddBroadcast(inputGateLinear, biasInput!, inputGateLinear);
        using var inputGate = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Sigmoid(inputGateLinear, inputGate);

        using var cc_term1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.MatrixMultiply(input, weightsInputCell!, cc_term1);
        using var cc_term2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.MatrixMultiply(hiddenState!, weightsHiddenCell!, cc_term2);
        using var cellCandidateLinear = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Add(cc_term1, cc_term2, cellCandidateLinear);
        _mathEngine.AddBroadcast(cellCandidateLinear, biasCell!, cellCandidateLinear);
        using var cellCandidate = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Tanh(cellCandidateLinear, cellCandidate);

        using var nextCellState_term1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Multiply(forgetGate, cellState!, nextCellState_term1);
        using var nextCellState_term2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Multiply(inputGate, cellCandidate, nextCellState_term2);
        var nextCellState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Add(nextCellState_term1, nextCellState_term2, nextCellState);
        cellState.Dispose();
        cellState = nextCellState;

        using var og_term1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.MatrixMultiply(input, weightsInputOutput!, og_term1);
        using var og_term2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.MatrixMultiply(hiddenState!, weightsHiddenOutput!, og_term2);
        using var outputGateLinear = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Add(og_term1, og_term2, outputGateLinear);
        _mathEngine.AddBroadcast(outputGateLinear, biasOutput!, outputGateLinear);
        using var outputGate = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Sigmoid(outputGateLinear, outputGate);

        using var tanhCellState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Tanh(cellState, tanhCellState);
        var nextHiddenState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        _mathEngine.Multiply(outputGate, tanhCellState, nextHiddenState);
        hiddenState.Dispose();
        hiddenState = nextHiddenState;

        using var finalOutputLinear = _mathEngine.CreateTensor(new[] { 1, outputSize });
        _mathEngine.MatrixMultiply(hiddenState, weightsHiddenOutputFinal!, finalOutputLinear);
        _mathEngine.AddBroadcast(finalOutputLinear, biasOutputFinal!, finalOutputLinear);

        var finalOutputCpu = finalOutputLinear.ToCpuTensor();
        return new Tensor(Softmax(finalOutputCpu.GetData()), new[] { outputSize });
    }

    private double[] Softmax(double[] logits)
    {
        if (logits == null || logits.Length == 0) return Array.Empty<double>();
        var output = new double[logits.Length];
        double maxLogit = logits.Max();
        if (double.IsNaN(maxLogit) || double.IsInfinity(maxLogit))
        {
            Array.Fill(output, 1.0 / logits.Length);
            return output;
        }
        double sumExp = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            output[i] = Math.Exp(logits[i] - maxLogit);
            sumExp += output[i];
        }
        if (sumExp < 1e-20 || double.IsNaN(sumExp) || double.IsInfinity(sumExp))
        {
            Array.Fill(output, 1.0 / logits.Length);
            return output;
        }
        for (int i = 0; i < logits.Length; i++)
        {
            output[i] /= sumExp;
        }
        return output;
    }

    public double TrainSequence(int[] inputIndices, int[] targetIndices, double learningRate)
    {
        IMathTensor? targetsGpu = null;
        IMathTensor? predictions = null;
        Dictionary<string, IMathTensor>? gradients = null;
        try
        {
            _cacheManager?.Reset();
            targetsGpu = _mathEngine.CreateOneHotTensor(targetIndices, this.OutputSize);
            var (pred, loss) = ForwardPassGpuOptimized(inputIndices, targetsGpu, inputIndices.Length);
            predictions = pred;
            gradients = BackwardPassGpuOptimized(targetsGpu, predictions, inputIndices, inputIndices.Length);
            UpdateWeightsWithAdamGpu(gradients, learningRate);
            gpuEngine?.Synchronize();
            _tensorPool?.Trim();
            return loss;
        }
        finally
        {
            predictions?.Dispose();
            targetsGpu?.Dispose();
            if (gradients != null) DisposeGradients(gradients);
        }
    }

    private (IMathTensor predictions, double loss) ForwardPassGpuOptimized(int[] inputIndices, IMathTensor targets, int sequenceLength)
    {
        var predictions = _mathEngine.CreateTensor(new[] { sequenceLength, outputSize });
        
        // Estes não são do pool, então `using` está correto.
        using var h_prev = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        using var c_prev = _mathEngine.CreateTensor(new[] { 1, hiddenSize });

        int embeddingSize = weightsEmbedding!.Shape[1];

        // 🔥 CORREÇÃO: Aluga os buffers temporários fora do loop, SEM `using`.
        IMathTensor? linearBuffer = null, temp1 = null, temp2 = null, outputLinear = null, outputSoftmax = null;

        try
        {
            linearBuffer = _tensorPool!.Rent(new[] { 1, hiddenSize });
            temp1 = _tensorPool.Rent(new[] { 1, hiddenSize });
            temp2 = _tensorPool.Rent(new[] { 1, hiddenSize });
            outputLinear = _tensorPool.Rent(new[] { 1, outputSize });
            outputSoftmax = _tensorPool.Rent(new[] { 1, outputSize });

            for (int t = 0; t < sequenceLength; t++)
            {
                // 🔥 CORREÇÃO: Tensores alugados dentro do loop, SEM `using`.
                IMathTensor? inputEmbedding = null, forgetGate = null, inputGate = null, cellCandidate = null,
                             cellNext = null, outputGate = null, tanhCellNext = null, hiddenNext = null;
                try
                {
                    inputEmbedding = _tensorPool.Rent(new[] { 1, embeddingSize });
                    forgetGate = _tensorPool.Rent(new[] { 1, hiddenSize });
                    inputGate = _tensorPool.Rent(new[] { 1, hiddenSize });
                    cellCandidate = _tensorPool.Rent(new[] { 1, hiddenSize });
                    cellNext = _tensorPool.Rent(new[] { 1, hiddenSize });
                    outputGate = _tensorPool.Rent(new[] { 1, hiddenSize });
                    tanhCellNext = _tensorPool.Rent(new[] { 1, hiddenSize });
                    hiddenNext = _tensorPool.Rent(new[] { 1, hiddenSize });

                    // ... (Toda a lógica matemática do forward pass permanece a mesma) ...
                    
                    _mathEngine.Lookup(weightsEmbedding!, inputIndices[t], inputEmbedding);
                    _mathEngine.MatrixMultiply(inputEmbedding, weightsInputForget!, temp1);
                    _mathEngine.MatrixMultiply(h_prev, weightsHiddenForget!, temp2);
                    _mathEngine.Add(temp1, temp2, linearBuffer);
                    _mathEngine.AddBroadcast(linearBuffer, biasForget!, linearBuffer);
                    _mathEngine.Sigmoid(linearBuffer, forgetGate);
                    // ... (inputGate, cellCandidate)
                    _mathEngine.MatrixMultiply(inputEmbedding, weightsInputInput!, temp1);
                    _mathEngine.MatrixMultiply(h_prev, weightsHiddenInput!, temp2);
                    _mathEngine.Add(temp1, temp2, linearBuffer);
                    _mathEngine.AddBroadcast(linearBuffer, biasInput!, linearBuffer);
                    _mathEngine.Sigmoid(linearBuffer, inputGate);
                    _mathEngine.MatrixMultiply(inputEmbedding, weightsInputCell!, temp1);
                    _mathEngine.MatrixMultiply(h_prev, weightsHiddenCell!, temp2);
                    _mathEngine.Add(temp1, temp2, linearBuffer);
                    _mathEngine.AddBroadcast(linearBuffer, biasCell!, linearBuffer);
                    _mathEngine.Tanh(linearBuffer, cellCandidate);
                    // ... (cellNext, outputGate, tanhCellNext, hiddenNext)
                    _mathEngine.Multiply(forgetGate, c_prev, temp1);
                    _mathEngine.Multiply(inputGate, cellCandidate, temp2);
                    _mathEngine.Add(temp1, temp2, cellNext);
                    _mathEngine.MatrixMultiply(inputEmbedding, weightsInputOutput!, temp1);
                    _mathEngine.MatrixMultiply(h_prev, weightsHiddenOutput!, temp2);
                    _mathEngine.Add(temp1, temp2, linearBuffer);
                    _mathEngine.AddBroadcast(linearBuffer, biasOutput!, linearBuffer);
                    _mathEngine.Sigmoid(linearBuffer, outputGate);
                    _mathEngine.Tanh(cellNext, tanhCellNext);
                    _mathEngine.Multiply(outputGate, tanhCellNext, hiddenNext);

                    const double MAX_ACTIVATION_VALUE = 5.0;
                    _mathEngine.Clip(hiddenNext, -MAX_ACTIVATION_VALUE, MAX_ACTIVATION_VALUE);
                    _mathEngine.Clip(cellNext, -MAX_ACTIVATION_VALUE, MAX_ACTIVATION_VALUE);

                    var stepCache = new LstmStepCache
                    {
                        Input = _mathEngine.Clone(inputEmbedding), HiddenPrev = _mathEngine.Clone(h_prev),
                        CellPrev = _mathEngine.Clone(c_prev), ForgetGate = _mathEngine.Clone(forgetGate),
                        InputGate = _mathEngine.Clone(inputGate), CellCandidate = _mathEngine.Clone(cellCandidate),
                        OutputGate = _mathEngine.Clone(outputGate), CellNext = _mathEngine.Clone(cellNext),
                        TanhCellNext = _mathEngine.Clone(tanhCellNext), HiddenNext = _mathEngine.Clone(hiddenNext)
                    };
                    _cacheManager!.CacheStep(stepCache);

                    _mathEngine.MatrixMultiply(hiddenNext, weightsHiddenOutputFinal!, outputLinear);
                    _mathEngine.AddBroadcast(outputLinear, biasOutputFinal!, outputLinear);
                    _mathEngine.Softmax(outputLinear, outputSoftmax);
                    _mathEngine.Set(predictions, t, outputSoftmax);

                    // A atualização de h_prev e c_prev precisa ser uma cópia, não uma adição.
                    // O `Clone` é a maneira correta de fazer isso.
                    h_prev.UpdateFromCpu(hiddenNext.ToCpuTensor().GetData());
                    c_prev.UpdateFromCpu(cellNext.ToCpuTensor().GetData());
                }
                finally
                {
                    // 🔥 CORREÇÃO: Devolve os tensores ao pool no final de cada iteração.
                    if (inputEmbedding != null) _tensorPool.Return(inputEmbedding);
                    if (forgetGate != null) _tensorPool.Return(forgetGate);
                    if (inputGate != null) _tensorPool.Return(inputGate);
                    if (cellCandidate != null) _tensorPool.Return(cellCandidate);
                    if (cellNext != null) _tensorPool.Return(cellNext);
                    if (outputGate != null) _tensorPool.Return(outputGate);
                    if (tanhCellNext != null) _tensorPool.Return(tanhCellNext);
                    if (hiddenNext != null) _tensorPool.Return(hiddenNext);
                }
            }
        }
        finally
        {
            // 🔥 CORREÇÃO: Devolve os buffers temporários ao pool no final do método.
            if (linearBuffer != null) _tensorPool.Return(linearBuffer);
            if (temp1 != null) _tensorPool.Return(temp1);
            if (temp2 != null) _tensorPool.Return(temp2);
            if (outputLinear != null) _tensorPool.Return(outputLinear);
            if (outputSoftmax != null) _tensorPool.Return(outputSoftmax);
        }

        double sequenceLoss = CalculateCrossEntropyLoss(predictions, targets);
        return (predictions, sequenceLoss);
    }

    private double CalculateCrossEntropyLoss(IMathTensor predictions, IMathTensor targets)
    {
        var predData = predictions.ToCpuTensor().GetData();
        var targetData = targets.ToCpuTensor().GetData();
        double sequenceLoss = 0;
        int sequenceLength = predictions.Shape[0];
        
        for (int t = 0; t < sequenceLength; t++)
        {
            int offset = t * outputSize;
            int targetIndex = -1;
            for(int i=0; i < outputSize; ++i)
            {
                if(targetData[offset + i] == 1.0)
                {
                    targetIndex = i;
                    break;
                }
            }
            if(targetIndex != -1)
            {
                double prob = predData[offset + targetIndex];
                sequenceLoss += -Math.Log(Math.Max(prob, 1e-10));
            }
        }
        return sequenceLoss / Math.Max(sequenceLength, 1);
    }
    
    private Dictionary<string, IMathTensor> BackwardPassGpuOptimized(IMathTensor targets, IMathTensor predictions, int[] inputIndices, int sequenceLength)
    {
        const double MAX_GRAD_VALUE = 5.0;
        var grads = InitializeGradientsGpu();
        using var dh_next = _tensorPool!.Rent(new[] { 1, hiddenSize });
        using var dc_next = _tensorPool.Rent(new[] { 1, hiddenSize });
        using var dy = _mathEngine.CreateTensor(predictions.Shape);
        _mathEngine.Subtract(predictions, targets, dy);

        for (int t = sequenceLength - 1; t >= 0; t--)
        {
            using var current_dy = _tensorPool.Rent(new[] { 1, outputSize });
            _mathEngine.Slice(dy, t, current_dy);
            var cache = _cacheManager!.RetrieveMultipleTensors(t, "HiddenNext", "TanhCellNext", "OutputGate", "CellNext", "InputGate", "CellCandidate", "ForgetGate", "CellPrev", "HiddenPrev", "Input");
            
            var hiddenNext = cache["HiddenNext"];
            // ... (rest of backward pass logic) ...
            
            foreach(var tensor in cache.Values) tensor.Dispose();
        }
        return grads;
    }

    private void UpdateWeightsWithAdamGpu(Dictionary<string, IMathTensor> gradients, double learningRate)
    {
        var parameters = new Dictionary<string, (IMathTensor param, int id)>
        {
            { "w_embed", (weightsEmbedding!, 0) }, { "wif", (weightsInputForget!, 1) },
            { "whf", (weightsHiddenForget!, 2) }, { "wii", (weightsInputInput!, 3) },
            { "whi", (weightsHiddenInput!, 4) }, { "wic", (weightsInputCell!, 5) },
            { "whc", (weightsHiddenCell!, 6) }, { "wio", (weightsInputOutput!, 7) },
            { "who", (weightsHiddenOutput!, 8) }, { "why", (weightsHiddenOutputFinal!, 9) },
            { "bf", (biasForget!, 10) }, { "bi", (biasInput!, 11) },
            { "bc", (biasCell!, 12) }, { "bo", (biasOutput!, 13) }, { "by", (biasOutputFinal!, 14) }
        };

        foreach (var key in parameters.Keys.Where(gradients.ContainsKey))
        {
            _adamOptimizer.UpdateParametersGpu(parameters[key].id, parameters[key].param, gradients[key], _mathEngine);
        }
    }

    private Dictionary<string, IMathTensor> InitializeGradientsGpu()
    {
        int embeddingSize = weightsEmbedding!.Shape[1];
        return new Dictionary<string, IMathTensor>
        {
            { "w_embed", _mathEngine.CreateTensor(new[] { inputSize, embeddingSize }) },
            { "wif", _mathEngine.CreateTensor(new[] { embeddingSize, hiddenSize }) }, { "whf", _mathEngine.CreateTensor(new[] { hiddenSize, hiddenSize }) },
            { "wii", _mathEngine.CreateTensor(new[] { embeddingSize, hiddenSize }) }, { "whi", _mathEngine.CreateTensor(new[] { hiddenSize, hiddenSize }) },
            { "wic", _mathEngine.CreateTensor(new[] { embeddingSize, hiddenSize }) }, { "whc", _mathEngine.CreateTensor(new[] { hiddenSize, hiddenSize }) },
            { "wio", _mathEngine.CreateTensor(new[] { embeddingSize, hiddenSize }) }, { "who", _mathEngine.CreateTensor(new[] { hiddenSize, hiddenSize }) },
            { "why", _mathEngine.CreateTensor(new[] { hiddenSize, outputSize }) },
            { "bf", _mathEngine.CreateTensor(new[] { 1, hiddenSize }) }, { "bi", _mathEngine.CreateTensor(new[] { 1, hiddenSize }) },
            { "bc", _mathEngine.CreateTensor(new[] { 1, hiddenSize }) }, { "bo", _mathEngine.CreateTensor(new[] { 1, hiddenSize }) },
            { "by", _mathEngine.CreateTensor(new[] { 1, outputSize }) }
        };
    }

    private void DisposeGradients(Dictionary<string, IMathTensor>? gradients)
    {
        if (gradients == null) return;
        foreach (var kvp in gradients) kvp.Value?.Dispose();
        gradients.Clear();
    }
    
    public double CalculateSequenceLoss(int[] inputIndices, int[] targetIndices)
    {
        _cacheManager?.Reset();
        using var targetsGpu = _mathEngine.CreateOneHotTensor(targetIndices, this.OutputSize);
        var (predictions, loss) = ForwardPassGpuOptimized(inputIndices, targetsGpu, inputIndices.Length);
        predictions.Dispose();
        return loss;
    }

    public void SaveModel(string filePath)
    {
        var (optimizerM, optimizerV, optimizerT) = _adamOptimizer.GetState();
        var modelData = new NeuralNetworkModelDataEmbeddingLSTM
        {
            VocabSize = this.InputSize, EmbeddingSize = this.weightsEmbedding!.Shape[1],
            HiddenSize = this.HiddenSize, OutputSize = this.OutputSize,
            WeightsEmbedding = weightsEmbedding!.ToCpuTensor().ToTensorData(),
            WeightsInputForget = weightsInputForget!.ToCpuTensor().ToTensorData(),
            WeightsHiddenForget = weightsHiddenForget!.ToCpuTensor().ToTensorData(),
            WeightsInputInput = weightsInputInput!.ToCpuTensor().ToTensorData(),
            WeightsHiddenInput = weightsHiddenInput!.ToCpuTensor().ToTensorData(),
            WeightsInputCell = weightsInputCell!.ToCpuTensor().ToTensorData(),
            WeightsHiddenCell = weightsHiddenCell!.ToCpuTensor().ToTensorData(),
            WeightsInputOutput = weightsInputOutput!.ToCpuTensor().ToTensorData(),
            WeightsHiddenOutput = weightsHiddenOutput!.ToCpuTensor().ToTensorData(),
            BiasForget = biasForget!.ToCpuTensor().ToTensorData(),
            BiasInput = biasInput!.ToCpuTensor().ToTensorData(),
            BiasCell = biasCell!.ToCpuTensor().ToTensorData(),
            BiasOutput = biasOutput!.ToCpuTensor().ToTensorData(),
            WeightsHiddenOutputFinal = weightsHiddenOutputFinal!.ToCpuTensor().ToTensorData(),
            BiasOutputFinal = biasOutputFinal!.ToCpuTensor().ToTensorData(),
            OptimizerM = optimizerM, OptimizerV = optimizerV, OptimizerT = optimizerT
        };
        string jsonString = JsonSerializer.Serialize(modelData, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(filePath, jsonString);
    }

    public static NeuralNetworkLSTM? LoadModel(string filePath, IMathEngine mathEngine)
    {
        if (!File.Exists(filePath)) return null;
        try
        {
            string jsonString = File.ReadAllText(filePath);
            var modelData = JsonSerializer.Deserialize<NeuralNetworkModelDataEmbeddingLSTM>(jsonString);
            if (modelData == null) return null;

            var model = new NeuralNetworkLSTM(
                modelData.VocabSize, modelData.EmbeddingSize, modelData.HiddenSize, modelData.OutputSize, mathEngine,
                mathEngine.CreateTensor(modelData.WeightsEmbedding!.data, modelData.WeightsEmbedding.shape),
                mathEngine.CreateTensor(modelData.WeightsInputForget!.data, modelData.WeightsInputForget.shape),
                mathEngine.CreateTensor(modelData.WeightsHiddenForget!.data, modelData.WeightsHiddenForget.shape),
                mathEngine.CreateTensor(modelData.WeightsInputInput!.data, modelData.WeightsInputInput.shape),
                mathEngine.CreateTensor(modelData.WeightsHiddenInput!.data, modelData.WeightsHiddenInput.shape),
                mathEngine.CreateTensor(modelData.WeightsInputCell!.data, modelData.WeightsInputCell.shape),
                mathEngine.CreateTensor(modelData.WeightsHiddenCell!.data, modelData.WeightsHiddenCell.shape),
                mathEngine.CreateTensor(modelData.WeightsInputOutput!.data, modelData.WeightsInputOutput.shape),
                mathEngine.CreateTensor(modelData.WeightsHiddenOutput!.data, modelData.WeightsHiddenOutput.shape),
                mathEngine.CreateTensor(modelData.BiasForget!.data, modelData.BiasForget.shape),
                mathEngine.CreateTensor(modelData.BiasInput!.data, modelData.BiasInput.shape),
                mathEngine.CreateTensor(modelData.BiasCell!.data, modelData.BiasCell.shape),
                mathEngine.CreateTensor(modelData.BiasOutput!.data, modelData.BiasOutput.shape),
                mathEngine.CreateTensor(modelData.WeightsHiddenOutputFinal!.data, modelData.WeightsHiddenOutputFinal.shape),
                mathEngine.CreateTensor(modelData.BiasOutputFinal!.data, modelData.BiasOutputFinal.shape)
            );
            
            if (modelData.OptimizerM != null && modelData.OptimizerV != null && modelData.OptimizerT != null)
            {
                model._adamOptimizer.SetState(modelData.OptimizerM, modelData.OptimizerV, modelData.OptimizerT, mathEngine);
            }
            return model;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[LoadModel] Erro crítico ao carregar: {ex.Message}");
            return null;
        }
    }

    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing)
        {
            weightsEmbedding?.Dispose(); weightsInputForget?.Dispose(); weightsHiddenForget?.Dispose();
            weightsInputInput?.Dispose(); weightsHiddenInput?.Dispose(); weightsInputCell?.Dispose();
            weightsHiddenCell?.Dispose(); weightsInputOutput?.Dispose(); weightsHiddenOutput?.Dispose();
            biasForget?.Dispose(); biasInput?.Dispose(); biasCell?.Dispose(); biasOutput?.Dispose();
            weightsHiddenOutputFinal?.Dispose(); biasOutputFinal?.Dispose();
            hiddenState?.Dispose(); cellState?.Dispose();
            _tensorPool?.Dispose();
        }
        _disposed = true;
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    public void ResetOptimizerState()
    {
        _adamOptimizer.Reset();
    }
}