using System.Text.Json;
using Galileu.Node.Core;
using Galileu.Node.Gpu;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Brain;

public class NeuralNetworkLSTM : IDisposable
{
    private readonly AdamOptimizer _adamOptimizer;

    // --- CORREÇÃO 1: Usando o gerenciador de cache correto e mais eficiente ---
    public DiskOnlyCacheManager _cacheManager;
    public GpuMathEngine gpuEngine;

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

    // SUBSTITUA o construtor principal do NeuralNetworkLSTM por este:

    public NeuralNetworkLSTM(int vocabSize, int embeddingSize, int hiddenSize, int outputSize, IMathEngine mathEngine)
    {
        this.inputSize = vocabSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        if (_mathEngine == null)
            this._mathEngine = mathEngine ?? throw new ArgumentNullException(nameof(mathEngine));

        if (_adamOptimizer == null)
            _adamOptimizer = new AdamOptimizer();

        if (_mathEngine.IsGpu)
        {
            _tensorPool = new TensorPool(_mathEngine);
        }

        Console.WriteLine($"[LSTM Init] Criando tensores de estado...");
        hiddenState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        cellState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });

        var rand = new Random(42); // 🔥 SEED FIXA PARA REPRODUTIBILIDADE

        Console.WriteLine($"[LSTM Init] Inicializando pesos...");
        Console.WriteLine($"  - Embedding: [{vocabSize} x {embeddingSize}]");

        // 🔥 INICIALIZAÇÃO MAIS CONSERVADORA
        weightsEmbedding = InitializeTensorSafe(vocabSize, embeddingSize, rand, "Embedding");

        Console.WriteLine($"  - LSTM Gates: [{embeddingSize} x {hiddenSize}] e [{hiddenSize} x {hiddenSize}]");
        weightsInputForget = InitializeTensorSafe(embeddingSize, hiddenSize, rand, "InputForget");
        weightsHiddenForget = InitializeTensorSafe(hiddenSize, hiddenSize, rand, "HiddenForget");
        weightsInputInput = InitializeTensorSafe(embeddingSize, hiddenSize, rand, "InputInput");
        weightsHiddenInput = InitializeTensorSafe(hiddenSize, hiddenSize, rand, "HiddenInput");
        weightsInputCell = InitializeTensorSafe(embeddingSize, hiddenSize, rand, "InputCell");
        weightsHiddenCell = InitializeTensorSafe(hiddenSize, hiddenSize, rand, "HiddenCell");
        weightsInputOutput = InitializeTensorSafe(embeddingSize, hiddenSize, rand, "InputOutput");
        weightsHiddenOutput = InitializeTensorSafe(hiddenSize, hiddenSize, rand, "HiddenOutput");

        Console.WriteLine($"  - Biases: [1 x {hiddenSize}]");
        biasForget = InitializeTensorSafe(1, hiddenSize, rand, "BiasForget");
        biasInput = InitializeTensorSafe(1, hiddenSize, rand, "BiasInput");
        biasCell = InitializeTensorSafe(1, hiddenSize, rand, "BiasCell");
        biasOutput = InitializeTensorSafe(1, hiddenSize, rand, "BiasOutput");

        Console.WriteLine($"  - Output Layer: [{hiddenSize} x {outputSize}]");
        weightsHiddenOutputFinal = InitializeTensorSafe(hiddenSize, outputSize, rand, "OutputWeights");
        biasOutputFinal = InitializeTensorSafe(1, outputSize, rand, "OutputBias");

        Console.WriteLine($"[LSTM Init] Inicialização de pesos concluída.");

        // 🔥 VALIDAÇÃO CRÍTICA PÓS-INICIALIZAÇÃO
        ValidateAllWeights();

        if (_cacheManager == null)
            _cacheManager = new DiskOnlyCacheManager(_mathEngine, embeddingSize, hiddenSize);
    }

// 🔥 NOVO: Método de inicialização com validação
    private IMathTensor InitializeTensorSafe(int rows, int cols, Random rand, string name)
    {
        double[] data = new double[rows * cols];

        // Xavier/Glorot initialization
        double limit = Math.Sqrt(6.0 / (rows + cols));

        // 🔥 REDUÇÃO ADICIONAL PARA CAMADAS GRANDES
        if (rows > 10000 || cols > 10000)
        {
            limit *= 0.01; // Muito mais conservador (1% do normal)
            Console.WriteLine($"    [{name}] Limite ultra-conservador: {limit:E4}");
        }
        else if (rows > 1000 || cols > 1000)
        {
            limit *= 0.1; // Conservador (10% do normal)
            Console.WriteLine($"    [{name}] Limite conservador: {limit:E4}");
        }

        // Gera valores aleatórios
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (rand.NextDouble() * 2 - 1) * limit;
        }

        // 🔥 VALIDAÇÃO IMEDIATA
        int nanCount = 0, infCount = 0, zeroCount = 0;
        double sum = 0, sumSquares = 0;

        for (int i = 0; i < data.Length; i++)
        {
            if (double.IsNaN(data[i])) nanCount++;
            else if (double.IsInfinity(data[i])) infCount++;
            else if (data[i] == 0) zeroCount++;

            sum += data[i];
            sumSquares += data[i] * data[i];
        }

        if (nanCount > 0 || infCount > 0)
        {
            throw new InvalidOperationException(
                $"[{name}] Inicialização produziu valores inválidos: NaN={nanCount}, Inf={infCount}"
            );
        }

        double mean = sum / data.Length;
        double variance = (sumSquares / data.Length) - (mean * mean);
        double stdDev = Math.Sqrt(variance);

        Console.WriteLine($"    [{name}] Stats: mean={mean:E4}, std={stdDev:E4}, zeros={zeroCount}");

        // Cria tensor
        var tensor = _mathEngine.CreateTensor(data, new[] { rows, cols });

        // 🔥 VALIDAÇÃO NO DISPOSITIVO (GPU/CPU)
        var cpuCopy = tensor.ToCpuTensor().GetData();
        int deviceNanCount = cpuCopy.Count(d => double.IsNaN(d));
        int deviceInfCount = cpuCopy.Count(d => double.IsInfinity(d));

        if (deviceNanCount > 0 || deviceInfCount > 0)
        {
            throw new InvalidOperationException(
                $"[{name}] Após transferência para dispositivo: NaN={deviceNanCount}, Inf={deviceInfCount}. " +
                "Possível bug no driver GPU."
            );
        }

        return tensor;
    }

// 🔥 NOVO: Validação completa de todos os pesos
    private void ValidateAllWeights()
    {
        Console.WriteLine($"[LSTM Init] Validando todos os pesos...");

        var weightsToValidate = new[]
        {
            (weightsEmbedding, "weightsEmbedding"),
            (weightsInputForget, "weightsInputForget"),
            (weightsHiddenForget, "weightsHiddenForget"),
            (weightsInputInput, "weightsInputInput"),
            (weightsHiddenInput, "weightsHiddenInput"),
            (weightsInputCell, "weightsInputCell"),
            (weightsHiddenCell, "weightsHiddenCell"),
            (weightsInputOutput, "weightsInputOutput"),
            (weightsHiddenOutput, "weightsHiddenOutput"),
            (biasForget, "biasForget"),
            (biasInput, "biasInput"),
            (biasCell, "biasCell"),
            (biasOutput, "biasOutput"),
            (weightsHiddenOutputFinal, "weightsHiddenOutputFinal"),
            (biasOutputFinal, "biasOutputFinal")
        };

        bool allValid = true;

        foreach (var (tensor, name) in weightsToValidate)
        {
            var data = tensor!.ToCpuTensor().GetData();
            int nanCount = data.Count(d => double.IsNaN(d));
            int infCount = data.Count(d => double.IsInfinity(d));

            if (nanCount > 0 || infCount > 0)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"  ❌ [{name}] INVÁLIDO: NaN={nanCount}, Inf={infCount}");
                Console.ResetColor();
                allValid = false;
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"  ✓ [{name}] Válido ({data.Length} valores)");
                Console.ResetColor();
            }
        }

        if (!allValid)
        {
            throw new InvalidOperationException(
                "Um ou mais tensores de peso foram inicializados com valores inválidos. " +
                "Isso pode indicar um bug no driver GPU ou na engine de matemática."
            );
        }

        Console.WriteLine($"[LSTM Init] ✓ Todos os pesos validados com sucesso!");
    }

// 🔥 NOVO: Teste de operação básica
    /// <summary>
    /// Executa uma bateria de testes para validar a integridade do modelo e dos kernels GPU.
    /// Este método DEVE ser chamado imediatamente após a construção do modelo.
    /// </summary>
    public void RunSanityCheck()
    {
        Console.WriteLine($"\n[LSTM Sanity Check] Executando teste de operação básica...");

        try
        {
            // ================================================================
            // TESTE 1: LOOKUP DE EMBEDDING
            // ================================================================
            Console.WriteLine($"  [1/7] Testando lookup de embedding...");

            var testEmbedding = _tensorPool?.Rent(new[] { 1, weightsEmbedding!.Shape[1] });
            if (testEmbedding == null)
            {
                testEmbedding = _mathEngine.CreateTensor(new[] { 1, weightsEmbedding!.Shape[1] });
            }

            // Testa lookup do token <PAD> (índice 0)
            _mathEngine.Lookup(weightsEmbedding!, 0, testEmbedding);
            var embData = testEmbedding.ToCpuTensor().GetData();

            if (embData.Any(d => double.IsNaN(d) || double.IsInfinity(d)))
            {
                throw new InvalidOperationException("Lookup de embedding produziu NaN/Inf");
            }

            Console.WriteLine(
                $"  ✓ Lookup OK (sample: [{string.Join(", ", embData.Take(5).Select(x => $"{x:F4}"))}...])");

            if (_tensorPool != null)
                _tensorPool.Return(testEmbedding);
            else
                testEmbedding.Dispose();

            // ================================================================
            // TESTE 2: MULTIPLICAÇÃO DE MATRIZ
            // ================================================================
            Console.WriteLine($"  [2/7] Testando multiplicação de matriz...");

            int embeddingSize = weightsEmbedding.Shape[1];
            var testInput = _mathEngine.CreateTensor(new[] { 1, embeddingSize });
            var testOutput = _mathEngine.CreateTensor(new[] { 1, hiddenSize });

            // Inicializa testInput com valores pequenos
            var testInputData = new double[embeddingSize];
            for (int i = 0; i < embeddingSize; i++)
            {
                testInputData[i] = 0.01 * (i % 10);
            }

            testInput.UpdateFromCpu(testInputData);

            _mathEngine.MatrixMultiply(testInput, weightsInputForget!, testOutput);
            var matMulData = testOutput.ToCpuTensor().GetData();

            if (matMulData.Any(d => double.IsNaN(d) || double.IsInfinity(d)))
            {
                throw new InvalidOperationException("Matrix multiply produziu NaN/Inf");
            }

            Console.WriteLine(
                $"  ✓ MatMul OK (sample: [{string.Join(", ", matMulData.Take(5).Select(x => $"{x:F4}"))}...])");

            testInput.Dispose();
            testOutput.Dispose();

            // ================================================================
            // TESTE 3: SIGMOID
            // ================================================================
            Console.WriteLine($"  [3/7] Testando sigmoid...");

            var testSigmoidIn = _mathEngine.CreateTensor(new double[] { -10, -1, 0, 1, 10 }, new[] { 1, 5 });
            var testSigmoidOut = _mathEngine.CreateTensor(new[] { 1, 5 });

            _mathEngine.Sigmoid(testSigmoidIn, testSigmoidOut);
            var sigmoidData = testSigmoidOut.ToCpuTensor().GetData();

            if (sigmoidData.Any(d => double.IsNaN(d) || double.IsInfinity(d)))
            {
                throw new InvalidOperationException("Sigmoid produziu NaN/Inf");
            }

            // Valida que sigmoid está no range correto [0, 1]
            if (sigmoidData.Any(d => d < 0 || d > 1))
            {
                throw new InvalidOperationException(
                    $"Sigmoid produziu valores fora do range [0,1]: [{string.Join(", ", sigmoidData)}]");
            }

            Console.WriteLine($"  ✓ Sigmoid OK: [{string.Join(", ", sigmoidData.Select(x => $"{x:F4}"))}]");

            testSigmoidIn.Dispose();
            testSigmoidOut.Dispose();

            // ================================================================
            // TESTE 4: SOFTMAX
            // ================================================================
            Console.WriteLine($"  [4/7] Testando softmax...");

            var testSoftmaxIn = _mathEngine.CreateTensor(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 }, new[] { 1, 5 });
            var testSoftmaxOut = _mathEngine.CreateTensor(new[] { 1, 5 });

            _mathEngine.Softmax(testSoftmaxIn, testSoftmaxOut);
            var softmaxData = testSoftmaxOut.ToCpuTensor().GetData();

            if (softmaxData.Any(d => double.IsNaN(d) || double.IsInfinity(d)))
            {
                throw new InvalidOperationException("Softmax produziu NaN/Inf");
            }

            double sumProbs = softmaxData.Sum();
            if (Math.Abs(sumProbs - 1.0) > 0.01)
            {
                throw new InvalidOperationException($"Softmax não soma 1.0 (soma={sumProbs})");
            }

            Console.WriteLine(
                $"  ✓ Softmax OK: [{string.Join(", ", softmaxData.Select(x => $"{x:F4}"))}] (sum={sumProbs:F4})");

            testSoftmaxIn.Dispose();
            testSoftmaxOut.Dispose();

            // ================================================================
            // TESTE 5: CONVERSÃO FLOAT↔DOUBLE (CRÍTICO PARA GPU)
            // ================================================================
            Console.WriteLine($"  [5/7] Testando conversão float↔double...");

            var testDoubles = new double[]
            {
                1e-10, // Muito pequeno
                0.5, // Normal
                1.0, // Normal
                100.0, // Grande
                1e10, // Muito grande (pode perder precisão em float)
                -1e-10, // Negativo pequeno
                -0.5, // Negativo normal
                -100.0 // Negativo grande
            };

            var testTensor = _mathEngine.CreateTensor(testDoubles, new[] { 1, testDoubles.Length });
            var retrieved = testTensor.ToCpuTensor().GetData();

            bool conversionOk = true;
            for (int i = 0; i < testDoubles.Length; i++)
            {
                double original = testDoubles[i];
                double retrievedVal = retrieved[i];
                double error = Math.Abs(original - retrievedVal);
                double relativeError = error / Math.Max(Math.Abs(original), 1e-10);

                if (double.IsNaN(retrievedVal) || double.IsInfinity(retrievedVal))
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"  ❌ Conversão produziu NaN/Inf: {original} → {retrievedVal}");
                    Console.ResetColor();
                    conversionOk = false;
                }
                else if (relativeError > 0.01) // Erro relativo > 1% (tolerante para float32)
                {
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine(
                        $"  ⚠️ Perda de precisão: {original:E4} → {retrievedVal:E4} (erro: {relativeError:P2})");
                    Console.ResetColor();
                }
            }

            if (conversionOk)
            {
                Console.WriteLine($"  ✓ Conversão OK: [{string.Join(", ", retrieved.Select(x => $"{x:E2}"))}]");
            }
            else
            {
                throw new InvalidOperationException("Conversão float↔double produziu valores inválidos");
            }

            testTensor.Dispose();

            // ================================================================
            // TESTE 6: ADD E ADDBROADCAST
            // ================================================================
            Console.WriteLine($"  [6/7] Testando Add e AddBroadcast...");

            var testA = _mathEngine.CreateTensor(new double[] { 1, 2, 3, 4 }, new[] { 1, 4 });
            var testB = _mathEngine.CreateTensor(new double[] { 0.1, 0.2, 0.3, 0.4 }, new[] { 1, 4 });
            var testResult = _mathEngine.CreateTensor(new[] { 1, 4 });

            // Teste Add
            _mathEngine.Add(testA, testB, testResult);
            var addData = testResult.ToCpuTensor().GetData();

            if (addData.Any(d => double.IsNaN(d) || double.IsInfinity(d)))
            {
                throw new InvalidOperationException("Add produziu NaN/Inf");
            }

            // Teste AddBroadcast
            var testMatrix = _mathEngine.CreateTensor(new double[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
            var testBias = _mathEngine.CreateTensor(new double[] { 0.1, 0.2, 0.3 }, new[] { 1, 3 });
            var testBroadcastResult = _mathEngine.CreateTensor(new[] { 2, 3 });

            _mathEngine.AddBroadcast(testMatrix, testBias, testBroadcastResult);
            var broadcastData = testBroadcastResult.ToCpuTensor().GetData();

            if (broadcastData.Any(d => double.IsNaN(d) || double.IsInfinity(d)))
            {
                throw new InvalidOperationException("AddBroadcast produziu NaN/Inf");
            }

            Console.WriteLine($"  ✓ Add OK: [{string.Join(", ", addData.Select(x => $"{x:F2}"))}]");
            Console.WriteLine($"  ✓ AddBroadcast OK: [{string.Join(", ", broadcastData.Select(x => $"{x:F2}"))}]");

            testA.Dispose();
            testB.Dispose();
            testResult.Dispose();
            testMatrix.Dispose();
            testBias.Dispose();
            testBroadcastResult.Dispose();

            // ================================================================
            // TESTE 7: FORWARD PASS COMPLETO
            // ================================================================
            Console.WriteLine($"  [7/7] Testando forward pass completo...");

            ResetHiddenState();

            // Cria um vetor de embedding dummy (já embedado, não usa lookup)
            var dummyEmbedding = new double[weightsEmbedding.Shape[1]];
            for (int i = 0; i < dummyEmbedding.Length; i++)
            {
                dummyEmbedding[i] = 0.01 * (i % 10); // Valores pequenos e variados
            }

            var dummyInput = new Tensor(dummyEmbedding, new[] { dummyEmbedding.Length });
            var output = Forward(dummyInput);

            if (output.GetData().Any(d => double.IsNaN(d) || double.IsInfinity(d)))
            {
                throw new InvalidOperationException("Forward pass produziu NaN/Inf");
            }

            // Valida que output é uma distribuição de probabilidade válida
            double outputSum = output.GetData().Sum();
            if (Math.Abs(outputSum - 1.0) > 0.01)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"  ⚠️ Output não soma exatamente 1.0 (soma={outputSum:F6})");
                Console.ResetColor();
            }

            Console.WriteLine(
                $"  ✓ Forward OK (sample output: [{string.Join(", ", output.GetData().Take(5).Select(x => $"{x:F4}"))}...])");
            Console.WriteLine($"  ✓ Output sum: {outputSum:F6}");

            // ================================================================
            // TESTE 8: TESTE ESPECÍFICO DE LOOKUP COM MÚLTIPLOS ÍNDICES
            // ================================================================
            Console.WriteLine($"  [EXTRA] Testando lookup com múltiplos tokens...");

            var testTokens = new int[] { 0, 1, 100, 1000, 10000, 19999 }; // <PAD>, <UNK>, e outros

            foreach (var tokenIdx in testTokens)
            {
                if (tokenIdx >= weightsEmbedding.Shape[0]) continue;

                var embTest = _mathEngine.CreateTensor(new[] { 1, weightsEmbedding.Shape[1] });
                _mathEngine.Lookup(weightsEmbedding, tokenIdx, embTest);
                var embTestData = embTest.ToCpuTensor().GetData();

                int nanCount = embTestData.Count(d => double.IsNaN(d));
                int infCount = embTestData.Count(d => double.IsInfinity(d));

                if (nanCount > 0 || infCount > 0)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"  ❌ Lookup falhou para token {tokenIdx}: NaN={nanCount}, Inf={infCount}");
                    Console.ResetColor();
                    embTest.Dispose();
                    throw new InvalidOperationException($"Lookup produziu valores inválidos para token {tokenIdx}");
                }

                embTest.Dispose();
            }

            Console.WriteLine($"  ✓ Lookup testado para tokens: [{string.Join(", ", testTokens)}]");

            // ================================================================
            // SUCESSO!
            // ================================================================
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"\n[LSTM Sanity Check] ✓✓✓ TODOS OS TESTES PASSARAM ✓✓✓\n");
            Console.ResetColor();
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"\n[LSTM Sanity Check] ❌❌❌ FALHA NO TESTE ❌❌❌");
            Console.WriteLine($"Erro: {ex.Message}");

            if (ex.InnerException != null)
            {
                Console.WriteLine($"Inner Exception: {ex.InnerException.Message}");
            }

            Console.WriteLine($"\nStack Trace:");
            Console.WriteLine(ex.StackTrace);
            Console.ResetColor();

            throw; // Re-lança a exceção para abortar o treinamento
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
        if (_adamOptimizer == null) _adamOptimizer = new AdamOptimizer();

        if (_mathEngine.IsGpu)
        {
            _tensorPool = new TensorPool(_mathEngine);
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

        // --- CORREÇÃO 1: Instanciando o gerenciador de cache correto ---
        if (_cacheManager == null) _cacheManager = new DiskOnlyCacheManager(_mathEngine, embeddingSize, hiddenSize);
    }

    private IMathTensor InitializeTensor(int rows, int cols, Random rand)
    {
        double[] data = new double[rows * cols];
        double limit = Math.Sqrt(6.0 / (rows + cols));
        for (int i = 0; i < data.Length; i++) data[i] = (rand.NextDouble() * 2 - 1) * limit;
        return _mathEngine.CreateTensor(data, new[] { rows, cols });
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

    // NeuralNetworkLSTM.cs
    private double[] Softmax(double[] logits)
    {
        if (logits == null || logits.Length == 0) return Array.Empty<double>();

        var output = new double[logits.Length];

        // Clipping de entrada para prevenir overflow
        double maxLogit = logits.Max();
        if (double.IsNaN(maxLogit) || double.IsInfinity(maxLogit))
        {
            Console.WriteLine("[AVISO] Softmax recebeu valores inválidos. Retornando distribuição uniforme.");
            Array.Fill(output, 1.0 / logits.Length);
            return output;
        }

        // Limita logits para prevenir exp() overflow
        const double MAX_LOGIT = 88.0; // exp(88) ≈ 1.65e38 (seguro para double)
        maxLogit = Math.Min(maxLogit, MAX_LOGIT);

        double sumExp = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            double clippedLogit = Math.Min(logits[i], MAX_LOGIT) - maxLogit;
            output[i] = Math.Exp(clippedLogit);
            sumExp += output[i];
        }

        // Proteção contra underflow
        if (sumExp < 1e-20 || double.IsNaN(sumExp) || double.IsInfinity(sumExp))
        {
            Console.WriteLine($"[AVISO] Softmax sumExp inválido: {sumExp}. Usando distribuição uniforme.");
            Array.Fill(output, 1.0 / logits.Length);
            return output;
        }

        // Normalização
        for (int i = 0; i < logits.Length; i++)
        {
            output[i] /= sumExp;

            // Proteção final
            if (double.IsNaN(output[i]) || double.IsInfinity(output[i]))
            {
                output[i] = 1e-10;
            }
        }

        return output;
    }

// ========================================
// TRAIN SEQUENCE - FIX CRÍTICO DE GRADIENTES
// Garante liberação de TODOS os tensores
// ========================================
    public double TrainSequence(int[] inputIndices, int[] targetIndices, double learningRate)
    {
        IMathTensor? targetsGpu = null;
        IMathTensor? predictions = null;
        Dictionary<string, IMathTensor>? gradients = null;

        try
        {
            // 🔥 VALIDAÇÃO ANTES DO FORWARD
            //Console.WriteLine($"\n[TrainSequence] Validando pesos ANTES do forward...");
            ValidateWeightsQuick("ANTES Forward");

            _cacheManager.Reset();

            targetsGpu = _mathEngine.CreateOneHotTensor(targetIndices, this.OutputSize);

            var (pred, loss) = ForwardPassGpuOptimized(inputIndices, targetsGpu, inputIndices.Length);
            predictions = pred;

            // 🔥 VALIDAÇÃO APÓS FORWARD, ANTES DO BACKWARD
            //Console.WriteLine($"[TrainSequence] Validando pesos APÓS forward...");
            ValidateWeightsQuick("APÓS Forward");

            gradients = BackwardPassGpuOptimized(targetsGpu, predictions, inputIndices, inputIndices.Length);

            // 🔥 VALIDAÇÃO APÓS BACKWARD, ANTES DO UPDATE
            //Console.WriteLine($"[TrainSequence] Validando pesos APÓS backward...");
            ValidateWeightsQuick("APÓS Backward");

            UpdateWeightsWithAdamGpu(gradients, learningRate);

            // 🔥 VALIDAÇÃO CRÍTICA APÓS UPDATE
            //Console.WriteLine($"[TrainSequence] Validando pesos APÓS update...");
            ValidateWeightsQuick("APÓS Update");

            foreach (var grad in gradients.Values)
            {
                grad?.Dispose();
            }

            gradients.Clear();
            gradients = null;

            predictions?.Dispose();
            predictions = null;
            targetsGpu?.Dispose();
            targetsGpu = null;

            if (_mathEngine.IsGpu)
            {
                if (gpuEngine == null) gpuEngine = _mathEngine as Galileu.Node.Gpu.GpuMathEngine;
                gpuEngine?.Synchronize();
            }

            _tensorPool?.Trim();
            GC.Collect(0, GCCollectionMode.Optimized, false);

            return loss;
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"[ERRO] TrainSequence falhou: {ex.Message}\nStack Trace: {ex.StackTrace}");
            Console.ResetColor();
            throw;
        }
        finally
        {
            if (predictions != null) predictions.Dispose();
            if (targetsGpu != null) targetsGpu.Dispose();
            if (gradients != null)
            {
                foreach (var grad in gradients.Values)
                {
                    grad?.Dispose();
                }

                gradients.Clear();
            }
        }
    }

    private void ValidateWeightsQuick(string stage)
    {
        var criticalWeights = new[]
        {
            (weightsInputForget, "weightsInputForget"),
            (weightsHiddenForget, "weightsHiddenForget"),
            (weightsEmbedding, "weightsEmbedding")
        };

        foreach (var (tensor, name) in criticalWeights)
        {
            var data = tensor!.ToCpuTensor().GetData();
            int nanCount = data.Count(d => double.IsNaN(d));
            int infCount = data.Count(d => double.IsInfinity(d));

            if (nanCount > 0 || infCount > 0)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                //Console.WriteLine($"  ❌ [{stage}] {name}: NaN={nanCount}, Inf={infCount}");

                // Amostra de valores corrompidos
                var samples = data.Where(d => double.IsNaN(d) || double.IsInfinity(d)).Take(10).ToArray();
                //Console.WriteLine($"     Samples inválidos: [{string.Join(", ", samples)}]");

                // Amostra de valores válidos (se houver)
                var validSamples = data.Where(d => !double.IsNaN(d) && !double.IsInfinity(d)).Take(5).ToArray();
                if (validSamples.Length > 0)
                {
                    //Console.WriteLine($"     Samples válidos: [{string.Join(", ", validSamples.Select(x => $"{x:F6}"))}]");
                }

                Console.ResetColor();

                throw new InvalidOperationException($"[{stage}] {name} contém NaN/Inf. Treinamento abortado.");
            }
            else
            {
                // Estatísticas resumidas
                double mean = data.Average();
                double std = Math.Sqrt(data.Average(d => Math.Pow(d - mean, 2)));
                double max = data.Max();
                double min = data.Min();

                //Console.WriteLine($"  ✓ [{stage}] {name}: OK (mean={mean:E4}, std={std:E4}, range=[{min:E4}, {max:E4}])");
            }
        }
    }

    private (IMathTensor predictions, double loss) ForwardPassGpuOptimized(
        int[] inputIndices, IMathTensor targets, int sequenceLength)
    {
        // 🔥 MODO DEBUG: Ativa apenas no primeiro batch
        bool debugMode = true;

        if (debugMode)
        {
            //Console.WriteLine($"\n[DEBUG] ======= FORWARD PASS DEBUG =======");
            //Console.WriteLine($"[DEBUG] Sequence length: {sequenceLength}");
            //Console.WriteLine($"[DEBUG] Input indices: [{string.Join(", ", inputIndices)}]");
        }

        var predictions = _mathEngine.CreateTensor(new[] { sequenceLength, outputSize });
        double sequenceLoss = 0;

        IMathTensor h_prev = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
        IMathTensor c_prev = _mathEngine.CreateTensor(new[] { 1, hiddenSize });

        int embeddingSize = weightsEmbedding!.Shape[1];

        IMathTensor linearBuffer = _tensorPool!.Rent(new[] { 1, hiddenSize });
        IMathTensor temp1 = _tensorPool.Rent(new[] { 1, hiddenSize });
        IMathTensor temp2 = _tensorPool.Rent(new[] { 1, hiddenSize });
        IMathTensor outputLinear = _tensorPool.Rent(new[] { 1, outputSize });
        IMathTensor outputSoftmax = _tensorPool.Rent(new[] { 1, outputSize });

        try
        {
            for (int t = 0; t < sequenceLength; t++)
            {
                if (debugMode && t == 0)
                {
                    //Console.WriteLine($"\n[DEBUG] --- Timestep {t} ---");
                }

                IMathTensor? inputEmbedding = null;
                IMathTensor? forgetGate = null;
                IMathTensor? inputGate = null;
                IMathTensor? cellCandidate = null;
                IMathTensor? outputGate = null;
                IMathTensor? cellNext = null;
                IMathTensor? tanhCellNext = null;
                IMathTensor? hiddenNext = null;
                IMathTensor? cloneHiddenPrev = null;
                IMathTensor? cloneCellPrev = null;

                try
                {
                    inputEmbedding = _tensorPool.Rent(new[] { 1, embeddingSize });
                    forgetGate = _tensorPool.Rent(new[] { 1, hiddenSize });
                    inputGate = _tensorPool.Rent(new[] { 1, hiddenSize });
                    cellCandidate = _tensorPool.Rent(new[] { 1, hiddenSize });
                    outputGate = _tensorPool.Rent(new[] { 1, hiddenSize });
                    cellNext = _tensorPool.Rent(new[] { 1, hiddenSize });
                    tanhCellNext = _tensorPool.Rent(new[] { 1, hiddenSize });
                    hiddenNext = _tensorPool.Rent(new[] { 1, hiddenSize });

                    // 🔥 PASSO 1: LOOKUP
                    int tokenIndex = inputIndices[t];

                    if (debugMode && t == 0)
                    {
                        //Console.WriteLine($"[DEBUG] Token index: {tokenIndex}");

                        // Valida matriz de embedding ANTES do lookup
                        var embWeightsBefore = weightsEmbedding.ToCpuTensor().GetData();
                        int offset = tokenIndex * embeddingSize;
                        var tokenWeights = embWeightsBefore.Skip(offset).Take(embeddingSize).ToArray();

                        int preNan = tokenWeights.Count(x => double.IsNaN(x));
                        int preInf = tokenWeights.Count(x => double.IsInfinity(x));

                        //Console.WriteLine($"[DEBUG] Pesos do token ANTES lookup: NaN={preNan}, Inf={preInf}");
                        //Console.WriteLine($"[DEBUG] Sample pesos: [{string.Join(", ", tokenWeights.Take(5).Select(x => $"{x:F6}"))}...]");
                    }

                    _mathEngine.Lookup(weightsEmbedding!, tokenIndex, inputEmbedding);

                    if (debugMode && t == 0)
                    {
                        var embData = inputEmbedding.ToCpuTensor().GetData();
                        int nanCount = embData.Count(x => double.IsNaN(x));
                        int infCount = embData.Count(x => double.IsInfinity(x));

                        //Console.WriteLine($"[DEBUG] APÓS lookup: NaN={nanCount}, Inf={infCount}");
                        //Console.WriteLine($"[DEBUG] Sample embedding: [{string.Join(", ", embData.Take(5).Select(x => $"{x:F6}"))}...]");

                        if (nanCount > 0 || infCount > 0)
                        {
                            Console.ForegroundColor = ConsoleColor.Red;
                            Console.WriteLine($"\n❌ LOOKUP PRODUZIU NaN/Inf!");
                            Console.WriteLine($"   Isso indica BUG NO KERNEL GPU 'lookup'");
                            Console.ResetColor();

                            throw new InvalidOperationException("Lookup kernel produziu valores inválidos");
                        }
                    }

                    // 🔥 PASSO 2: FORGET GATE - MatMul 1
                    if (debugMode && t == 0)
                        //Console.WriteLine($"[DEBUG] [1] MatMul: embedding × weightsInputForget...");
                    _mathEngine.MatrixMultiply(inputEmbedding, weightsInputForget!, temp1);

                    if (debugMode && t == 0)
                    {
                        var data = temp1.ToCpuTensor().GetData();
                        //Console.WriteLine($"[DEBUG]     Resultado: NaN={data.Count(x => double.IsNaN(x))}, Sample=[{string.Join(", ", data.Take(3).Select(x => $"{x:F6}"))}...]");

                        if (data.Any(x => double.IsNaN(x)))
                        {
                            Console.ForegroundColor = ConsoleColor.Red;
                            Console.WriteLine($"❌ PRIMEIRA MATMUL PRODUZIU NaN!");
                            Console.ResetColor();
                            throw new InvalidOperationException("MatMul (embedding × weightsInputForget) produziu NaN");
                        }
                    }

                    // 🔥 PASSO 3: FORGET GATE - MatMul 2
                    //if (debugMode && t == 0) Console.WriteLine($"[DEBUG] [2] MatMul: h_prev × weightsHiddenForget...");
                    _mathEngine.MatrixMultiply(h_prev, weightsHiddenForget!, temp2);

                    if (debugMode && t == 0)
                    {
                        var data = temp2.ToCpuTensor().GetData();
                        //Console.WriteLine($"[DEBUG]     Resultado: NaN={data.Count(x => double.IsNaN(x))}, Sample=[{string.Join(", ", data.Take(3).Select(x => $"{x:F6}"))}...]");

                        if (data.Any(x => double.IsNaN(x)))
                        {
                            Console.ForegroundColor = ConsoleColor.Red;
                            Console.WriteLine($"❌ SEGUNDA MATMUL PRODUZIU NaN!");
                            Console.ResetColor();
                            throw new InvalidOperationException("MatMul (h_prev × weightsHiddenForget) produziu NaN");
                        }
                    }

                    // 🔥 PASSO 4: ADD
                    //if (debugMode && t == 0) Console.WriteLine($"[DEBUG] [3] Add: temp1 + temp2...");
                    _mathEngine.Add(temp1, temp2, linearBuffer);

                    if (debugMode && t == 0)
                    {
                        var data = linearBuffer.ToCpuTensor().GetData();
                        //Console.WriteLine($"[DEBUG]     Resultado: NaN={data.Count(x => double.IsNaN(x))}, Sample=[{string.Join(", ", data.Take(3).Select(x => $"{x:F6}"))}...]");

                        if (data.Any(x => double.IsNaN(x)))
                        {
                            Console.ForegroundColor = ConsoleColor.Red;
                            Console.WriteLine($"❌ ADD PRODUZIU NaN!");
                            Console.ResetColor();
                            throw new InvalidOperationException("Add produziu NaN");
                        }
                    }

                    // 🔥 PASSO 5: ADD BROADCAST
                    //if (debugMode && t == 0) Console.WriteLine($"[DEBUG] [4] AddBroadcast: linearBuffer + biasForget...");
                    _mathEngine.AddBroadcast(linearBuffer, biasForget!, linearBuffer);

                    if (debugMode && t == 0)
                    {
                        var data = linearBuffer.ToCpuTensor().GetData();
                        //Console.WriteLine($"[DEBUG]     Resultado: NaN={data.Count(x => double.IsNaN(x))}, Sample=[{string.Join(", ", data.Take(3).Select(x => $"{x:F6}"))}...]");

                        if (data.Any(x => double.IsNaN(x)))
                        {
                            Console.ForegroundColor = ConsoleColor.Red;
                            Console.WriteLine($"❌ ADDBROADCAST PRODUZIU NaN!");
                            Console.ResetColor();
                            throw new InvalidOperationException("AddBroadcast produziu NaN");
                        }
                    }

                    // 🔥 PASSO 6: SIGMOID
                    //if (debugMode && t == 0) Console.WriteLine($"[DEBUG] [5] Sigmoid...");
                    _mathEngine.Sigmoid(linearBuffer, forgetGate);

                    if (debugMode && t == 0)
                    {
                        var data = forgetGate.ToCpuTensor().GetData();
                        //Console.WriteLine($"[DEBUG]     Resultado: NaN={data.Count(x => double.IsNaN(x))}, Sample=[{string.Join(", ", data.Take(3).Select(x => $"{x:F6}"))}...]");

                        if (data.Any(x => double.IsNaN(x)))
                        {
                            Console.ForegroundColor = ConsoleColor.Red;
                            Console.WriteLine($"❌ SIGMOID PRODUZIU NaN!");
                            Console.ResetColor();
                            throw new InvalidOperationException("Sigmoid produziu NaN");
                        }
                    }

                    // Continua com resto do LSTM (sem debug para economizar output)
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

                    cloneHiddenPrev = _mathEngine.Clone(h_prev);
                    cloneCellPrev = _mathEngine.Clone(c_prev);

                    var stepCache = new LstmStepCache
                    {
                        Input = inputEmbedding,
                        HiddenPrev = cloneHiddenPrev,
                        CellPrev = cloneCellPrev,
                        ForgetGate = forgetGate,
                        InputGate = inputGate,
                        CellCandidate = cellCandidate,
                        OutputGate = outputGate,
                        CellNext = cellNext,
                        TanhCellNext = tanhCellNext,
                        HiddenNext = hiddenNext
                    };

                    _cacheManager.CacheStep(stepCache);

                    // 🔥 OUTPUT LAYER
                    //if (debugMode && t == 0) Console.WriteLine($"[DEBUG] [6] Output Layer: hiddenNext × weightsHiddenOutputFinal...");
                    _mathEngine.MatrixMultiply(hiddenNext, weightsHiddenOutputFinal!, outputLinear);

                    if (debugMode && t == 0)
                    {
                        var data = outputLinear.ToCpuTensor().GetData();
                        int nanCount = data.Count(x => double.IsNaN(x));
                        //Console.WriteLine($"[DEBUG]     Antes AddBroadcast: NaN={nanCount}, Sample=[{string.Join(", ", data.Take(3).Select(x => $"{x:F6}"))}...]");

                        if (nanCount > 0)
                        {
                            Console.ForegroundColor = ConsoleColor.Red;
                            Console.WriteLine($"❌ OUTPUT MATMUL PRODUZIU NaN!");
                            Console.ResetColor();
                            throw new InvalidOperationException(
                                "MatMul (hiddenNext × weightsHiddenOutputFinal) produziu NaN");
                        }
                    }

                    _mathEngine.AddBroadcast(outputLinear, biasOutputFinal!, outputLinear);

                    if (debugMode && t == 0)
                    {
                        var data = outputLinear.ToCpuTensor().GetData();
                        int nanCount = data.Count(x => double.IsNaN(x));
                        //Console.WriteLine($"[DEBUG]     Após AddBroadcast: NaN={nanCount}, Sample=[{string.Join(", ", data.Take(3).Select(x => $"{x:F6}"))}...]");

                        if (nanCount > 0)
                        {
                            Console.ForegroundColor = ConsoleColor.Red;
                            Console.WriteLine($"❌ OUTPUT ADDBROADCAST PRODUZIU NaN!");
                            Console.ResetColor();
                            throw new InvalidOperationException("AddBroadcast (output bias) produziu NaN");
                        }
                    }

                    _mathEngine.Softmax(outputLinear, outputSoftmax);
                    _mathEngine.Set(predictions, t, outputSoftmax);

                    h_prev.Dispose();
                    c_prev.Dispose();

                    h_prev = _mathEngine.Clone(hiddenNext);
                    c_prev = _mathEngine.Clone(cellNext);

                    //if (debugMode && t == 0) Console.WriteLine($"[DEBUG] ✓ Timestep 0 completado com sucesso!");
                }
                finally
                {
                    if (inputEmbedding != null) _tensorPool.Return(inputEmbedding);
                    if (forgetGate != null) _tensorPool.Return(forgetGate);
                    if (inputGate != null) _tensorPool.Return(inputGate);
                    if (cellCandidate != null) _tensorPool.Return(cellCandidate);
                    if (outputGate != null) _tensorPool.Return(outputGate);
                    if (cellNext != null) _tensorPool.Return(cellNext);
                    if (tanhCellNext != null) _tensorPool.Return(tanhCellNext);
                    if (hiddenNext != null) _tensorPool.Return(hiddenNext);
                    cloneHiddenPrev?.Dispose();
                    cloneCellPrev?.Dispose();
                }
            }

            // Cálculo de loss
            var predData = predictions.ToCpuTensor().GetData();
            var targetData = targets.ToCpuTensor().GetData();

            for (int t = 0; t < sequenceLength; t++)
            {
                int offset = t * outputSize;
                int targetIndexInFullArray = Array.IndexOf(targetData, 1.0, offset, outputSize);

                if (targetIndexInFullArray != -1)
                {
                    int targetLocalIndex = targetIndexInFullArray - offset;
                    double prob = predData[offset + targetLocalIndex];
                    prob = Math.Max(prob, 1e-10);
                    prob = Math.Min(prob, 1.0 - 1e-10);
                    sequenceLoss += -Math.Log(prob);
                }
            }

            double avgLoss = sequenceLoss / Math.Max(sequenceLength, 1);
            return (predictions, avgLoss);
        }
        finally
        {
            _tensorPool.Return(linearBuffer);
            _tensorPool.Return(temp1);
            _tensorPool.Return(temp2);
            _tensorPool.Return(outputLinear);
            _tensorPool.Return(outputSoftmax);
            h_prev?.Dispose();
            c_prev?.Dispose();
        }
    }

// ========================================
// BACKWARD PASS - PREVENÇÃO ATIVA DE EXPLOSÃO DE GRADIENTES
// Implementa clipping proativo em gradientes intermediários.
// ========================================
private Dictionary<string, IMathTensor> BackwardPassGpuOptimized(
    IMathTensor targets, IMathTensor predictions,
    int[] inputIndices, int sequenceLength)
{
    // Limite para o clipping de valor. Impede que qualquer componente único do gradiente
    // se torne muito grande, prevenindo overflows para Infinito (Inf).
    const double MAX_GRAD_VALUE = 5.0; 
    var grads = InitializeGradientsGpu();

    IMathTensor? dh_next = null, dc_next = null, dy = null, d_embedding = null;
    IMathTensor? current_dy = null, dh = null, dc = null, dog = null, dfg = null, dig = null, dcc = null;
    IMathTensor? temp_deriv = null, temp_mult = null, temp_grad_why = null, temp_grad_w_in = null, temp_grad_w_hid = null;

    try
    {
        // FASE 1: Aloca buffers temporários
        dh_next = _tensorPool!.Rent(new[] { 1, hiddenSize });
        dc_next = _tensorPool.Rent(new[] { 1, hiddenSize });
        dy = _tensorPool.Rent(new[] { sequenceLength, outputSize });
        d_embedding = _tensorPool.Rent(new[] { 1, weightsEmbedding!.Shape[1] });
        current_dy = _tensorPool.Rent(new[] { 1, outputSize });
        dh = _tensorPool.Rent(new[] { 1, hiddenSize });
        dc = _tensorPool.Rent(new[] { 1, hiddenSize });
        dog = _tensorPool.Rent(new[] { 1, hiddenSize });
        dfg = _tensorPool.Rent(new[] { 1, hiddenSize });
        dig = _tensorPool.Rent(new[] { 1, hiddenSize });
        dcc = _tensorPool.Rent(new[] { 1, hiddenSize });
        temp_deriv = _tensorPool.Rent(new[] { 1, hiddenSize });
        temp_mult = _tensorPool.Rent(new[] { 1, hiddenSize });
        temp_grad_why = _tensorPool.Rent(new[] { hiddenSize, outputSize });
        temp_grad_w_in = _tensorPool.Rent(new[] { weightsEmbedding.Shape[1], hiddenSize });
        temp_grad_w_hid = _tensorPool.Rent(new[] { hiddenSize, hiddenSize });

        _mathEngine.Subtract(predictions, targets, dy);

        for (int t = sequenceLength - 1; t >= 0; t--)
        {
            _mathEngine.Slice(dy, t, current_dy);
            Dictionary<string, IMathTensor>? timestepCache = null;

            try
            {
                timestepCache = _cacheManager.RetrieveMultipleTensors(t,
                    DiskOnlyCacheManager.TensorNames.Input, DiskOnlyCacheManager.TensorNames.HiddenPrev,
                    DiskOnlyCacheManager.TensorNames.HiddenNext, DiskOnlyCacheManager.TensorNames.CellPrev,
                    DiskOnlyCacheManager.TensorNames.ForgetGate, DiskOnlyCacheManager.TensorNames.InputGate,
                    DiskOnlyCacheManager.TensorNames.CellCandidate, DiskOnlyCacheManager.TensorNames.OutputGate,
                    DiskOnlyCacheManager.TensorNames.TanhCellNext
                );

                var input = timestepCache[DiskOnlyCacheManager.TensorNames.Input];
                var hiddenPrev = timestepCache[DiskOnlyCacheManager.TensorNames.HiddenPrev];
                var hiddenNext = timestepCache[DiskOnlyCacheManager.TensorNames.HiddenNext];
                var cellPrev = timestepCache[DiskOnlyCacheManager.TensorNames.CellPrev];
                var forgetGate = timestepCache[DiskOnlyCacheManager.TensorNames.ForgetGate];
                var inputGate = timestepCache[DiskOnlyCacheManager.TensorNames.InputGate];
                var cellCandidate = timestepCache[DiskOnlyCacheManager.TensorNames.CellCandidate];
                var outputGate = timestepCache[DiskOnlyCacheManager.TensorNames.OutputGate];
                var tanhCellNext = timestepCache[DiskOnlyCacheManager.TensorNames.TanhCellNext];
                
                // Gradiente de Output Final
                _mathEngine.MatrixMultiplyTransposeA(hiddenNext, current_dy, temp_grad_why);
                _mathEngine.Clip(temp_grad_why, -MAX_GRAD_VALUE, MAX_GRAD_VALUE); // 🔥 Clipping Proativo
                _mathEngine.Add(grads["why"], temp_grad_why, grads["why"]);
                _mathEngine.Add(grads["by"], current_dy, grads["by"]);

                _mathEngine.MatrixMultiplyTransposeB(current_dy, weightsHiddenOutputFinal!, dh);
                _mathEngine.Add(dh, dh_next!, dh);
                
                _mathEngine.Multiply(dh, tanhCellNext, dog);
                _mathEngine.SigmoidDerivative(outputGate, temp_deriv);
                _mathEngine.Multiply(dog, temp_deriv, dog);
                
                _mathEngine.Multiply(dh, outputGate, temp_mult);
                _mathEngine.TanhDerivative(tanhCellNext, temp_deriv);
                _mathEngine.Multiply(temp_mult, temp_deriv, temp_mult);
                _mathEngine.Add(dc_next!, temp_mult, dc);

                _mathEngine.Clip(dh, -MAX_GRAD_VALUE, MAX_GRAD_VALUE);
                _mathEngine.Clip(dc, -MAX_GRAD_VALUE, MAX_GRAD_VALUE);
                
                // Atualiza pesos do Output Gate
                _mathEngine.MatrixMultiplyTransposeA(input, dog, temp_grad_w_in);
                _mathEngine.Clip(temp_grad_w_in, -MAX_GRAD_VALUE, MAX_GRAD_VALUE); // 🔥 Clipping Proativo
                _mathEngine.Add(grads["wio"], temp_grad_w_in, grads["wio"]);
                _mathEngine.MatrixMultiplyTransposeA(hiddenPrev, dog, temp_grad_w_hid);
                _mathEngine.Clip(temp_grad_w_hid, -MAX_GRAD_VALUE, MAX_GRAD_VALUE); // 🔥 Clipping Proativo
                _mathEngine.Add(grads["who"], temp_grad_w_hid, grads["who"]);
                _mathEngine.Add(grads["bo"], dog, grads["bo"]);

                // Gradiente e pesos do Forget Gate
                _mathEngine.Multiply(dc, cellPrev, dfg);
                _mathEngine.SigmoidDerivative(forgetGate, temp_deriv);
                _mathEngine.Multiply(dfg, temp_deriv, dfg);
                _mathEngine.MatrixMultiplyTransposeA(input, dfg, temp_grad_w_in);
                _mathEngine.Clip(temp_grad_w_in, -MAX_GRAD_VALUE, MAX_GRAD_VALUE); // 🔥 Clipping Proativo
                _mathEngine.Add(grads["wif"], temp_grad_w_in, grads["wif"]);
                _mathEngine.MatrixMultiplyTransposeA(hiddenPrev, dfg, temp_grad_w_hid);
                _mathEngine.Clip(temp_grad_w_hid, -MAX_GRAD_VALUE, MAX_GRAD_VALUE); // 🔥 Clipping Proativo
                _mathEngine.Add(grads["whf"], temp_grad_w_hid, grads["whf"]);
                _mathEngine.Add(grads["bf"], dfg, grads["bf"]);

                // Gradiente e pesos do Input Gate
                _mathEngine.Multiply(dc, cellCandidate, dig);
                _mathEngine.SigmoidDerivative(inputGate, temp_deriv);
                _mathEngine.Multiply(dig, temp_deriv, dig);
                _mathEngine.MatrixMultiplyTransposeA(input, dig, temp_grad_w_in);
                _mathEngine.Clip(temp_grad_w_in, -MAX_GRAD_VALUE, MAX_GRAD_VALUE); // 🔥 Clipping Proativo
                _mathEngine.Add(grads["wii"], temp_grad_w_in, grads["wii"]);
                _mathEngine.MatrixMultiplyTransposeA(hiddenPrev, dig, temp_grad_w_hid);
                _mathEngine.Clip(temp_grad_w_hid, -MAX_GRAD_VALUE, MAX_GRAD_VALUE); // 🔥 Clipping Proativo
                _mathEngine.Add(grads["whi"], temp_grad_w_hid, grads["whi"]);
                _mathEngine.Add(grads["bi"], dig, grads["bi"]);
                
                // Gradiente e pesos do Cell Candidate
                _mathEngine.Multiply(dc, inputGate, dcc);
                _mathEngine.TanhDerivative(cellCandidate, temp_deriv);
                _mathEngine.Multiply(dcc, temp_deriv, dcc);
                _mathEngine.MatrixMultiplyTransposeA(input, dcc, temp_grad_w_in);
                _mathEngine.Clip(temp_grad_w_in, -MAX_GRAD_VALUE, MAX_GRAD_VALUE); // 🔥 Clipping Proativo
                _mathEngine.Add(grads["wic"], temp_grad_w_in, grads["wic"]);
                _mathEngine.MatrixMultiplyTransposeA(hiddenPrev, dcc, temp_grad_w_hid);
                _mathEngine.Clip(temp_grad_w_hid, -MAX_GRAD_VALUE, MAX_GRAD_VALUE); // 🔥 Clipping Proativo
                _mathEngine.Add(grads["whc"], temp_grad_w_hid, grads["whc"]);
                _mathEngine.Add(grads["bc"], dcc, grads["bc"]);

                // Gradiente de Embedding (acumulativo)
                _mathEngine.MatrixMultiplyTransposeB(dfg, weightsInputForget!, d_embedding);
                _mathEngine.MatrixMultiplyTransposeB(dig, weightsInputInput!, temp_mult);
                _mathEngine.Add(d_embedding, temp_mult, d_embedding);
                _mathEngine.MatrixMultiplyTransposeB(dcc, weightsInputCell!, temp_mult);
                _mathEngine.Add(d_embedding, temp_mult, d_embedding);
                _mathEngine.MatrixMultiplyTransposeB(dog, weightsInputOutput!, temp_mult);
                _mathEngine.Add(d_embedding, temp_mult, d_embedding);
                _mathEngine.Clip(d_embedding, -MAX_GRAD_VALUE, MAX_GRAD_VALUE); // 🔥 Clipping Proativo
                _mathEngine.AccumulateGradient(grads["w_embed"], d_embedding, inputIndices[t]);

                // Propagação para timestep anterior
                _mathEngine.MatrixMultiplyTransposeB(dfg, weightsHiddenForget!, dh_next!);
                _mathEngine.MatrixMultiplyTransposeB(dig, weightsHiddenInput!, temp_mult);
                _mathEngine.Add(dh_next!, temp_mult, dh_next!);
                _mathEngine.MatrixMultiplyTransposeB(dcc, weightsHiddenCell!, temp_mult);
                _mathEngine.Add(dh_next!, temp_mult, dh_next!);
                _mathEngine.MatrixMultiplyTransposeB(dog, weightsHiddenOutput!, temp_mult);
                _mathEngine.Add(dh_next!, temp_mult, dh_next!);
                _mathEngine.Multiply(dc, forgetGate, dc_next!);
            }
            finally
            {
                if (timestepCache != null)
                {
                    foreach (var tensor in timestepCache.Values) tensor?.Dispose();
                    timestepCache.Clear();
                }
            }
        }

        #if DEBUG_GRADIENTS
        // O bloco de diagnóstico continua útil para observabilidade
        #endif

        // A rede de segurança final (clipping de norma total) ainda é mantida.
        double totalNorm = 0;
        foreach (var grad in grads.Values)
        {
            var gradData = grad.ToCpuTensor().GetData();
            if (gradData.Any(d => double.IsNaN(d) || double.IsInfinity(d)))
            {
                totalNorm = double.PositiveInfinity;
                break;
            }
            totalNorm += gradData.Sum(x => x * x);
        }
        totalNorm = Math.Sqrt(totalNorm);

        const double MAX_GRAD_NORM = 1.0;
        if (double.IsNaN(totalNorm) || double.IsInfinity(totalNorm) || totalNorm > MAX_GRAD_NORM)
        {
            if (!double.IsNaN(totalNorm) && !double.IsInfinity(totalNorm))
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.Write($" [Clipping Ativo: {totalNorm:F2} -> {MAX_GRAD_NORM}] ");
                Console.ResetColor();
            }
            
            double clipScale = MAX_GRAD_NORM / (totalNorm + 1e-8);
            foreach (var grad in grads.Values) _mathEngine.Scale(grad, clipScale);
        }

        return grads;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[ERRO] BackwardPass falhou: {ex.Message}\nStack: {ex.StackTrace}");
        DisposeGradients(grads);
        throw;
    }
    finally
    {
        // Limpeza dos buffers temporários
        if (dh_next != null) _tensorPool!.Return(dh_next);
        if (dc_next != null) _tensorPool!.Return(dc_next);
        if (dy != null) _tensorPool!.Return(dy);
        if (d_embedding != null) _tensorPool!.Return(d_embedding);
        if (current_dy != null) _tensorPool!.Return(current_dy);
        if (dh != null) _tensorPool!.Return(dh);
        if (dc != null) _tensorPool!.Return(dc);
        if (dog != null) _tensorPool!.Return(dog);
        if (dfg != null) _tensorPool!.Return(dfg);
        if (dig != null) _tensorPool!.Return(dig);
        if (dcc != null) _tensorPool!.Return(dcc);
        if (temp_deriv != null) _tensorPool!.Return(temp_deriv);
        if (temp_mult != null) _tensorPool!.Return(temp_mult);
        if (temp_grad_why != null) _tensorPool!.Return(temp_grad_why);
        if (temp_grad_w_in != null) _tensorPool!.Return(temp_grad_w_in);
        if (temp_grad_w_hid != null) _tensorPool!.Return(temp_grad_w_hid);
    }
}

    /// <summary>
    /// Aplica os gradientes calculados para atualizar os pesos do modelo usando o otimizador Adam.
    /// OTIMIZADO: Esta função não move mais dados entre CPU e GPU. Ela delega a lógica de
    /// atualização para o AdamOptimizer, que por sua vez executa um kernel otimizado na GPU
    /// para atualizar os pesos, momentos (m, v) e gradientes in-place.
    /// </summary>
    /// <param name="gradients">Um dicionário contendo os tensores de gradiente para cada camada de peso.</param>
    /// <param name="learningRate">A taxa de aprendizado (não usada diretamente aqui, pois já está configurada no AdamOptimizer).</param>
    // SUBSTITUA o método UpdateWeightsWithAdamGpu no NeuralNetworkLSTM.cs
    private void UpdateWeightsWithAdamGpu(Dictionary<string, IMathTensor> gradients, double learningRate)
    {
        var parameters = new Dictionary<string, IMathTensor>
        {
            { "w_embed", weightsEmbedding! },
            { "wif", weightsInputForget! },
            { "whf", weightsHiddenForget! },
            { "wii", weightsInputInput! },
            { "whi", weightsHiddenInput! },
            { "wic", weightsInputCell! },
            { "whc", weightsHiddenCell! },
            { "wio", weightsInputOutput! },
            { "who", weightsHiddenOutput! },
            { "why", weightsHiddenOutputFinal! },
            { "bf", biasForget! },
            { "bi", biasInput! },
            { "bc", biasCell! },
            { "bo", biasOutput! },
            { "by", biasOutputFinal! }
        };

        int layerId = 0;

        foreach (var key in parameters.Keys.Where(gradients.ContainsKey))
        {
            var paramTensor = parameters[key];
            var gradTensor = gradients[key];

            // 🔥 VALIDAÇÃO DOS GRADIENTES ANTES DE APLICAR
            var gradData = gradTensor.ToCpuTensor().GetData();
            int nanCount = gradData.Count(d => double.IsNaN(d));
            int infCount = gradData.Count(d => double.IsInfinity(d));

            if (nanCount > 0 || infCount > 0)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[UpdateWeights] AVISO: Gradiente '{key}' contém valores inválidos:");
                Console.WriteLine($"  NaN: {nanCount}, Inf: {infCount}");
                Console.WriteLine($"  Pulando atualização deste tensor para prevenir corrupção");
                Console.ResetColor();

                // Não aplica update se gradiente for inválido
                layerId++;
                continue;
            }

            // 🔥 VALIDAÇÃO DOS PARÂMETROS ANTES DA ATUALIZAÇÃO
            var paramDataBefore = paramTensor.ToCpuTensor().GetData();
            int paramNanBefore = paramDataBefore.Count(d => double.IsNaN(d));
            int paramInfBefore = paramDataBefore.Count(d => double.IsInfinity(d));

            if (paramNanBefore > 0 || paramInfBefore > 0)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[UpdateWeights] ERRO: Parâmetro '{key}' JÁ estava corrompido ANTES do update:");
                Console.WriteLine($"  NaN: {paramNanBefore}, Inf: {paramInfBefore}");
                Console.ResetColor();

                throw new InvalidOperationException($"Parâmetro '{key}' corrompido antes do Adam update");
            }

            // Aplica Adam update
            _adamOptimizer.UpdateParametersGpu(layerId, paramTensor, gradTensor, _mathEngine);

            // 🔥 SINCRONIZAÇÃO FORÇADA
            if (_mathEngine.IsGpu)
            {
                if (gpuEngine == null) gpuEngine = _mathEngine as Galileu.Node.Gpu.GpuMathEngine;
                gpuEngine?.Synchronize();
            }

            // 🔥 VALIDAÇÃO DOS PARÂMETROS APÓS A ATUALIZAÇÃO
            var paramDataAfter = paramTensor.ToCpuTensor().GetData();
            int paramNanAfter = paramDataAfter.Count(d => double.IsNaN(d));
            int paramInfAfter = paramDataAfter.Count(d => double.IsInfinity(d));

            if (paramNanAfter > 0 || paramInfAfter > 0)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[UpdateWeights] ❌ ADAM UPDATE CORROMPEU '{key}':");
                Console.WriteLine($"  NaN: {paramNanBefore} → {paramNanAfter}");
                Console.WriteLine($"  Inf: {paramInfBefore} → {paramInfAfter}");

                // Debug: mostra estatísticas do gradiente
                double gradMean = gradData.Average();
                double gradStd = Math.Sqrt(gradData.Average(d => Math.Pow(d - gradMean, 2)));
                Console.WriteLine($"  Gradiente stats: mean={gradMean:E4}, std={gradStd:E4}");

                // Debug: mostra parâmetros antes e depois
                var validBefore = paramDataBefore.Where(d => !double.IsNaN(d) && !double.IsInfinity(d)).Take(5)
                    .ToArray();
                var validAfter = paramDataAfter.Where(d => !double.IsNaN(d) && !double.IsInfinity(d)).Take(5).ToArray();
                Console.WriteLine($"  Params antes: [{string.Join(", ", validBefore.Select(x => $"{x:F6}"))}]");
                Console.WriteLine($"  Params depois: [{string.Join(", ", validAfter.Select(x => $"{x:F6}"))}]");

                Console.ResetColor();

                throw new InvalidOperationException($"Adam update corrompeu parâmetro '{key}'");
            }

            layerId++;
        }

        //Console.WriteLine($"[UpdateWeights] ✓ Todos os {layerId} tensores atualizados com sucesso");
    }

// ========================================
// INITIALIZE GRADIENTS GPU - AUDITORIA
// ========================================

    private Dictionary<string, IMathTensor> InitializeGradientsGpu()
    {
        int embeddingSize = weightsEmbedding!.Shape[1];

        // ✅ Cria dicionário com 15 tensores
        var gradients = new Dictionary<string, IMathTensor>
        {
            // ✅ Embedding (maior tensor: vocabSize × embeddingSize)
            { "w_embed", _mathEngine.CreateTensor(new[] { inputSize, embeddingSize }) },

            // ✅ LSTM Weights - Input Gates (8 tensores)
            { "wif", _mathEngine.CreateTensor(new[] { embeddingSize, hiddenSize }) },
            { "whf", _mathEngine.CreateTensor(new[] { hiddenSize, hiddenSize }) },
            { "wii", _mathEngine.CreateTensor(new[] { embeddingSize, hiddenSize }) },
            { "whi", _mathEngine.CreateTensor(new[] { hiddenSize, hiddenSize }) },
            { "wic", _mathEngine.CreateTensor(new[] { embeddingSize, hiddenSize }) },
            { "whc", _mathEngine.CreateTensor(new[] { hiddenSize, hiddenSize }) },
            { "wio", _mathEngine.CreateTensor(new[] { embeddingSize, hiddenSize }) },
            { "who", _mathEngine.CreateTensor(new[] { hiddenSize, hiddenSize }) },

            // ✅ Output Layer (segundo maior tensor: hiddenSize × outputSize)
            { "why", _mathEngine.CreateTensor(new[] { hiddenSize, outputSize }) },

            // ✅ Biases (5 tensores pequenos)
            { "bf", _mathEngine.CreateTensor(new[] { 1, hiddenSize }) },
            { "bi", _mathEngine.CreateTensor(new[] { 1, hiddenSize }) },
            { "bc", _mathEngine.CreateTensor(new[] { 1, hiddenSize }) },
            { "bo", _mathEngine.CreateTensor(new[] { 1, hiddenSize }) },
            { "by", _mathEngine.CreateTensor(new[] { 1, outputSize }) }
        };

        // ⚠️ AVISO CRÍTICO:
        // Estes 15 tensores DEVEM ser liberados após uso!
        // O chamador (BackwardPass) retorna este dicionário,
        // e quem recebe (TrainSequence) DEVE fazer Dispose de TODOS.

        return gradients;
    }
// ========================================
// CÁLCULO DE MEMÓRIA DOS GRADIENTES
// ========================================

    private long CalculateGradientsMemoryMB()
    {
        int embeddingSize = weightsEmbedding!.Shape[1];
        long totalParams = 0;

        // Embedding
        totalParams += inputSize * embeddingSize;

        // LSTM Weights (4 gates × 2 tipos)
        totalParams += 4 * embeddingSize * hiddenSize; // Input weights
        totalParams += 4 * hiddenSize * hiddenSize; // Hidden weights

        // Output Layer
        totalParams += hiddenSize * outputSize;

        // Biases
        totalParams += 4 * hiddenSize; // LSTM biases
        totalParams += outputSize; // Output bias

        // Cada parâmetro: 8 bytes (double)
        long totalBytes = totalParams * sizeof(double);
        long totalMB = totalBytes / (1024 * 1024);

        return totalMB;
    }
// ========================================
// MÉTODO AUXILIAR: Libera Gradientes
// ========================================

    /// <summary>
    /// Libera TODOS os tensores de um dicionário de gradientes.
    /// Use este método para garantir que nenhum tensor fique órfão.
    /// </summary>
    private void DisposeGradients(Dictionary<string, IMathTensor>? gradients)
    {
        if (gradients == null) return;

        foreach (var kvp in gradients)
        {
            try
            {
                kvp.Value?.Dispose();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[AVISO] Erro ao liberar gradiente '{kvp.Key}': {ex.Message}");
            }
        }

        gradients.Clear();
    }

    private IMathTensor CreateSequenceTensor(Tensor[] sequence)
    {
        int sequenceLength = sequence.Length;
        if (sequenceLength == 0) return _mathEngine.CreateTensor(new[] { 0, 0 });
        int featureSize = sequence[0].GetShape()[0];
        var flatData = new double[sequenceLength * featureSize];
        for (int i = 0; i < sequenceLength; i++)
        {
            Array.Copy(sequence[i].GetData(), 0, flatData, i * featureSize, featureSize);
        }

        return _mathEngine.CreateTensor(flatData, new[] { sequenceLength, featureSize });
    }

    /// <summary>
    /// Calcula a perda (loss) para uma sequência de dados sem executar o backward pass ou a atualização de pesos.
    /// Este método é usado principalmente para validação do modelo.
    /// OTIMIZADO: Recebe os índices dos alvos e delega a criação do tensor de alvos para a IMathEngine,
    /// garantindo um baixo consumo de memória na CPU.
    /// </summary>
    /// <param name="inputIndices">Array de índices dos tokens de entrada para a sequência.</param>
    /// <param name="targetIndices">Array de índices dos tokens de alvo para a sequência.</param>
    /// <returns>O valor da perda (cross-entropy loss) calculado para a sequência.</returns>
    public double CalculateSequenceLoss(int[] inputIndices, int[] targetIndices)
    {
        // 1. Reseta o cache em disco antes de iniciar o forward pass.
        _cacheManager.Reset();

        // 2. Cria o tensor de alvos one-hot diretamente na engine de backend (GPU).
        // O 'using' garante que o tensor será liberado ('Dispose') ao final do escopo.
        using var targetsGpu = _mathEngine.CreateOneHotTensor(targetIndices, this.OutputSize);

        // 3. Executa apenas o forward pass para obter as predições e a perda.
        // O IMathTensor de predições é retornado e deve ser liberado.
        var (predictions, loss) = ForwardPassGpuOptimized(inputIndices, targetsGpu, inputIndices.Length);

        // 4. Libera a memória do tensor de predições, que não é mais necessário.
        predictions.Dispose();

        // 5. Retorna o valor da perda calculado.
        return loss;
    }

    /// <summary>
    /// Salva o estado completo do modelo, incluindo os pesos e o estado do otimizador Adam.
    /// Este método é a chave para a estratégia de checkpoint entre épocas.
    /// </summary>
    /// <param name="filePath">O caminho do arquivo onde o modelo será salvo como JSON.</param>
    public void SaveModel(string filePath)
    {
        // 1. Obtém o estado atual do otimizador. Isso envolve copiar os tensores m e v
        //    da GPU para a CPU para que possam ser serializados.
        var (optimizerM, optimizerV, optimizerT) = _adamOptimizer.GetState();

        // 2. Cria o objeto de dados que será serializado, populando-o com os pesos
        //    (também copiados da GPU para a CPU) e o estado do otimizador.
        var modelData = new NeuralNetworkModelDataEmbeddingLSTM
        {
            VocabSize = this.InputSize,
            EmbeddingSize = this.weightsEmbedding!.Shape[1],
            HiddenSize = this.HiddenSize,
            OutputSize = this.OutputSize,

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

            // Salva o estado do otimizador no mesmo arquivo JSON
            OptimizerM = optimizerM,
            OptimizerV = optimizerV,
            OptimizerT = optimizerT
        };
        
        // 3. Serializa o objeto de dados para uma string JSON e a escreve em disco.
        string jsonString = JsonSerializer.Serialize(modelData, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(filePath, jsonString);
    }

    /// <summary>
    /// Carrega um modelo a partir de um arquivo JSON, restaurando os pesos e o estado do otimizador.
    /// Este método é usado no início de cada época (exceto a primeira) para continuar o treinamento.
    /// </summary>
    /// <param name="filePath">O caminho do arquivo de checkpoint do modelo.</param>
    /// <param name="mathEngine">A engine de matemática (CPU ou GPU) a ser usada para criar os tensores.</param>
    /// <returns>Uma nova instância de NeuralNetworkLSTM com o estado restaurado, ou null se ocorrer um erro.</returns>
    public static NeuralNetworkLSTM? LoadModel(string filePath, IMathEngine mathEngine)
    {
        if (!File.Exists(filePath))
        {
            Console.WriteLine($"[LoadModel] Erro: Arquivo de checkpoint não encontrado em '{filePath}'.");
            return null;
        }
            
        try
        {
            // 1. Lê o arquivo JSON do disco e o desserializa para o objeto de dados.
            string jsonString = File.ReadAllText(filePath);
            var modelData = JsonSerializer.Deserialize<NeuralNetworkModelDataEmbeddingLSTM>(jsonString);
            if (modelData == null)
            {
                Console.WriteLine("[LoadModel] Erro: Falha ao desserializar o arquivo de modelo.");
                return null;
            }

            // 2. Recria cada tensor de peso, transferindo os dados para o dispositivo de computação (GPU/CPU).
            var wEmbed = mathEngine.CreateTensor(modelData.WeightsEmbedding!.data, modelData.WeightsEmbedding.shape);
            var wif = mathEngine.CreateTensor(modelData.WeightsInputForget!.data, modelData.WeightsInputForget.shape);
            var whf = mathEngine.CreateTensor(modelData.WeightsHiddenForget!.data, modelData.WeightsHiddenForget.shape);
            var wii = mathEngine.CreateTensor(modelData.WeightsInputInput!.data, modelData.WeightsInputInput.shape);
            var whi = mathEngine.CreateTensor(modelData.WeightsHiddenInput!.data, modelData.WeightsHiddenInput.shape);
            var wic = mathEngine.CreateTensor(modelData.WeightsInputCell!.data, modelData.WeightsInputCell.shape);
            var whc = mathEngine.CreateTensor(modelData.WeightsHiddenCell!.data, modelData.WeightsHiddenCell.shape);
            var wio = mathEngine.CreateTensor(modelData.WeightsInputOutput!.data, modelData.WeightsInputOutput.shape);
            var who = mathEngine.CreateTensor(modelData.WeightsHiddenOutput!.data, modelData.WeightsHiddenOutput.shape);
            var bf = mathEngine.CreateTensor(modelData.BiasForget!.data, modelData.BiasForget.shape);
            var bi = mathEngine.CreateTensor(modelData.BiasInput!.data, modelData.BiasInput.shape);
            var bc = mathEngine.CreateTensor(modelData.BiasCell!.data, modelData.BiasCell.shape);
            var bo = mathEngine.CreateTensor(modelData.BiasOutput!.data, modelData.BiasOutput.shape);
            var why = mathEngine.CreateTensor(modelData.WeightsHiddenOutputFinal!.data,
                modelData.WeightsHiddenOutputFinal.shape);
            var by = mathEngine.CreateTensor(modelData.BiasOutputFinal!.data, modelData.BiasOutputFinal.shape);

            // 3. Cria uma nova instância do modelo, passando todos os tensores já carregados.
            var model = new NeuralNetworkLSTM(
                modelData.VocabSize, modelData.EmbeddingSize, modelData.HiddenSize, modelData.OutputSize, mathEngine,
                wEmbed, wif, whf, wii, whi, wic, whc, wio, who, bf, bi, bc, bo, why, by
            );
            
            // 4. Restaura o estado do otimizador Adam, se presente no arquivo.
            if (modelData.OptimizerM != null && modelData.OptimizerV != null && modelData.OptimizerT != null)
            {
                model._adamOptimizer.SetState(modelData.OptimizerM, modelData.OptimizerV, modelData.OptimizerT, mathEngine);
                Console.WriteLine("[LoadModel] Estado do otimizador Adam restaurado com sucesso.");
            }
            else
            {
                Console.WriteLine("[LoadModel] Aviso: Estado do otimizador não encontrado no arquivo. O otimizador começará do zero.");
            }

            return model;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[LoadModel] Erro crítico ao carregar o modelo: {ex.Message}");
            return null;
        }
    }

    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing)
        {
            weightsEmbedding?.Dispose();
            weightsInputForget?.Dispose();
            weightsHiddenForget?.Dispose();
            weightsInputInput?.Dispose();
            weightsHiddenInput?.Dispose();
            weightsInputCell?.Dispose();
            weightsHiddenCell?.Dispose();
            weightsInputOutput?.Dispose();
            weightsHiddenOutput?.Dispose();
            biasForget?.Dispose();
            biasInput?.Dispose();
            biasCell?.Dispose();
            biasOutput?.Dispose();
            weightsHiddenOutputFinal?.Dispose();
            biasOutputFinal?.Dispose();
            hiddenState?.Dispose();
            cellState?.Dispose();
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