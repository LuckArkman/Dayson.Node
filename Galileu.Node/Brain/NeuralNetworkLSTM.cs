using System.Diagnostics;
using System.Text.Json;
using Galileu.Node.Core;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Brain
{
    /// <summary>
    /// Gerenciador centralizado de tensores com serialização total em disco.
    /// TODOS os tensores são mantidos em disco, apenas índices ficam em RAM.
    /// </summary>
    public class UnifiedDiskTensorManager : IDisposable
    {
        private readonly IMathEngine _mathEngine;
        private readonly string _tensorDirectory;
        private readonly Dictionary<string, TensorMetadata> _tensorIndex;
        private int _nextTensorId = 0;
        private bool _disposed = false;

        private class TensorMetadata
        {
            public string Id { get; set; } = string.Empty;
            public string FilePath { get; set; } = string.Empty;
            public int[] Shape { get; set; } = Array.Empty<int>();
            public DateTime CreatedAt { get; set; }
            public DateTime LastAccessed { get; set; }
            public long SizeBytes { get; set; }
            public int AccessCount { get; set; }
        }

        public UnifiedDiskTensorManager(IMathEngine mathEngine, string sessionId)
        {
            _mathEngine = mathEngine;
            _tensorDirectory = Path.Combine(Environment.CurrentDirectory, "Dayson", "TensorCache", sessionId);
            _tensorIndex = new Dictionary<string, TensorMetadata>();

            if (Directory.Exists(_tensorDirectory))
                Directory.Delete(_tensorDirectory, recursive: true);

            Directory.CreateDirectory(_tensorDirectory);
            Console.WriteLine($"[UnifiedDiskTensorManager] Sessão '{sessionId}' inicializada: {_tensorDirectory}");
        }

        /// <summary>
        /// Cria tensor, serializa imediatamente e retorna apenas ID.
        /// </summary>
        public string CreateAndStore(float[] data, int[] shape, string name)
        {
            using var tensor = _mathEngine.CreateTensor(data, shape);
            return StoreTensor(tensor, name);
        }

        /// <summary>
        /// Cria tensor zerado, serializa e retorna ID.
        /// </summary>
        public string CreateAndStoreZeros(int[] shape, string name)
        {
            using var tensor = _mathEngine.CreateTensor(shape);
            return StoreTensor(tensor, name);
        }

        /// <summary>
        /// Serializa tensor existente e retorna ID único.
        /// </summary>
        public string StoreTensor(IMathTensor tensor, string name)
        {
            string id = $"{name}_{_nextTensorId++:D6}_{Guid.NewGuid():N}";
            string filePath = Path.Combine(_tensorDirectory, $"{id}.tensor");

            using (var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None, 65536))
            using (var writer = new BinaryWriter(fs))
            {
                tensor.WriteToStream(writer);
                writer.Flush();
                fs.Flush(true);
            }

            var metadata = new TensorMetadata
            {
                Id = id,
                FilePath = filePath,
                Shape = (int[])tensor.Shape.Clone(),
                CreatedAt = DateTime.UtcNow,
                LastAccessed = DateTime.UtcNow,
                SizeBytes = new FileInfo(filePath).Length,
                AccessCount = 0
            };

            _tensorIndex[id] = metadata;
            return id;
        }

        /// <summary>
        /// Desserializa tensor do disco (ciclo: load).
        /// </summary>
        public IMathTensor LoadTensor(string id)
        {
            if (!_tensorIndex.TryGetValue(id, out var metadata))
                throw new KeyNotFoundException($"Tensor ID '{id}' não encontrado.");

            metadata.LastAccessed = DateTime.UtcNow;
            metadata.AccessCount++;

            using var fs = new FileStream(metadata.FilePath, FileMode.Open, FileAccess.Read, FileShare.Read, 65536);
            using var reader = new BinaryReader(fs);

            var tensor = _mathEngine.CreateTensor(metadata.Shape);

            // ================== CORREÇÃO APLICADA AQUI ==================
            long numElements = reader.ReadInt64(); // Lê o número de floats que foi escrito
            if (numElements != tensor.Length)
            {
                // Sanity check para garantir que os metadados e o arquivo estão em sincronia
                throw new InvalidDataException(
                    $"Inconsistência no tensor '{id}': metadados indicam {tensor.Length} elementos, mas o arquivo contém {numElements}.");
            }

            // Lê a quantidade correta de bytes (número de elementos * tamanho de um float)
            var dataBytes = reader.ReadBytes((int)numElements * sizeof(float));

            // Converte os bytes lidos de volta para um array de float
            var floatData = new float[numElements];
            Buffer.BlockCopy(dataBytes, 0, floatData, 0, dataBytes.Length);

            tensor.UpdateFromCpu(floatData);
            // =============================================================

            return tensor;
        }

        /// <summary>
        /// CICLO COMPLETO: Desserializa → Operação → Resserializa → Libera.
        /// </summary>
        public void UpdateTensor(string id, Action<IMathTensor> operation)
        {
            // Desserialização
            using var tensor = LoadTensor(id);

            // Operação
            operation(tensor);

            // Resserialização (sobrescreve arquivo)
            if (!_tensorIndex.TryGetValue(id, out var metadata))
                throw new KeyNotFoundException($"Tensor ID '{id}' não encontrado.");

            using (var fs = new FileStream(metadata.FilePath, FileMode.Create, FileAccess.Write, FileShare.None, 65536))
            using (var writer = new BinaryWriter(fs))
            {
                tensor.WriteToStream(writer);
                writer.Flush();
                fs.Flush(true);
            }

            metadata.LastAccessed = DateTime.UtcNow;
            metadata.SizeBytes = new FileInfo(metadata.FilePath).Length;

            // Liberação automática ao sair do 'using'
        }

        /// <summary>
        /// Operação binária: load(a) + load(b) → operation → store(result).
        /// </summary>
        public void BinaryOp(string idA, string idB, string idResult, Action<IMathTensor, IMathTensor, IMathTensor> op)
        {
            using var tensorA = LoadTensor(idA);
            using var tensorB = LoadTensor(idB);
            using var result = LoadTensor(idResult);

            op(tensorA, tensorB, result);

            // Resserializa apenas resultado
            UpdateTensor(idResult, _ => { });
        }

        /// <summary>
        /// Cria cópia de um tensor (clone).
        /// </summary>
        public string CloneTensor(string sourceId, string newName)
        {
            using var source = LoadTensor(sourceId);
            using var clone = _mathEngine.Clone(source);
            return StoreTensor(clone, newName);
        }

        /// <summary>
        /// Remove tensor do índice e disco.
        /// </summary>
        public void DeleteTensor(string id)
        {
            if (_tensorIndex.TryGetValue(id, out var metadata))
            {
                try
                {
                    if (File.Exists(metadata.FilePath))
                        File.Delete(metadata.FilePath);
                }
                catch
                {
                }

                _tensorIndex.Remove(id);
            }
        }

        /// <summary>
        /// Retorna shape de um tensor sem carregá-lo.
        /// </summary>
        public int[] GetShape(string id)
        {
            if (_tensorIndex.TryGetValue(id, out var metadata))
                return (int[])metadata.Shape.Clone();
            throw new KeyNotFoundException($"Tensor ID '{id}' não encontrado.");
        }

        /// <summary>
        /// Estatísticas do gerenciador.
        /// </summary>
        public (int Count, long DiskMB, long RamMB, int TotalAccesses) GetStatistics()
        {
            long totalBytes = _tensorIndex.Values.Sum(m => m.SizeBytes);
            long ramUsage = _tensorIndex.Count * 256; // ~256 bytes por metadata
            int totalAccesses = _tensorIndex.Values.Sum(m => m.AccessCount);

            return (_tensorIndex.Count, totalBytes / (1024 * 1024), ramUsage / (1024 * 1024), totalAccesses);
        }

        /// <summary>
        /// Lista tensores com filtro opcional.
        /// </summary>
        public List<string> ListTensors(string? nameFilter = null)
        {
            var query = _tensorIndex.Keys.AsEnumerable();

            if (!string.IsNullOrEmpty(nameFilter))
                query = query.Where(id => id.Contains(nameFilter));

            return query.ToList();
        }

        /// <summary>
        /// Remove todos os tensores (mantém estrutura de diretório).
        /// </summary>
        public void Clear()
        {
            Console.WriteLine($"[UnifiedDiskTensorManager] Limpando {_tensorIndex.Count} tensores...");

            foreach (var metadata in _tensorIndex.Values)
            {
                try
                {
                    if (File.Exists(metadata.FilePath))
                        File.Delete(metadata.FilePath);
                }
                catch
                {
                }
            }

            _tensorIndex.Clear();
            _nextTensorId = 0;
        }

        public void Dispose()
        {
            if (_disposed) return;

            var stats = GetStatistics();
            Console.WriteLine(
                $"[UnifiedDiskTensorManager] Dispose: {stats.Count} tensores, {stats.DiskMB}MB em disco, {stats.TotalAccesses} acessos totais");

            Clear();

            try
            {
                if (Directory.Exists(_tensorDirectory))
                    Directory.Delete(_tensorDirectory, recursive: true);
            }
            catch
            {
            }

            _disposed = true;
        }
    }

    /// <summary>
    /// NeuralNetworkLSTM refatorado com arquitetura 100% disk-backed.
    /// Zero tensores em RAM permanentemente - apenas índices de referência.
    /// </summary>
    public class NeuralNetworkLSTM : IDisposable
    {
        protected readonly AdamOptimizer _adamOptimizer;
        protected readonly UnifiedDiskTensorManager _tensorManager;
        protected readonly IMathEngine _mathEngine;
        private readonly int inputSize;
        private readonly int hiddenSize;
        private readonly int outputSize;
        private readonly string _sessionId;
        private bool _disposed = false;

        // ═══════════════════════════════════════════════════════════
        // APENAS ÍNDICES EM RAM (não tensores!)
        // ═══════════════════════════════════════════════════════════
        protected string _weightsEmbeddingId = null!;
        protected string _weightsInputForgetId = null!;
        protected string _weightsHiddenForgetId = null!;
        protected string _weightsInputInputId = null!;
        protected string _weightsHiddenInputId = null!;
        protected string _weightsInputCellId = null!;
        protected string _weightsHiddenCellId = null!;
        protected string _weightsInputOutputId = null!;
        protected string _weightsHiddenOutputId = null!;
        protected string _biasForgetId = null!;
        protected string _biasInputId = null!;
        protected string _biasCellId = null!;
        protected string _biasOutputId = null!;
        protected string _weightsHiddenOutputFinalId = null!;
        protected string _biasOutputFinalId = null!;
        protected string _hiddenStateId = null!;
        protected string _cellStateId = null!;

        // Cache de tensores temporários do forward pass (também apenas IDs)
        private readonly Dictionary<int, Dictionary<string, string>> _forwardCache = new();

        public int InputSize => inputSize;
        public int HiddenSize => hiddenSize;
        public int OutputSize => outputSize;
        public IMathEngine GetMathEngine() => _mathEngine;

        public NeuralNetworkLSTM(int vocabSize, int embeddingSize, int hiddenSize, int outputSize,
            IMathEngine mathEngine)
        {
            this.inputSize = vocabSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;
            this._mathEngine = mathEngine ?? throw new ArgumentNullException(nameof(mathEngine));
            this._adamOptimizer = new AdamOptimizer();
            this._sessionId = $"session_{DateTime.UtcNow:yyyyMMdd_HHmmss}_{Guid.NewGuid():N}";
            this._tensorManager = new UnifiedDiskTensorManager(mathEngine, _sessionId);

            Console.WriteLine("\n╔════════════════════════════════════════════════════════╗");
            Console.WriteLine("║   INICIALIZANDO MODELO DISK-BACKED (Zero RAM)         ║");
            Console.WriteLine("╚════════════════════════════════════════════════════════╝");

            var rand = new Random(42);

            _weightsEmbeddingId = InitializeWeight(vocabSize, embeddingSize, rand, "WeightsEmbedding");
            _weightsInputForgetId = InitializeWeight(embeddingSize, hiddenSize, rand, "WeightsInputForget");
            _weightsHiddenForgetId = InitializeWeight(hiddenSize, hiddenSize, rand, "WeightsHiddenForget");
            _weightsInputInputId = InitializeWeight(embeddingSize, hiddenSize, rand, "WeightsInputInput");
            _weightsHiddenInputId = InitializeWeight(hiddenSize, hiddenSize, rand, "WeightsHiddenInput");
            _weightsInputCellId = InitializeWeight(embeddingSize, hiddenSize, rand, "WeightsInputCell");
            _weightsHiddenCellId = InitializeWeight(hiddenSize, hiddenSize, rand, "WeightsHiddenCell");
            _weightsInputOutputId = InitializeWeight(embeddingSize, hiddenSize, rand, "WeightsInputOutput");
            _weightsHiddenOutputId = InitializeWeight(hiddenSize, hiddenSize, rand, "WeightsHiddenOutput");
            _biasForgetId = InitializeWeight(1, hiddenSize, rand, "BiasForget");
            _biasInputId = InitializeWeight(1, hiddenSize, rand, "BiasInput");
            _biasCellId = InitializeWeight(1, hiddenSize, rand, "BiasCell");
            _biasOutputId = InitializeWeight(1, hiddenSize, rand, "BiasOutput");
            _weightsHiddenOutputFinalId = InitializeWeight(hiddenSize, outputSize, rand, "WeightsOutputFinal");
            _biasOutputFinalId = InitializeWeight(1, outputSize, rand, "BiasOutputFinal");

            _hiddenStateId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, "HiddenState");
            _cellStateId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, "CellState");

            var stats = _tensorManager.GetStatistics();
            Console.WriteLine($"[NeuralNetworkLSTM] Inicialização completa:");
            Console.WriteLine($"  ├─ Tensores em disco: {stats.Count}");
            Console.WriteLine($"  ├─ Espaço em disco: {stats.DiskMB} MB");
            Console.WriteLine($"  ├─ RAM usada (índices): {stats.RamMB} MB");
            Console.WriteLine($"  └─ Redução de RAM: ~{(stats.DiskMB * 100.0 / Math.Max(stats.RamMB, 1)):F1}x\n");
        }

        // NOVO E CORRIGIDO: Construtor protegido para "envolver" um modelo já carregado.
        protected NeuralNetworkLSTM(NeuralNetworkLSTM existingModel)
        {
            this.inputSize = existingModel.inputSize;
            this.hiddenSize = existingModel.hiddenSize;
            this.outputSize = existingModel.outputSize;
            this._mathEngine = existingModel._mathEngine;
            this._adamOptimizer = existingModel._adamOptimizer;
            this._sessionId = existingModel._sessionId;
            this._tensorManager = existingModel._tensorManager;

            this._weightsEmbeddingId = existingModel._weightsEmbeddingId;
            this._weightsInputForgetId = existingModel._weightsInputForgetId;
            this._weightsHiddenForgetId = existingModel._weightsHiddenForgetId;
            this._weightsInputInputId = existingModel._weightsInputInputId;
            this._weightsHiddenInputId = existingModel._weightsHiddenInputId;
            this._weightsInputCellId = existingModel._weightsInputCellId;
            this._weightsHiddenCellId = existingModel._weightsHiddenCellId;
            this._weightsInputOutputId = existingModel._weightsInputOutputId;
            this._weightsHiddenOutputId = existingModel._weightsHiddenOutputId;
            this._biasForgetId = existingModel._biasForgetId;
            this._biasInputId = existingModel._biasInputId;
            this._biasCellId = existingModel._biasCellId;
            this._biasOutputId = existingModel._biasOutputId;
            this._weightsHiddenOutputFinalId = existingModel._weightsHiddenOutputFinalId;
            this._biasOutputFinalId = existingModel._biasOutputFinalId;
            this._hiddenStateId = existingModel._hiddenStateId;
            this._cellStateId = existingModel._cellStateId;
        }

        /// <summary>
        /// Inicializa tensor com Xavier, serializa e retorna apenas ID.
        /// </summary>
        private string InitializeWeight(int rows, int cols, Random rand, string name)
        {
            var data = new float[rows * cols];
            double limit = Math.Sqrt(6.0 / (rows + cols));

            // Reduz limite para matrizes muito grandes
            if (rows > 10000 || cols > 10000)
                limit *= 0.01;
            else if (rows > 1000 || cols > 1000)
                limit *= 0.1;

            for (int i = 0; i < data.Length; i++)
                data[i] = (float)((rand.NextDouble() * 2 - 1) * limit);

            // Valida antes de criar
            if (data.Any(float.IsNaN) || data.Any(float.IsInfinity))
                throw new InvalidOperationException($"[{name}] Inicialização gerou NaN/Inf");

            return _tensorManager.CreateAndStore(data, new[] { rows, cols }, name);
        }

        /// <summary>
        /// Forward pass com desserialização sob demanda.
        /// ZERO tensores permanecem em RAM após execução.
        /// </summary>
        public Tensor Forward(Tensor embeddedInput)
        {
            Console.WriteLine("[Forward] Iniciando (disk-backed)...");
            var sw = Stopwatch.StartNew();

            // Cria input temporário
            var inputData = embeddedInput.GetData();
            string inputId = _tensorManager.CreateAndStore(inputData, new[] { 1, inputData.Length }, "TempInput");

            // Carrega estados
            using var hiddenState = _tensorManager.LoadTensor(_hiddenStateId);
            using var cellState = _tensorManager.LoadTensor(_cellStateId);

            // ═══ FORGET GATE ═══
            using var weightsInputForget = _tensorManager.LoadTensor(_weightsInputForgetId);
            using var weightsHiddenForget = _tensorManager.LoadTensor(_weightsHiddenForgetId);
            using var biasForget = _tensorManager.LoadTensor(_biasForgetId);
            using var input = _tensorManager.LoadTensor(inputId);

            using var fg_term1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
            using var fg_term2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
            using var forgetGateLinear = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
            using var forgetGate = _mathEngine.CreateTensor(new[] { 1, hiddenSize });

            _mathEngine.MatrixMultiply(input, weightsInputForget, fg_term1);
            _mathEngine.MatrixMultiply(hiddenState, weightsHiddenForget, fg_term2);
            _mathEngine.Add(fg_term1, fg_term2, forgetGateLinear);
            _mathEngine.AddBroadcast(forgetGateLinear, biasForget, forgetGateLinear);
            _mathEngine.Sigmoid(forgetGateLinear, forgetGate);

            // ═══ INPUT GATE ═══
            using var weightsInputInput = _tensorManager.LoadTensor(_weightsInputInputId);
            using var weightsHiddenInput = _tensorManager.LoadTensor(_weightsHiddenInputId);
            using var biasInput = _tensorManager.LoadTensor(_biasInputId);

            using var ig_term1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
            using var ig_term2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
            using var inputGateLinear = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
            using var inputGate = _mathEngine.CreateTensor(new[] { 1, hiddenSize });

            _mathEngine.MatrixMultiply(input, weightsInputInput, ig_term1);
            _mathEngine.MatrixMultiply(hiddenState, weightsHiddenInput, ig_term2);
            _mathEngine.Add(ig_term1, ig_term2, inputGateLinear);
            _mathEngine.AddBroadcast(inputGateLinear, biasInput, inputGateLinear);
            _mathEngine.Sigmoid(inputGateLinear, inputGate);

            // ═══ CELL CANDIDATE ═══
            using var weightsInputCell = _tensorManager.LoadTensor(_weightsInputCellId);
            using var weightsHiddenCell = _tensorManager.LoadTensor(_weightsHiddenCellId);
            using var biasCell = _tensorManager.LoadTensor(_biasCellId);

            using var cc_term1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
            using var cc_term2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
            using var cellCandidateLinear = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
            using var cellCandidate = _mathEngine.CreateTensor(new[] { 1, hiddenSize });

            _mathEngine.MatrixMultiply(input, weightsInputCell, cc_term1);
            _mathEngine.MatrixMultiply(hiddenState, weightsHiddenCell, cc_term2);
            _mathEngine.Add(cc_term1, cc_term2, cellCandidateLinear);
            _mathEngine.AddBroadcast(cellCandidateLinear, biasCell, cellCandidateLinear);
            _mathEngine.Tanh(cellCandidateLinear, cellCandidate);

            // ═══ CELL STATE UPDATE ═══
            using var nextCellState_term1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
            using var nextCellState_term2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
            using var nextCellState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });

            _mathEngine.Multiply(forgetGate, cellState, nextCellState_term1);
            _mathEngine.Multiply(inputGate, cellCandidate, nextCellState_term2);
            _mathEngine.Add(nextCellState_term1, nextCellState_term2, nextCellState);

            // Resserializa cell state
            _tensorManager.UpdateTensor(_cellStateId, tensor =>
                tensor.UpdateFromCpu(nextCellState.ToCpuTensor().GetData()));

            // ═══ OUTPUT GATE ═══
            using var weightsInputOutput = _tensorManager.LoadTensor(_weightsInputOutputId);
            using var weightsHiddenOutput = _tensorManager.LoadTensor(_weightsHiddenOutputId);
            using var biasOutput = _tensorManager.LoadTensor(_biasOutputId);

            using var og_term1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
            using var og_term2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
            using var outputGateLinear = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
            using var outputGate = _mathEngine.CreateTensor(new[] { 1, hiddenSize });

            _mathEngine.MatrixMultiply(input, weightsInputOutput, og_term1);
            _mathEngine.MatrixMultiply(hiddenState, weightsHiddenOutput, og_term2);
            _mathEngine.Add(og_term1, og_term2, outputGateLinear);
            _mathEngine.AddBroadcast(outputGateLinear, biasOutput, outputGateLinear);
            _mathEngine.Sigmoid(outputGateLinear, outputGate);

            // ═══ HIDDEN STATE UPDATE ═══
            using var tanhCellState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
            using var nextHiddenState = _mathEngine.CreateTensor(new[] { 1, hiddenSize });

            _mathEngine.Tanh(nextCellState, tanhCellState);
            _mathEngine.Multiply(outputGate, tanhCellState, nextHiddenState);

            // Resserializa hidden state
            _tensorManager.UpdateTensor(_hiddenStateId, tensor =>
                tensor.UpdateFromCpu(nextHiddenState.ToCpuTensor().GetData()));

            // ═══ OUTPUT LAYER ═══
            using var weightsOutputFinal = _tensorManager.LoadTensor(_weightsHiddenOutputFinalId);
            using var biasOutputFinal = _tensorManager.LoadTensor(_biasOutputFinalId);
            using var finalOutputLinear = _mathEngine.CreateTensor(new[] { 1, outputSize });

            _mathEngine.MatrixMultiply(nextHiddenState, weightsOutputFinal, finalOutputLinear);
            _mathEngine.AddBroadcast(finalOutputLinear, biasOutputFinal, finalOutputLinear);

            // Softmax em CPU
            var finalOutputCpu = finalOutputLinear.ToCpuTensor();
            var softmaxResult = Softmax(finalOutputCpu.GetData());

            // Limpa input temporário
            _tensorManager.DeleteTensor(inputId);

            sw.Stop();
            Console.WriteLine(
                $"[Forward] Concluído em {sw.ElapsedMilliseconds}ms. RAM: {_tensorManager.GetStatistics().RamMB}MB");

            return new Tensor(softmaxResult, new[] { outputSize });
        }

        private float[] Softmax(float[] logits)
        {
            if (logits == null || logits.Length == 0) return Array.Empty<float>();
            var output = new float[logits.Length];
            float maxLogit = logits.Max();

            if (float.IsNaN(maxLogit) || float.IsInfinity(maxLogit))
            {
                Array.Fill(output, 1.0f / logits.Length);
                return output;
            }

            const float MAX_LOGIT = 88.0f;
            maxLogit = Math.Min(maxLogit, MAX_LOGIT);
            float sumExp = 0.0f;

            for (int i = 0; i < logits.Length; i++)
            {
                float clippedLogit = Math.Min(logits[i], MAX_LOGIT) - maxLogit;
                output[i] = MathF.Exp(clippedLogit);
                sumExp += output[i];
            }

            if (sumExp < 1e-20f)
            {
                Array.Fill(output, 1.0f / logits.Length);
                return output;
            }

            for (int i = 0; i < logits.Length; i++)
            {
                output[i] /= sumExp;
                if (float.IsNaN(output[i]) || float.IsInfinity(output[i]))
                    output[i] = 1e-10f;
            }

            return output;
        }

        public void ResetHiddenState()
        {
            var zeros = new float[hiddenSize];
            _tensorManager.UpdateTensor(_hiddenStateId, t => t.UpdateFromCpu(zeros));
            _tensorManager.UpdateTensor(_cellStateId, t => t.UpdateFromCpu(zeros));
            Console.WriteLine("[ResetHiddenState] Estados zerados e resserializados.");
        }

        /// <summary>
        /// Sanity check: valida operações básicas.
        /// </summary>
        public void RunSanityCheck()
        {
            Console.WriteLine("\n[SanityCheck] Validando modelo disk-backed...");

            try
            {
                // Testa lookup
                using var embeddingMatrix = _tensorManager.LoadTensor(_weightsEmbeddingId);
                int embeddingSize = _tensorManager.GetShape(_weightsEmbeddingId)[1];
                using var testEmbedding = _mathEngine.CreateTensor(new[] { 1, embeddingSize });
                _mathEngine.Lookup(embeddingMatrix, 0, testEmbedding);
                var embData = testEmbedding.ToCpuTensor().GetData();
                if (embData.Any(float.IsNaN) || embData.Any(float.IsInfinity))
                    throw new Exception("Lookup produziu NaN/Inf");

                // Testa forward completo
                var dummyEmbedding = new float[embeddingSize];
                for (int i = 0; i < embeddingSize; i++) dummyEmbedding[i] = 0.01f * (i % 10);
                var dummyInput = new Tensor(dummyEmbedding, new[] { embeddingSize });
                var output = Forward(dummyInput);
                if (output.GetData().Any(float.IsNaN) || output.GetData().Any(float.IsInfinity))
                    throw new Exception("Forward produziu NaN/Inf");

                Console.WriteLine("[SanityCheck] ✓ Modelo validado com sucesso.");
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"[SanityCheck] Falha: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Salva metadados do modelo (índices dos tensores).
        /// Os arquivos .tensor permanecem em disco.
        /// </summary>
        public void SaveModel(string filePath)
        {
            var embeddingSize = _tensorManager.GetShape(_weightsEmbeddingId)[1];

            var modelData = new
            {
                VocabSize = inputSize,
                EmbeddingSize = embeddingSize,
                HiddenSize = hiddenSize,
                OutputSize = outputSize,
                SessionId = _sessionId,
                TensorDirectory = _tensorManager.GetStatistics().Count > 0
                    ? Path.GetDirectoryName(_tensorManager.ListTensors().First())
                    : "",
                TensorIds = new Dictionary<string, string>
                {
                    ["WeightsEmbedding"] = _weightsEmbeddingId,
                    ["WeightsInputForget"] = _weightsInputForgetId,
                    ["WeightsHiddenForget"] = _weightsHiddenForgetId,
                    ["WeightsInputInput"] = _weightsInputInputId,
                    ["WeightsHiddenInput"] = _weightsHiddenInputId,
                    ["WeightsInputCell"] = _weightsInputCellId,
                    ["WeightsHiddenCell"] = _weightsHiddenCellId,
                    ["WeightsInputOutput"] = _weightsInputOutputId,
                    ["WeightsHiddenOutput"] = _weightsHiddenOutputId,
                    ["BiasForget"] = _biasForgetId,
                    ["BiasInput"] = _biasInputId,
                    ["BiasCell"] = _biasCellId,
                    ["BiasOutput"] = _biasOutputId,
                    ["WeightsOutputFinal"] = _weightsHiddenOutputFinalId,
                    ["BiasOutputFinal"] = _biasOutputFinalId,
                    ["HiddenState"] = _hiddenStateId,
                    ["CellState"] = _cellStateId
                }
            };

            string jsonString = JsonSerializer.Serialize(modelData, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(filePath, jsonString);

            var stats = _tensorManager.GetStatistics();
            Console.WriteLine($"[SaveModel] Modelo salvo em {filePath}");
            Console.WriteLine($"  ├─ {stats.Count} tensores permanecem em disco");
            Console.WriteLine($"  └─ Espaço total: {stats.DiskMB} MB");
        }

        /// <summary>
        /// Carrega modelo: reconstrói índices a partir de metadados salvos.
        /// Os tensores já estão em disco, apenas recria referências.
        /// </summary>
        public static NeuralNetworkLSTM? LoadModel(string filePath, IMathEngine mathEngine)
        {
            if (!File.Exists(filePath)) return null;

            try
            {
                Console.WriteLine($"[LoadModel] Carregando de {filePath}...");

                string jsonString = File.ReadAllText(filePath);
                using var doc = JsonDocument.Parse(jsonString);
                var root = doc.RootElement;

                int vocabSize = root.GetProperty("VocabSize").GetInt32();
                int embeddingSize = root.GetProperty("EmbeddingSize").GetInt32();
                int hiddenSize = root.GetProperty("HiddenSize").GetInt32();
                int outputSize = root.GetProperty("OutputSize").GetInt32();
                string sessionId = root.GetProperty("SessionId").GetString() ?? Guid.NewGuid().ToString();

                // Cria modelo vazio
                var model = new NeuralNetworkLSTM(vocabSize, embeddingSize, hiddenSize, outputSize, mathEngine);

                // Substitui IDs pelos salvos (tensores já estão em disco)
                var tensorIds = root.GetProperty("TensorIds");
                model._weightsEmbeddingId = tensorIds.GetProperty("WeightsEmbedding").GetString()!;
                model._weightsInputForgetId = tensorIds.GetProperty("WeightsInputForget").GetString()!;
                model._weightsHiddenForgetId = tensorIds.GetProperty("WeightsHiddenForget").GetString()!;
                model._weightsInputInputId = tensorIds.GetProperty("WeightsInputInput").GetString()!;
                model._weightsHiddenInputId = tensorIds.GetProperty("WeightsHiddenInput").GetString()!;
                model._weightsInputCellId = tensorIds.GetProperty("WeightsInputCell").GetString()!;
                model._weightsHiddenCellId = tensorIds.GetProperty("WeightsHiddenCell").GetString()!;
                model._weightsInputOutputId = tensorIds.GetProperty("WeightsInputOutput").GetString()!;
                model._weightsHiddenOutputId = tensorIds.GetProperty("WeightsHiddenOutput").GetString()!;
                model._biasForgetId = tensorIds.GetProperty("BiasForget").GetString()!;
                model._biasInputId = tensorIds.GetProperty("BiasInput").GetString()!;
                model._biasCellId = tensorIds.GetProperty("BiasCell").GetString()!;
                model._biasOutputId = tensorIds.GetProperty("BiasOutput").GetString()!;
                model._weightsHiddenOutputFinalId = tensorIds.GetProperty("WeightsOutputFinal").GetString()!;
                model._biasOutputFinalId = tensorIds.GetProperty("BiasOutputFinal").GetString()!;
                model._hiddenStateId = tensorIds.GetProperty("HiddenState").GetString()!;
                model._cellStateId = tensorIds.GetProperty("CellState").GetString()!;

                Console.WriteLine("[LoadModel] ✓ Modelo carregado com sucesso.");
                return model;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[LoadModel] Erro: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Forward pass otimizado para treinamento com caching.
        /// Salva estados intermediários para backward pass.
        /// </summary>
        /// <summary>
        /// Forward pass otimizado para treinamento com caching.
        /// Salva estados intermediários para backward pass.
        /// CORREÇÃO: Não deleta tensores do último timestep.
        /// </summary>
        public (string predictionsId, float loss) ForwardPassGpuOptimized(int[] inputIndices, int[] targetIndices, int sequenceLength)
        {
            //Console.WriteLine($"[ForwardOptimized] Processando sequência de {sequenceLength} timesteps...");
            var sw = Stopwatch.StartNew();

            var targetsData = new float[sequenceLength * outputSize];
            for (int i = 0; i < sequenceLength; i++)
                targetsData[i * outputSize + targetIndices[i]] = 1.0f;

            string targetsId =
                _tensorManager.CreateAndStore(targetsData, new[] { sequenceLength, outputSize }, "Targets");
            string predictionsId = _tensorManager.CreateAndStore(new float[sequenceLength * outputSize],
                new[] { sequenceLength, outputSize }, "Predictions");

            string h_prevId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, "H_Prev");
            string c_prevId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, "C_Prev");

            int embeddingSize = _tensorManager.GetShape(_weightsEmbeddingId)[1];
            float sequenceLoss = 0;

            for (int t = 0; t < sequenceLength; t++)
            {
                var stepCache = new Dictionary<string, string>();

                using var weightsEmb = _tensorManager.LoadTensor(_weightsEmbeddingId);
                using var inputEmbedding = _mathEngine.CreateTensor(new[] { 1, embeddingSize });
                _mathEngine.Lookup(weightsEmb, inputIndices[t], inputEmbedding);
                string inputId = _tensorManager.StoreTensor(inputEmbedding, $"Input_t{t}");
                stepCache["Input"] = inputId;
                stepCache["HiddenPrev"] = _tensorManager.CloneTensor(h_prevId, $"HiddenPrev_t{t}");
                stepCache["CellPrev"] = _tensorManager.CloneTensor(c_prevId, $"CellPrev_t{t}");

                using var wif = _tensorManager.LoadTensor(_weightsInputForgetId);
                using var whf = _tensorManager.LoadTensor(_weightsHiddenForgetId);
                using var bf = _tensorManager.LoadTensor(_biasForgetId);
                using var inp = _tensorManager.LoadTensor(inputId);
                using var h_prev = _tensorManager.LoadTensor(h_prevId);
                using var fg1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                using var fg2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                using var fgLinear = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                using var forgetGate = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                _mathEngine.MatrixMultiply(inp, wif, fg1);
                _mathEngine.MatrixMultiply(h_prev, whf, fg2);
                _mathEngine.Add(fg1, fg2, fgLinear);
                _mathEngine.AddBroadcast(fgLinear, bf, fgLinear);
                _mathEngine.Sigmoid(fgLinear, forgetGate);
                stepCache["ForgetGate"] = _tensorManager.StoreTensor(forgetGate, $"ForgetGate_t{t}");

                using var wii = _tensorManager.LoadTensor(_weightsInputInputId);
                using var whi = _tensorManager.LoadTensor(_weightsHiddenInputId);
                using var bi = _tensorManager.LoadTensor(_biasInputId);
                using var ig1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                using var ig2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                using var igLinear = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                using var inputGate = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                _mathEngine.MatrixMultiply(inp, wii, ig1);
                _mathEngine.MatrixMultiply(h_prev, whi, ig2);
                _mathEngine.Add(ig1, ig2, igLinear);
                _mathEngine.AddBroadcast(igLinear, bi, igLinear);
                _mathEngine.Sigmoid(igLinear, inputGate);
                stepCache["InputGate"] = _tensorManager.StoreTensor(inputGate, $"InputGate_t{t}");

                using var wic = _tensorManager.LoadTensor(_weightsInputCellId);
                using var whc = _tensorManager.LoadTensor(_weightsHiddenCellId);
                using var bc = _tensorManager.LoadTensor(_biasCellId);
                using var cc1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                using var cc2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                using var ccLinear = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                using var cellCandidate = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                _mathEngine.MatrixMultiply(inp, wic, cc1);
                _mathEngine.MatrixMultiply(h_prev, whc, cc2);
                _mathEngine.Add(cc1, cc2, ccLinear);
                _mathEngine.AddBroadcast(ccLinear, bc, ccLinear);
                _mathEngine.Tanh(ccLinear, cellCandidate);
                stepCache["CellCandidate"] = _tensorManager.StoreTensor(cellCandidate, $"CellCandidate_t{t}");

                using var c_prev = _tensorManager.LoadTensor(c_prevId);
                using var cs1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                using var cs2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                using var cellNext = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                _mathEngine.Multiply(forgetGate, c_prev, cs1);
                _mathEngine.Multiply(inputGate, cellCandidate, cs2);
                _mathEngine.Add(cs1, cs2, cellNext);
                string cellNextId = _tensorManager.StoreTensor(cellNext, $"CellNext_t{t}");
                stepCache["CellNext"] = cellNextId;

                using var wio = _tensorManager.LoadTensor(_weightsInputOutputId);
                using var who = _tensorManager.LoadTensor(_weightsHiddenOutputId);
                using var bo = _tensorManager.LoadTensor(_biasOutputId);
                using var og1 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                using var og2 = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                using var ogLinear = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                using var outputGate = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                _mathEngine.MatrixMultiply(inp, wio, og1);
                _mathEngine.MatrixMultiply(h_prev, who, og2);
                _mathEngine.Add(og1, og2, ogLinear);
                _mathEngine.AddBroadcast(ogLinear, bo, ogLinear);
                _mathEngine.Sigmoid(ogLinear, outputGate);
                stepCache["OutputGate"] = _tensorManager.StoreTensor(outputGate, $"OutputGate_t{t}");

                using var tanhCell = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                using var hiddenNext = _mathEngine.CreateTensor(new[] { 1, hiddenSize });
                _mathEngine.Tanh(cellNext, tanhCell);
                stepCache["TanhCellNext"] = _tensorManager.StoreTensor(tanhCell, $"TanhCellNext_t{t}");
                _mathEngine.Multiply(outputGate, tanhCell, hiddenNext);
                string hiddenNextId = _tensorManager.StoreTensor(hiddenNext, $"HiddenNext_t{t}");
                stepCache["HiddenNext"] = hiddenNextId;

                using var why = _tensorManager.LoadTensor(_weightsHiddenOutputFinalId);
                using var by = _tensorManager.LoadTensor(_biasOutputFinalId);
                using var outLinear = _mathEngine.CreateTensor(new[] { 1, outputSize });
                using var outSoftmax = _mathEngine.CreateTensor(new[] { 1, outputSize });
                _mathEngine.MatrixMultiply(hiddenNext, why, outLinear);
                _mathEngine.AddBroadcast(outLinear, by, outLinear);
                _mathEngine.Softmax(outLinear, outSoftmax);

                var outData = outSoftmax.ToCpuTensor().GetData();
                _tensorManager.UpdateTensor(predictionsId, pred =>
                {
                    var predData = pred.ToCpuTensor().GetData();
                    Array.Copy(outData, 0, predData, t * outputSize, outputSize);
                    pred.UpdateFromCpu(predData);
                });

                float prob = outData[targetIndices[t]];
                sequenceLoss += -MathF.Log(Math.Max(prob, 1e-10f));

                // A deleção de h_prevId e c_prevId foi removida daqui.
                h_prevId = hiddenNextId;
                c_prevId = cellNextId;
                _forwardCache[t] = stepCache;
            }
            
            // A deleção de targetsId, h_prevId e c_prevId também foi removida.
            
            sw.Stop();
            var stats = _tensorManager.GetStatistics();
            Console.WriteLine($"[ForwardOptimized] Concluído em {sw.ElapsedMilliseconds}ms");
            Console.WriteLine($"  ├─ Loss: {sequenceLoss / sequenceLength:F4}");
            Console.WriteLine($"  ├─ Tensores em disco: {stats.Count}");
            Console.WriteLine($"  └─ RAM: {stats.RamMB}MB");

            return (predictionsId, sequenceLoss / sequenceLength);
        }

        /// <summary>
        /// Backward pass otimizado: BPTT completo com gradientes em disco.
        /// ATUALIZADO: Não apaga nenhum tensor, delegando a limpeza para o final da época.
        /// </summary>
        /// <summary>
        /// Backward pass otimizado: BPTT completo com gradientes em disco.
        /// ATUALIZADO: Não apaga nenhum tensor, delegando a limpeza para o final da época.
        /// CORRIGIDO: Seção de Gradient Clipping restaurada para definir 'totalNorm'.
        /// </summary>
        public Dictionary<string, string> BackwardPassGpuOptimized(string predictionsId, int[] targetIndices,
            int[] inputIndices, int sequenceLength)
        {
            //Console.WriteLine($"[BackwardOptimized] Iniciando BPTT para {sequenceLength} timesteps...");
            var sw = Stopwatch.StartNew();

            // Inicializa gradientes zerados
            int embeddingSize = _tensorManager.GetShape(_weightsEmbeddingId)[1];
            var grads = new Dictionary<string, string>
            {
                ["w_embed"] = _tensorManager.CreateAndStoreZeros(new[] { inputSize, embeddingSize }, "Grad_WEmbed"),
                ["wif"] = _tensorManager.CreateAndStoreZeros(new[] { embeddingSize, hiddenSize }, "Grad_WIF"),
                ["whf"] = _tensorManager.CreateAndStoreZeros(new[] { hiddenSize, hiddenSize }, "Grad_WHF"),
                ["wii"] = _tensorManager.CreateAndStoreZeros(new[] { embeddingSize, hiddenSize }, "Grad_WII"),
                ["whi"] = _tensorManager.CreateAndStoreZeros(new[] { hiddenSize, hiddenSize }, "Grad_WHI"),
                ["wic"] = _tensorManager.CreateAndStoreZeros(new[] { embeddingSize, hiddenSize }, "Grad_WIC"),
                ["whc"] = _tensorManager.CreateAndStoreZeros(new[] { hiddenSize, hiddenSize }, "Grad_WHC"),
                ["wio"] = _tensorManager.CreateAndStoreZeros(new[] { embeddingSize, hiddenSize }, "Grad_WIO"),
                ["who"] = _tensorManager.CreateAndStoreZeros(new[] { hiddenSize, hiddenSize }, "Grad_WHO"),
                ["why"] = _tensorManager.CreateAndStoreZeros(new[] { hiddenSize, outputSize }, "Grad_WHY"),
                ["bf"] = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, "Grad_BF"),
                ["bi"] = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, "Grad_BI"),
                ["bc"] = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, "Grad_BC"),
                ["bo"] = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, "Grad_BO"),
                ["by"] = _tensorManager.CreateAndStoreZeros(new[] { 1, outputSize }, "Grad_BY")
            };

            // Calcula dy = predictions - targets
            var targetsData = new float[sequenceLength * outputSize];
            for (int i = 0; i < sequenceLength; i++)
                targetsData[i * outputSize + targetIndices[i]] = 1.0f;

            string targetsId =
                _tensorManager.CreateAndStore(targetsData, new[] { sequenceLength, outputSize }, "TempTargets");
            string dyId = _tensorManager.CreateAndStoreZeros(new[] { sequenceLength, outputSize }, "DY");

            using (var predictions = _tensorManager.LoadTensor(predictionsId))
            using (var targets = _tensorManager.LoadTensor(targetsId))
            using (var dy = _tensorManager.LoadTensor(dyId))
            {
                _mathEngine.Subtract(predictions, targets, dy);
                _tensorManager.UpdateTensor(dyId, t => t.UpdateFromCpu(dy.ToCpuTensor().GetData()));
            }

            // Gradientes acumulados através do tempo
            string dh_nextId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, "DH_Next");
            string dc_nextId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, "DC_Next");

            // BPTT: loop reverso
            for (int t = sequenceLength - 1; t >= 0; t--)
            {
                if (!_forwardCache.TryGetValue(t, out var cache))
                    throw new Exception($"Cache não encontrado para timestep {t}");

                // ═══ EXTRAI dy[t] ═══
                string current_dyId = _tensorManager.CreateAndStoreZeros(new[] { 1, outputSize }, $"CurrentDY_t{t}");
                using (var dyFull = _tensorManager.LoadTensor(dyId))
                using (var current_dy = _tensorManager.LoadTensor(current_dyId))
                {
                    _mathEngine.Slice(dyFull, t, current_dy);
                    _tensorManager.UpdateTensor(current_dyId,
                        tensor => tensor.UpdateFromCpu(current_dy.ToCpuTensor().GetData()));
                }

                // ═══ GRADIENTE DE WHY e BY ═══
                using (var hiddenNext = _tensorManager.LoadTensor(cache["HiddenNext"]))
                using (var currDy = _tensorManager.LoadTensor(current_dyId))
                using (var temp_grad_why = _mathEngine.CreateTensor(new[] { hiddenSize, outputSize }))
                {
                    _mathEngine.MatrixMultiplyTransposeA(hiddenNext, currDy, temp_grad_why);
                    using (var grad_why = _tensorManager.LoadTensor(grads["why"]))
                    using (var grad_why_updated = _mathEngine.CreateTensor(temp_grad_why.Shape))
                    {
                        _mathEngine.Add(grad_why, temp_grad_why, grad_why_updated);
                        _tensorManager.UpdateTensor(grads["why"],
                            g => g.UpdateFromCpu(grad_why_updated.ToCpuTensor().GetData()));
                    }
                    using (var grad_by = _tensorManager.LoadTensor(grads["by"]))
                    using (var grad_by_updated = _mathEngine.CreateTensor(grad_by.Shape))
                    {
                        _mathEngine.Add(grad_by, currDy, grad_by_updated);
                        _tensorManager.UpdateTensor(grads["by"],
                            g => g.UpdateFromCpu(grad_by_updated.ToCpuTensor().GetData()));
                    }
                }

                // ═══ dh = dy * why^T + dh_next ═══
                string dhId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, $"DH_t{t}");
                using (var why = _tensorManager.LoadTensor(_weightsHiddenOutputFinalId))
                using (var currDy = _tensorManager.LoadTensor(current_dyId))
                using (var dh_temp = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
                using (var dh_next = _tensorManager.LoadTensor(dh_nextId))
                using (var dh = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
                {
                    _mathEngine.MatrixMultiplyTransposeB(currDy, why, dh_temp);
                    _mathEngine.Add(dh_temp, dh_next, dh);
                    _tensorManager.UpdateTensor(dhId, tensor => tensor.UpdateFromCpu(dh.ToCpuTensor().GetData()));
                }

                // ═══ BACKWARD ATRAVÉS DO OUTPUT GATE ═══
                string do_gateId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, $"DO_Gate_t{t}");
                using (var dh = _tensorManager.LoadTensor(dhId))
                using (var tanhCellNext = _tensorManager.LoadTensor(cache["TanhCellNext"]))
                using (var outputGate = _tensorManager.LoadTensor(cache["OutputGate"]))
                using (var dh_raw = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
                using (var sig_deriv = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
                using (var do_gate = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
                {
                    _mathEngine.Multiply(dh, tanhCellNext, dh_raw);
                    _mathEngine.SigmoidDerivative(outputGate, sig_deriv);
                    _mathEngine.Multiply(dh_raw, sig_deriv, do_gate);
                    _tensorManager.UpdateTensor(do_gateId, g => g.UpdateFromCpu(do_gate.ToCpuTensor().GetData()));
                }

                // Gradientes de wio, who, bo
                using (var input = _tensorManager.LoadTensor(cache["Input"]))
                using (var hiddenPrev = _tensorManager.LoadTensor(cache["HiddenPrev"]))
                using (var do_gate = _tensorManager.LoadTensor(do_gateId))
                using (var grad_wio_new = _mathEngine.CreateTensor(new[] { embeddingSize, hiddenSize }))
                using (var grad_who_new = _mathEngine.CreateTensor(new[] { hiddenSize, hiddenSize }))
                {
                    _mathEngine.MatrixMultiplyTransposeA(input, do_gate, grad_wio_new);
                    _mathEngine.MatrixMultiplyTransposeA(hiddenPrev, do_gate, grad_who_new);
                    using (var grad_wio = _tensorManager.LoadTensor(grads["wio"]))
                    using (var grad_wio_updated = _mathEngine.CreateTensor(grad_wio.Shape))
                    {
                        _mathEngine.Add(grad_wio, grad_wio_new, grad_wio_updated);
                        _tensorManager.UpdateTensor(grads["wio"],
                            g => g.UpdateFromCpu(grad_wio_updated.ToCpuTensor().GetData()));
                    }
                    using (var grad_who = _tensorManager.LoadTensor(grads["who"]))
                    using (var grad_who_updated = _mathEngine.CreateTensor(grad_who.Shape))
                    {
                        _mathEngine.Add(grad_who, grad_who_new, grad_who_updated);
                        _tensorManager.UpdateTensor(grads["who"],
                            g => g.UpdateFromCpu(grad_who_updated.ToCpuTensor().GetData()));
                    }
                    using (var grad_bo = _tensorManager.LoadTensor(grads["bo"]))
                    using (var grad_bo_updated = _mathEngine.CreateTensor(grad_bo.Shape))
                    {
                        _mathEngine.Add(grad_bo, do_gate, grad_bo_updated);
                        _tensorManager.UpdateTensor(grads["bo"],
                            g => g.UpdateFromCpu(grad_bo_updated.ToCpuTensor().GetData()));
                    }
                }

                // ═══ BACKWARD ATRAVÉS DA CELL STATE ═══
                string dcId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, $"DC_t{t}");
                using (var dh = _tensorManager.LoadTensor(dhId))
                using (var outputGate = _tensorManager.LoadTensor(cache["OutputGate"]))
                using (var tanhCellNext = _tensorManager.LoadTensor(cache["TanhCellNext"]))
                using (var dc_next = _tensorManager.LoadTensor(dc_nextId))
                using (var dc_from_h = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
                using (var tanh_deriv = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
                using (var dc = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
                {
                    _mathEngine.Multiply(dh, outputGate, dc_from_h);
                    _mathEngine.TanhDerivative(tanhCellNext, tanh_deriv);
                    _mathEngine.Multiply(dc_from_h, tanh_deriv, dc);
                    _mathEngine.Add(dc, dc_next, dc);
                    _tensorManager.UpdateTensor(dcId, g => g.UpdateFromCpu(dc.ToCpuTensor().GetData()));
                }

                // ═══ BACKWARD ATRAVÉS DO FORGET GATE ═══
                string df_gateId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, $"DF_Gate_t{t}");
                using (var dc = _tensorManager.LoadTensor(dcId))
                using (var cellPrev = _tensorManager.LoadTensor(cache["CellPrev"]))
                using (var forgetGate = _tensorManager.LoadTensor(cache["ForgetGate"]))
                using (var dc_times_cprev = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
                using (var sig_deriv = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
                using (var df_gate = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
                {
                    _mathEngine.Multiply(dc, cellPrev, dc_times_cprev);
                    _mathEngine.SigmoidDerivative(forgetGate, sig_deriv);
                    _mathEngine.Multiply(dc_times_cprev, sig_deriv, df_gate);
                    _tensorManager.UpdateTensor(df_gateId, g => g.UpdateFromCpu(df_gate.ToCpuTensor().GetData()));
                }

                // Gradientes de wif, whf, bf
                using (var input = _tensorManager.LoadTensor(cache["Input"]))
                using (var hiddenPrev = _tensorManager.LoadTensor(cache["HiddenPrev"]))
                using (var df_gate = _tensorManager.LoadTensor(df_gateId))
                using (var grad_wif_new = _mathEngine.CreateTensor(new[] { embeddingSize, hiddenSize }))
                using (var grad_whf_new = _mathEngine.CreateTensor(new[] { hiddenSize, hiddenSize }))
                {
                    _mathEngine.MatrixMultiplyTransposeA(input, df_gate, grad_wif_new);
                    _mathEngine.MatrixMultiplyTransposeA(hiddenPrev, df_gate, grad_whf_new);
                    using (var grad_wif = _tensorManager.LoadTensor(grads["wif"]))
                    using (var grad_wif_updated = _mathEngine.CreateTensor(grad_wif.Shape))
                    {
                        _mathEngine.Add(grad_wif, grad_wif_new, grad_wif_updated);
                        _tensorManager.UpdateTensor(grads["wif"],
                            g => g.UpdateFromCpu(grad_wif_updated.ToCpuTensor().GetData()));
                    }
                    using (var grad_whf = _tensorManager.LoadTensor(grads["whf"]))
                    using (var grad_whf_updated = _mathEngine.CreateTensor(grad_whf.Shape))
                    {
                        _mathEngine.Add(grad_whf, grad_whf_new, grad_whf_updated);
                        _tensorManager.UpdateTensor(grads["whf"],
                            g => g.UpdateFromCpu(grad_whf_updated.ToCpuTensor().GetData()));
                    }
                    using (var grad_bf = _tensorManager.LoadTensor(grads["bf"]))
                    using (var grad_bf_updated = _mathEngine.CreateTensor(grad_bf.Shape))
                    {
                        _mathEngine.Add(grad_bf, df_gate, grad_bf_updated);
                        _tensorManager.UpdateTensor(grads["bf"],
                            g => g.UpdateFromCpu(grad_bf_updated.ToCpuTensor().GetData()));
                    }
                }

                // ═══ BACKWARD ATRAVÉS DO INPUT GATE ═══
                string di_gateId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, $"DI_Gate_t{t}");
                using (var dc = _tensorManager.LoadTensor(dcId))
                using (var cellCandidate = _tensorManager.LoadTensor(cache["CellCandidate"]))
                using (var inputGate = _tensorManager.LoadTensor(cache["InputGate"]))
                using (var dc_times_cand = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
                using (var sig_deriv = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
                using (var di_gate = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
                {
                    _mathEngine.Multiply(dc, cellCandidate, dc_times_cand);
                    _mathEngine.SigmoidDerivative(inputGate, sig_deriv);
                    _mathEngine.Multiply(dc_times_cand, sig_deriv, di_gate);
                    _tensorManager.UpdateTensor(di_gateId, g => g.UpdateFromCpu(di_gate.ToCpuTensor().GetData()));
                }

                // Gradientes de wii, whi, bi
                using (var input = _tensorManager.LoadTensor(cache["Input"]))
                using (var hiddenPrev = _tensorManager.LoadTensor(cache["HiddenPrev"]))
                using (var di_gate = _tensorManager.LoadTensor(di_gateId))
                using (var grad_wii_new = _mathEngine.CreateTensor(new[] { embeddingSize, hiddenSize }))
                using (var grad_whi_new = _mathEngine.CreateTensor(new[] { hiddenSize, hiddenSize }))
                {
                    _mathEngine.MatrixMultiplyTransposeA(input, di_gate, grad_wii_new);
                    _mathEngine.MatrixMultiplyTransposeA(hiddenPrev, di_gate, grad_whi_new);
                    using (var grad_wii = _tensorManager.LoadTensor(grads["wii"]))
                    using (var grad_wii_updated = _mathEngine.CreateTensor(grad_wii.Shape))
                    {
                        _mathEngine.Add(grad_wii, grad_wii_new, grad_wii_updated);
                        _tensorManager.UpdateTensor(grads["wii"],
                            g => g.UpdateFromCpu(grad_wii_updated.ToCpuTensor().GetData()));
                    }
                    using (var grad_whi = _tensorManager.LoadTensor(grads["whi"]))
                    using (var grad_whi_updated = _mathEngine.CreateTensor(grad_whi.Shape))
                    {
                        _mathEngine.Add(grad_whi, grad_whi_new, grad_whi_updated);
                        _tensorManager.UpdateTensor(grads["whi"],
                            g => g.UpdateFromCpu(grad_whi_updated.ToCpuTensor().GetData()));
                    }
                    using (var grad_bi = _tensorManager.LoadTensor(grads["bi"]))
                    using (var grad_bi_updated = _mathEngine.CreateTensor(grad_bi.Shape))
                    {
                        _mathEngine.Add(grad_bi, di_gate, grad_bi_updated);
                        _tensorManager.UpdateTensor(grads["bi"],
                            g => g.UpdateFromCpu(grad_bi_updated.ToCpuTensor().GetData()));
                    }
                }

                // ═══ BACKWARD ATRAVÉS DO CELL CANDIDATE ═══
                string dc_candId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, $"DC_Cand_t{t}");
                using (var dc = _tensorManager.LoadTensor(dcId))
                using (var inputGate = _tensorManager.LoadTensor(cache["InputGate"]))
                using (var cellCandidate = _tensorManager.LoadTensor(cache["CellCandidate"]))
                using (var dc_times_igate = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
                using (var tanh_deriv = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
                using (var dc_cand = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
                {
                    _mathEngine.Multiply(dc, inputGate, dc_times_igate);
                    _mathEngine.TanhDerivative(cellCandidate, tanh_deriv);
                    _mathEngine.Multiply(dc_times_igate, tanh_deriv, dc_cand);
                    _tensorManager.UpdateTensor(dc_candId, g => g.UpdateFromCpu(dc_cand.ToCpuTensor().GetData()));
                }

                // Gradientes de wic, whc, bc
                using (var input = _tensorManager.LoadTensor(cache["Input"]))
                using (var hiddenPrev = _tensorManager.LoadTensor(cache["HiddenPrev"]))
                using (var dc_cand = _tensorManager.LoadTensor(dc_candId))
                using (var grad_wic_new = _mathEngine.CreateTensor(new[] { embeddingSize, hiddenSize }))
                using (var grad_whc_new = _mathEngine.CreateTensor(new[] { hiddenSize, hiddenSize }))
                {
                    _mathEngine.MatrixMultiplyTransposeA(input, dc_cand, grad_wic_new);
                    _mathEngine.MatrixMultiplyTransposeA(hiddenPrev, dc_cand, grad_whc_new);
                    using (var grad_wic = _tensorManager.LoadTensor(grads["wic"]))
                    using (var grad_wic_updated = _mathEngine.CreateTensor(grad_wic.Shape))
                    {
                        _mathEngine.Add(grad_wic, grad_wic_new, grad_wic_updated);
                        _tensorManager.UpdateTensor(grads["wic"],
                            g => g.UpdateFromCpu(grad_wic_updated.ToCpuTensor().GetData()));
                    }
                    using (var grad_whc = _tensorManager.LoadTensor(grads["whc"]))
                    using (var grad_whc_updated = _mathEngine.CreateTensor(grad_whc.Shape))
                    {
                        _mathEngine.Add(grad_whc, grad_whc_new, grad_whc_updated);
                        _tensorManager.UpdateTensor(grads["whc"],
                            g => g.UpdateFromCpu(grad_whc_updated.ToCpuTensor().GetData()));
                    }
                    using (var grad_bc = _tensorManager.LoadTensor(grads["bc"]))
                    using (var grad_bc_updated = _mathEngine.CreateTensor(grad_bc.Shape))
                    {
                        _mathEngine.Add(grad_bc, dc_cand, grad_bc_updated);
                        _tensorManager.UpdateTensor(grads["bc"],
                            g => g.UpdateFromCpu(grad_bc_updated.ToCpuTensor().GetData()));
                    }
                }

                // ═══ PROPAGA GRADIENTES PARA dh_prev e dc_prev ═══
                string dh_prevId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, $"DH_Prev_t{t}");
                using (var dh_prev_accum = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
                {
                    using (var whf = _tensorManager.LoadTensor(_weightsHiddenForgetId))
                    using (var df_gate = _tensorManager.LoadTensor(df_gateId))
                    using (var dh_from_f = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
                    {
                        _mathEngine.MatrixMultiplyTransposeB(df_gate, whf, dh_from_f);
                        _mathEngine.Add(dh_prev_accum, dh_from_f, dh_prev_accum);
                    }
                    using (var whi = _tensorManager.LoadTensor(_weightsHiddenInputId))
                    using (var di_gate = _tensorManager.LoadTensor(di_gateId))
                    using (var dh_from_i = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
                    {
                        _mathEngine.MatrixMultiplyTransposeB(di_gate, whi, dh_from_i);
                        _mathEngine.Add(dh_prev_accum, dh_from_i, dh_prev_accum);
                    }
                    using (var whc = _tensorManager.LoadTensor(_weightsHiddenCellId))
                    using (var dc_cand = _tensorManager.LoadTensor(dc_candId))
                    using (var dh_from_c = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
                    {
                        _mathEngine.MatrixMultiplyTransposeB(dc_cand, whc, dh_from_c);
                        _mathEngine.Add(dh_prev_accum, dh_from_c, dh_prev_accum);
                    }
                    using (var who = _tensorManager.LoadTensor(_weightsHiddenOutputId))
                    using (var do_gate = _tensorManager.LoadTensor(do_gateId))
                    using (var dh_from_o = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
                    {
                        _mathEngine.MatrixMultiplyTransposeB(do_gate, who, dh_from_o);
                        _mathEngine.Add(dh_prev_accum, dh_from_o, dh_prev_accum);
                    }
                    _tensorManager.UpdateTensor(dh_prevId,
                        tensor => tensor.UpdateFromCpu(dh_prev_accum.ToCpuTensor().GetData()));
                }
                
                string dc_prevId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, $"DC_Prev_t{t}");
                using (var dc = _tensorManager.LoadTensor(dcId))
                using (var forgetGate = _tensorManager.LoadTensor(cache["ForgetGate"]))
                using (var dc_prev = _mathEngine.CreateTensor(new[] { 1, hiddenSize }))
                {
                    _mathEngine.Multiply(dc, forgetGate, dc_prev);
                    _tensorManager.UpdateTensor(dc_prevId,
                        tensor => tensor.UpdateFromCpu(dc_prev.ToCpuTensor().GetData()));
                }

                // ═══ GRADIENTE DE EMBEDDING ═══
                string d_inputId = _tensorManager.CreateAndStoreZeros(new[] { 1, embeddingSize }, $"DInput_t{t}");
                using (var d_input_accum = _mathEngine.CreateTensor(new[] { 1, embeddingSize }))
                {
                    using (var wif = _tensorManager.LoadTensor(_weightsInputForgetId))
                    using (var df_gate = _tensorManager.LoadTensor(df_gateId))
                    using (var d_from_f = _mathEngine.CreateTensor(new[] { 1, embeddingSize }))
                    {
                        _mathEngine.MatrixMultiplyTransposeB(df_gate, wif, d_from_f);
                        _mathEngine.Add(d_input_accum, d_from_f, d_input_accum);
                    }
                    using (var wii = _tensorManager.LoadTensor(_weightsInputInputId))
                    using (var di_gate = _tensorManager.LoadTensor(di_gateId))
                    using (var d_from_i = _mathEngine.CreateTensor(new[] { 1, embeddingSize }))
                    {
                        _mathEngine.MatrixMultiplyTransposeB(di_gate, wii, d_from_i);
                        _mathEngine.Add(d_input_accum, d_from_i, d_input_accum);
                    }
                    using (var wic = _tensorManager.LoadTensor(_weightsInputCellId))
                    using (var dc_cand = _tensorManager.LoadTensor(dc_candId))
                    using (var d_from_c = _mathEngine.CreateTensor(new[] { 1, embeddingSize }))
                    {
                        _mathEngine.MatrixMultiplyTransposeB(dc_cand, wic, d_from_c);
                        _mathEngine.Add(d_input_accum, d_from_c, d_input_accum);
                    }
                    using (var wio = _tensorManager.LoadTensor(_weightsInputOutputId))
                    using (var do_gate = _tensorManager.LoadTensor(do_gateId))
                    using (var d_from_o = _mathEngine.CreateTensor(new[] { 1, embeddingSize }))
                    {
                        _mathEngine.MatrixMultiplyTransposeB(do_gate, wio, d_from_o);
                        _mathEngine.Add(d_input_accum, d_from_o, d_input_accum);
                    }
                    _tensorManager.UpdateTensor(d_inputId,
                        tensor => tensor.UpdateFromCpu(d_input_accum.ToCpuTensor().GetData()));
                }

                using (var grad_w_embed = _tensorManager.LoadTensor(grads["w_embed"]))
                using (var d_input = _tensorManager.LoadTensor(d_inputId))
                {
                    int tokenIndex = inputIndices[t];
                    _mathEngine.AccumulateGradient(grad_w_embed, d_input, tokenIndex);
                    _tensorManager.UpdateTensor(grads["w_embed"],
                        tensor => tensor.UpdateFromCpu(grad_w_embed.ToCpuTensor().GetData()));
                }

                dh_nextId = dh_prevId;
                dc_nextId = dc_prevId;
            }

            // ═══ GRADIENT CLIPPING (RESTAURADO) ═══
            float totalNorm = ComputeTotalNormFromIds(grads.Values.ToList());
            const float MAX_NORM = 1.0f;

            if (totalNorm > MAX_NORM)
            {
                float clipScale = MAX_NORM / (totalNorm + 1e-8f);
                Console.WriteLine($"[BackwardOptimized] Clipping gradientes: norm={totalNorm:F4} → scale={clipScale:F4}");

                foreach (var gradId in grads.Values)
                {
                    _tensorManager.UpdateTensor(gradId, t =>
                    {
                        var data = t.ToCpuTensor().GetData();
                        for (int i = 0; i < data.Length; i++)
                            data[i] *= clipScale;
                        t.UpdateFromCpu(data);
                    });
                }
            }

            sw.Stop();
            Console.WriteLine($"[BackwardOptimized] Concluído em {sw.ElapsedMilliseconds}ms");
            Console.WriteLine($"  └─ Gradient norm: {totalNorm:F4}");

            return grads;
        }


        /// <summary>
        /// Calcula a norma L2 total de todos os gradientes.
        /// Usado para gradient clipping global.
        /// </summary>
        private float ComputeTotalNormFromIds(List<string> gradientIds)
        {
            float totalSumSquares = 0;
            int processedGradients = 0;

            foreach (var id in gradientIds)
            {
                try
                {
                    using var grad = _tensorManager.LoadTensor(id);
                    var data = grad.ToCpuTensor().GetData();

                    // Ignora gradientes com NaN/Inf
                    if (data.Any(g => float.IsNaN(g) || float.IsInfinity(g)))
                    {
                        Console.WriteLine($"[ComputeNorm] Aviso: Gradiente {id} contém NaN/Inf, ignorando.");
                        continue;
                    }

                    totalSumSquares += data.Sum(d => d * d);
                    processedGradients++;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[ComputeNorm] Erro ao processar {id}: {ex.Message}");
                }
            }

            if (processedGradients == 0)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("[ComputeNorm] AVISO: Nenhum gradiente válido encontrado!");
                Console.ResetColor();
                return 0.0f;
            }

            return MathF.Sqrt(totalSumSquares);
        }

        /// <summary>
        /// Atualiza pesos usando Adam.
        /// ATUALIZADO: Não apaga mais os tensores de gradiente.
        /// </summary>
        public void UpdateWeightsWithAdamGpu(Dictionary<string, string> gradientIds, float learningRate)
        {
            Console.WriteLine("[UpdateWeights] Aplicando Adam...");
            var sw = Stopwatch.StartNew();

            var weightMappings = new Dictionary<string, string>
            {
                ["w_embed"] = _weightsEmbeddingId,
                ["wif"] = _weightsInputForgetId,
                ["whf"] = _weightsHiddenForgetId,
                ["wii"] = _weightsInputInputId,
                ["whi"] = _weightsHiddenInputId,
                ["wic"] = _weightsInputCellId,
                ["whc"] = _weightsHiddenCellId,
                ["wio"] = _weightsInputOutputId,
                ["who"] = _weightsHiddenOutputId,
                ["why"] = _weightsHiddenOutputFinalId,
                ["bf"] = _biasForgetId,
                ["bi"] = _biasInputId,
                ["bc"] = _biasCellId,
                ["bo"] = _biasOutputId,
                ["by"] = _biasOutputFinalId
            };

            int layerId = 0;
            int updatedLayers = 0;
            int skippedLayers = 0;

            foreach (var (key, weightId) in weightMappings)
            {
                if (!gradientIds.ContainsKey(key))
                {
                    Console.WriteLine($"[UpdateWeights] Aviso: Gradiente '{key}' não encontrado.");
                    layerId++;
                    skippedLayers++;
                    continue;
                }

                using var grad = _tensorManager.LoadTensor(gradientIds[key]);
                var gradData = grad.ToCpuTensor().GetData();

                if (gradData.Any(g => float.IsNaN(g) || float.IsInfinity(g)))
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"[UpdateWeights] ✗ Pulando '{key}': Contém NaN/Inf");
                    Console.ResetColor();
                    layerId++;
                    skippedLayers++;
                    continue;
                }

                try
                {
                    using var weight = _tensorManager.LoadTensor(weightId);
                    _adamOptimizer.UpdateParametersGpu(layerId, weight, grad, _mathEngine);
                    _tensorManager.UpdateTensor(weightId, t =>
                        t.UpdateFromCpu(weight.ToCpuTensor().GetData()));
                    updatedLayers++;
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"[UpdateWeights] ✗ Erro ao atualizar '{key}': {ex.Message}");
                    Console.ResetColor();
                    skippedLayers++;
                }

                layerId++;
            }

            // A SEÇÃO DE CLEANUP FOI REMOVIDA DAQUI.

            sw.Stop();
            Console.WriteLine($"[UpdateWeights] ✓ Concluído em {sw.ElapsedMilliseconds}ms");
            Console.WriteLine($"  ├─ Camadas atualizadas: {updatedLayers}/{weightMappings.Count}");
            if (skippedLayers > 0)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"  └─ Camadas puladas: {skippedLayers}");
                Console.ResetColor();
            }
        }
        
        // Adicione estes dois novos métodos à classe NeuralNetworkLSTM em NeuralNetworkLSTM.cs

        /// <summary>
        /// Retorna um conjunto com os IDs de todos os tensores permanentes do modelo.
        /// </summary>
        private HashSet<string> GetPermanentTensorIds()
        {
            return new HashSet<string>
            {
                _weightsEmbeddingId,
                _weightsInputForgetId,
                _weightsHiddenForgetId,
                _weightsInputInputId,
                _weightsHiddenInputId,
                _weightsInputCellId,
                _weightsHiddenCellId,
                _weightsInputOutputId,
                _weightsHiddenOutputId,
                _biasForgetId,
                _biasInputId,
                _biasCellId,
                _biasOutputId,
                _weightsHiddenOutputFinalId,
                _biasOutputFinalId,
                _hiddenStateId,
                _cellStateId
            };
        }

        /// <summary>
        /// Destrói todos os tensores temporários (intermediários e de gradiente)
        /// que foram criados no disco durante uma época de treinamento,
        /// preservando apenas os pesos permanentes do modelo.
        /// </summary>
        public void ClearEpochTemporaryTensors()
        {
            Console.WriteLine("[Cleanup] Iniciando limpeza de tensores temporários da época...");
            var sw = Stopwatch.StartNew();

            var permanentIds = GetPermanentTensorIds();
            var allTensorIds = _tensorManager.ListTensors();

            // Para evitar modificar a coleção enquanto iteramos, criamos uma lista de exclusão
            var tensorsToDelete = allTensorIds
                .Where(id => !permanentIds.Contains(id))
                .ToList();

            int deleteCount = 0;
            foreach (var tensorId in tensorsToDelete)
            {
                _tensorManager.DeleteTensor(tensorId);
                deleteCount++;
            }
            
            // O cache do forward já deve estar limpo, mas garantimos aqui.
            _forwardCache.Clear();

            sw.Stop();
            Console.WriteLine($"[Cleanup] ✓ {deleteCount} tensores temporários removidos em {sw.ElapsedMilliseconds}ms.");
        }

        /// <summary>
        /// Treina uma sequência completa: Forward → Backward → Update.
        /// </summary>
        /// <summary>
        /// Treina uma sequência completa: Forward → Backward → Update.
        /// Gerencia lifecycle completo de tensores temporários.
        /// </summary>
        public float TrainSequence(int[] inputIndices, int[] targetIndices, float learningRate)
        {
            //Console.WriteLine($"\n[TrainSequence] Sequência de {inputIndices.Length} tokens...");
            var swTotal = Stopwatch.StartNew();

            // Estatísticas antes
            var statsBefore = _tensorManager.GetStatistics();

            try
            {
                // ═══ FORWARD PASS ═══
                var (predsId, loss) = ForwardPassGpuOptimized(inputIndices, targetIndices, inputIndices.Length);

                // ═══ BACKWARD PASS ═══
                var grads = BackwardPassGpuOptimized(predsId, targetIndices, inputIndices, inputIndices.Length);

                // ═══ WEIGHT UPDATE ═══
                UpdateWeightsWithAdamGpu(grads, learningRate);

                // ═══ CLEANUP ═══
                _tensorManager.DeleteTensor(predsId);

                // Forward cache já foi limpo no BackwardPass
                // Mas garantimos novamente por segurança
                ClearForwardCache();

                swTotal.Stop();
                var statsAfter = _tensorManager.GetStatistics();

                // ═══ RELATÓRIO ═══
                Console.WriteLine($"[TrainSequence] ✓ Completo em {swTotal.ElapsedMilliseconds}ms");
                Console.WriteLine($"  ├─ Loss: {loss:F4}");
                Console.WriteLine($"  ├─ Tensores: {statsBefore.Count} → {statsAfter.Count}");
                Console.WriteLine($"  ├─ RAM: {statsAfter.RamMB}MB");
                Console.WriteLine(
                    $"  └─ Throughput: {inputIndices.Length * 1000.0 / swTotal.ElapsedMilliseconds:F1} tokens/s\n");

                return loss;
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[TrainSequence] ✗ Erro: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
                Console.ResetColor();

                // Tenta cleanup mesmo em caso de erro
                try
                {
                    ClearForwardCache();
                }
                catch
                {
                }

                throw;
            }
        }

        public void Dispose()
        {
            if (_disposed) return;

            Console.WriteLine("\n[Dispose] Finalizando modelo disk-backed...");
            var stats = _tensorManager.GetStatistics();
            Console.WriteLine($"  ├─ {stats.Count} tensores processados");
            Console.WriteLine($"  ├─ {stats.TotalAccesses} acessos totais");
            Console.WriteLine($"  └─ Removendo todos os arquivos .tensor...");

            _tensorManager?.Dispose();
            _adamOptimizer?.Reset();

            _disposed = true;
            GC.SuppressFinalize(this);
            Console.WriteLine("[Dispose] ✓ Limpeza completa.\n");
        }

        // NOVO: Método público para limpar o cache do forward pass.
        // Necessário para a validação, onde o backward pass não é chamado.
        public void ClearForwardCache()
        {
            if (_forwardCache.Count == 0) return;

            foreach (var stepCache in _forwardCache.Values)
            {
                foreach (var tensorId in stepCache.Values)
                    _tensorManager.DeleteTensor(tensorId);
            }

            _forwardCache.Clear();
        }

        // NOVO: Método público para deletar um tensor temporário
        public void DeleteTensor(string tensorId)
        {
            _tensorManager.DeleteTensor(tensorId);
        }

        // NOVO: Método público para resetar o estado do otimizador
        public void ResetOptimizerState()
        {
            _adamOptimizer.Reset();
        }
    }
}