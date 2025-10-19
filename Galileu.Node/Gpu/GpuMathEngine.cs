using Galileu.Node.Interfaces;
using OpenCL.NetCore;
using System;
using System.Linq;
using static OpenCL.NetCore.Cl;

namespace Galileu.Node.Gpu;

public class GpuMathEngine : IMathEngine, IDisposable
{
    public bool IsGpu => true;
    private readonly Context _context;
    private readonly CommandQueue _commandQueue;
    private readonly OpenCL.NetCore.Program _program;

    // Kernels
    private readonly Kernel _matrixMultiplyKernel;
    private readonly Kernel _vectorMatrixMultiplyKernel;
    private readonly Kernel _addKernel;
    private readonly Kernel _addBroadcastKernel;
    private readonly Kernel _multiplyKernel;
    private readonly Kernel _sigmoidKernel;
    private readonly Kernel _tanhKernel;
    private readonly Kernel _cloneKernel;
    private readonly Kernel _transposeKernel;
    private readonly Kernel _subtractKernel;
    private readonly Kernel _sigmoidDerivativeKernel;
    private readonly Kernel _tanhDerivativeKernel;
    private readonly Kernel _matrixMultiplyTransposeAKernel;
    private readonly Kernel _matrixMultiplyTransposeBKernel;
    private readonly Kernel _addScaledKernel;
    private readonly Kernel _subtractScaledKernel;
    private readonly Kernel _sliceKernel;
    private readonly Kernel _setKernel;
    private readonly Kernel _clipKernel;
    private readonly Kernel _scaleKernel;
    private readonly Kernel _softmaxKernel;
    private readonly Kernel _lookupKernel;
    private readonly Kernel _accumulateGradientKernel;
    private readonly Kernel _oneHotEncodeKernel;
    private readonly Kernel _adamUpdateKernel;

    private bool _disposed = false;

    #region Kernels Source

    public void Synchronize()
    {
        Finish(_commandQueue);
    }

    private const string ProgramSource = @"
            __kernel void matrix_multiply(__global const float* A, __global const float* B, __global float* C, int M, int N, int P) { 
                int i = get_global_id(0); int j = get_global_id(1); 
                if (i < M && j < P) { 
                    float sum = 0.0f; 
                    for (int k = 0; k < N; ++k) { sum += A[i * N + k] * B[k * P + j]; } 
                    C[i * P + j] = sum; 
                } 
            }

            __kernel void vector_matrix_multiply(__global const float* V, __global const float* M, __global float* R, int N, int P) {
                int j = get_global_id(0);
                if (j < P) {
                    float sum = 0.0f;
                    for (int k = 0; k < N; ++k) { sum += V[k] * M[k * P + j]; }
                    R[j] = sum;
                }
            }
            
            __kernel void add(__global const float* a, __global const float* b, __global float* result) { int gid = get_global_id(0); result[gid] = a[gid] + b[gid]; }
            
            // CORRIGIDO: Kernel agora é out-of-place para corresponder à assinatura do método
            __kernel void add_broadcast(
    __global const float* a,        // Input matrix [M x N]
    __global const float* bias,     // Bias vector [N]
    __global float* result,         // Output matrix [M x N]
    int bias_size)                  // N
{
    int gid = get_global_id(0);
    int col = gid % bias_size;
    
    // Proteção contra índices inválidos
    if (col < bias_size && gid < get_global_size(0))
    {
        float a_val = a[gid];
        float bias_val = bias[col];
        
        // Verifica valores inválidos
        if (isnan(a_val) || isinf(a_val))
        {
            result[gid] = 0.0f; // Fallback seguro
        }
        else if (isnan(bias_val) || isinf(bias_val))
        {
            result[gid] = a_val; // Usa apenas o valor de 'a'
        }
        else
        {
            result[gid] = a_val + bias_val;
        }
    }
}

            __kernel void multiply(__global const float* a, __global const float* b, __global float* result) { int gid = get_global_id(0); result[gid] = a[gid] * b[gid]; }
            __kernel void sigmoid(__global const float* a, __global float* result) { int gid = get_global_id(0); result[gid] = 1.0f / (1.0f + exp(-a[gid])); }
            __kernel void tanh_activation(__global const float* a, __global float* result) { int gid = get_global_id(0); result[gid] = tanh(a[gid]); }
            __kernel void clone_buffer(__global const float* input, __global float* output) { int gid = get_global_id(0); output[gid] = input[gid]; }
            __kernel void transpose(__global const float* input, __global float* output, int rows, int cols) { int i = get_global_id(0); int j = get_global_id(1); if (i < rows && j < cols) { output[j * rows + i] = input[i * cols + j]; } }
            __kernel void subtract(__global const float* a, __global const float* b, __global float* result) { int gid = get_global_id(0); result[gid] = a[gid] - b[gid]; }
            __kernel void sigmoid_derivative(__global const float* output, __global float* result) { int gid = get_global_id(0); float o = output[gid]; result[gid] = o * (1.0f - o); }
            __kernel void tanh_derivative(__global const float* output, __global float* result) { int gid = get_global_id(0); float o = output[gid]; result[gid] = 1.0f - o * o; }
            __kernel void matrix_multiply_transpose_a(__global const float* A, __global const float* B, __global float* C, int M, int K, int P) { int i = get_global_id(0); int j = get_global_id(1); if (i < M && j < P) { float sum = 0.0f; for (int k = 0; k < K; ++k) { sum += A[k * M + i] * B[k * P + j]; } C[i * P + j] = sum; } }
            __kernel void matrix_multiply_transpose_b(__global const float* A, __global const float* B, __global float* C, int M, int K, int P) { int i = get_global_id(0); int j = get_global_id(1); if (i < M && j < P) { float sum = 0.0f; for (int k = 0; k < K; ++k) { sum += A[i * K + k] * B[j * K + k]; } C[i * P + j] = sum; } }
            __kernel void add_scaled(__global float* target, __global const float* source, float scalar) { int gid = get_global_id(0); target[gid] += source[gid] * scalar; }
            __kernel void subtract_scaled(__global float* target, __global const float* source, float scalar) { int gid = get_global_id(0); target[gid] -= source[gid] * scalar; }
            __kernel void slice(__global const float* source, __global float* dest, int offset, int size) { int gid = get_global_id(0); if (gid < size) { dest[gid] = source[offset + gid]; } }
            __kernel void set(__global float* dest, __global const float* source, int offset, int size) { int gid = get_global_id(0); if (gid < size) { dest[offset + gid] = source[gid]; } }
            __kernel void clip(__global float* data, float min_val, float max_val) { int gid = get_global_id(0); data[gid] = fmax(min_val, fmin(max_val, data[gid])); }
            __kernel void scale(__global float* data, float scalar) { int gid = get_global_id(0); data[gid] *= scalar; }
            __kernel void softmax(__global const float* input, __global float* output, int size) {
                int row = get_global_id(0);
                int offset = row * size;
                float maxVal = input[offset];
                for (int i = 1; i < size; i++) { if (input[offset + i] > maxVal) maxVal = input[offset + i]; }
                float sumExp = 0.0f;
                for (int i = 0; i < size; i++) { output[offset + i] = exp(input[offset + i] - maxVal); sumExp += output[offset + i]; }
                for (int i = 0; i < size; i++) { output[offset + i] /= sumExp; }
            }
            __kernel void lookup(__global const float* embedding_matrix, __global float* result, int index, int embedding_size) {
                int gid = get_global_id(0);
                if (gid < embedding_size) { result[gid] = embedding_matrix[index * embedding_size + gid]; }
            }
            __kernel void accumulate_gradient_no_atomic(__global float* embedding_gradients, __global const float* gradient, int index, int embedding_size) {
                int gid = get_global_id(0);
                if (gid < embedding_size) { embedding_gradients[index * embedding_size + gid] += gradient[gid]; }
            }
             __kernel void one_hot_encode(__global float* output, __global const int* indices, int total_classes) {
                int i = get_global_id(0);
                int row_offset = i * total_classes;
                for(int j = 0; j < total_classes; ++j) { output[row_offset + j] = 0.0f; }
                int one_hot_index = indices[i];
                output[row_offset + one_hot_index] = 1.0f;
            }
            __kernel void adam_update(
    __global float* p,              // Parâmetros
    __global const float* g,        // Gradientes
    __global float* m,              // Momento de primeira ordem
    __global float* v,              // Momento de segunda ordem
    float lr,                       // Learning rate
    float beta1,
    float beta2,
    float epsilon,
    int t)                          // Timestep
{
    int i = get_global_id(0);
    
    // Proteção: verifica se gradiente é válido
    float grad = g[i];
    if (isnan(grad) || isinf(grad))
    {
        // Se gradiente inválido, não atualiza este parâmetro
        return;
    }
    
    // Clipping de gradiente individual (proteção adicional)
    grad = fmax(grad, -10.0f);
    grad = fmin(grad, 10.0f);
    
    // Atualiza momentos
    float m_val = beta1 * m[i] + (1.0f - beta1) * grad;
    float v_val = beta2 * v[i] + (1.0f - beta2) * (grad * grad);
    
    // Correção de bias
    float beta1_pow_t = pow(beta1, (float)t);
    float beta2_pow_t = pow(beta2, (float)t);
    
    // Proteção contra divisão por zero
    if (beta1_pow_t >= 1.0f) beta1_pow_t = 1.0f - 1e-7f;
    if (beta2_pow_t >= 1.0f) beta2_pow_t = 1.0f - 1e-7f;
    
    float m_hat = m_val / (1.0f - beta1_pow_t);
    float v_hat = v_val / (1.0f - beta2_pow_t);
    
    // Calcula atualização
    float denominator = sqrt(v_hat) + epsilon;
    float update = lr * m_hat / denominator;
    
    // Proteção: verifica se atualização é válida
    if (isnan(update) || isinf(update))
    {
        // Se atualização inválida, não atualiza
        return;
    }
    
    // Clipping de atualização (proteção adicional)
    update = fmax(update, -1.0f);  // Limite de -1.0
    update = fmin(update, 1.0f);   // Limite de +1.0
    
    // Atualiza parâmetro
    float new_param = p[i] - update;
    
    // Proteção final: verifica se novo parâmetro é válido
    if (isnan(new_param) || isinf(new_param))
    {
        // Se novo valor inválido, mantém valor anterior
        return;
    }
    
    // Atualiza valores
    m[i] = m_val;
    v[i] = v_val;
    p[i] = new_param;
}

        ";

    #endregion

    public GpuMathEngine()
    {
        ErrorCode error;
        Platform[] platforms = GetPlatformIDs(out error);
        CheckError(error);
        var platform = platforms.First();
        Device[] devices = GetDeviceIDs(platform, DeviceType.Gpu, out error);
        if (error != ErrorCode.Success || devices.Length == 0)
        {
            devices = GetDeviceIDs(platform, DeviceType.Cpu, out error);
            CheckError(error);
        }

        var device = devices[0];
        Console.WriteLine($"[OpenCL] Usando dispositivo: {GetDeviceInfo(device, DeviceInfo.Name, out error)}");
        _context = CreateContext(null, 1, new[] { device }, null, IntPtr.Zero, out error);
        CheckError(error);
        _commandQueue = CreateCommandQueue(_context, device, CommandQueueProperties.None, out error);
        CheckError(error);
        _program = CreateProgramWithSource(_context, 1, new[] { ProgramSource }, null, out error);
        CheckError(error);
        error = BuildProgram(_program, 1, new[] { device }, string.Empty, null, IntPtr.Zero);
        if (error != ErrorCode.Success)
        {
            throw new OpenClException(
                $"Erro ao compilar kernels: {GetProgramBuildInfo(_program, device, ProgramBuildInfo.Log, out _)}",
                error);
        }

        _matrixMultiplyKernel = CreateKernel(_program, "matrix_multiply", out error);
        CheckError(error);
        _vectorMatrixMultiplyKernel = CreateKernel(_program, "vector_matrix_multiply", out error);
        CheckError(error);
        _addKernel = CreateKernel(_program, "add", out error);
        CheckError(error);
        _addBroadcastKernel = CreateKernel(_program, "add_broadcast", out error);
        CheckError(error);
        _multiplyKernel = CreateKernel(_program, "multiply", out error);
        CheckError(error);
        _sigmoidKernel = CreateKernel(_program, "sigmoid", out error);
        CheckError(error);
        _tanhKernel = CreateKernel(_program, "tanh_activation", out error);
        CheckError(error);
        _cloneKernel = CreateKernel(_program, "clone_buffer", out error);
        CheckError(error);
        _transposeKernel = CreateKernel(_program, "transpose", out error);
        CheckError(error);
        _subtractKernel = CreateKernel(_program, "subtract", out error);
        CheckError(error);
        _sigmoidDerivativeKernel = CreateKernel(_program, "sigmoid_derivative", out error);
        CheckError(error);
        _tanhDerivativeKernel = CreateKernel(_program, "tanh_derivative", out error);
        CheckError(error);
        _matrixMultiplyTransposeAKernel = CreateKernel(_program, "matrix_multiply_transpose_a", out error);
        CheckError(error);
        _matrixMultiplyTransposeBKernel = CreateKernel(_program, "matrix_multiply_transpose_b", out error);
        CheckError(error);
        _addScaledKernel = CreateKernel(_program, "add_scaled", out error);
        CheckError(error);
        _subtractScaledKernel = CreateKernel(_program, "subtract_scaled", out error);
        CheckError(error);
        _sliceKernel = CreateKernel(_program, "slice", out error);
        CheckError(error);
        _setKernel = CreateKernel(_program, "set", out error);
        CheckError(error);
        _clipKernel = CreateKernel(_program, "clip", out error);
        CheckError(error);
        _scaleKernel = CreateKernel(_program, "scale", out error);
        CheckError(error);
        _softmaxKernel = CreateKernel(_program, "softmax", out error);
        CheckError(error);
        _lookupKernel = CreateKernel(_program, "lookup", out error);
        CheckError(error);
        _accumulateGradientKernel = CreateKernel(_program, "accumulate_gradient_no_atomic", out error);
        CheckError(error);
        _oneHotEncodeKernel = CreateKernel(_program, "one_hot_encode", out error);
        CheckError(error);
        _adamUpdateKernel = CreateKernel(_program, "adam_update", out error);
        CheckError(error);
    }

    public IMathTensor CreateTensor(int[] shape) => new GpuTensor(shape, _context, _commandQueue);
    public IMathTensor CreateTensor(double[] data, int[] shape) => new GpuTensor(data, shape, _context, _commandQueue);

    // ====================================================================
    // MÉTODOS PÚBLICOS DE OPERAÇÕES MATEMÁTICAS (AGORA MAIS LIMPOS)
    // ====================================================================

    public void VectorMatrixMultiply(IMathTensor vec, IMathTensor mat, IMathTensor res)
    {
        var tensorV = (GpuTensor)vec;
        var tensorM = (GpuTensor)mat;
        int N = tensorV.Shape[1];
        int P = tensorM.Shape[1];

        ExecuteKernel1D(_vectorMatrixMultiplyKernel, P, tensorV, tensorM, (GpuTensor)res, N, P);
    }

    public void MatrixMultiply(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        var tensorA = (GpuTensor)a;
        var tensorB = (GpuTensor)b;
        int M = tensorA.Shape[0];
        int N = tensorA.Shape[1];
        int P = tensorB.Shape[1];

        ExecuteKernel2D(_matrixMultiplyKernel, M, P, tensorA, tensorB, (GpuTensor)result, M, N, P);
    }

    public void Add(IMathTensor a, IMathTensor b, IMathTensor result)
        => ExecuteKernel1D(_addKernel, a.Length, (GpuTensor)a, (GpuTensor)b, (GpuTensor)result);

    public void AddBroadcast(IMathTensor a, IMathTensor bias, IMathTensor result)
        => ExecuteKernel1D(_addBroadcastKernel, a.Length, (GpuTensor)a, (GpuTensor)bias, (GpuTensor)result,
            (int)bias.Length);

    public void Multiply(IMathTensor a, IMathTensor b, IMathTensor result)
        => ExecuteKernel1D(_multiplyKernel, a.Length, (GpuTensor)a, (GpuTensor)b, (GpuTensor)result);

    public void Sigmoid(IMathTensor a, IMathTensor result)
        => ExecuteKernel1D(_sigmoidKernel, a.Length, (GpuTensor)a, (GpuTensor)result);

    public void Tanh(IMathTensor a, IMathTensor result)
        => ExecuteKernel1D(_tanhKernel, a.Length, (GpuTensor)a, (GpuTensor)result);

    public IMathTensor Clone(IMathTensor tensor)
    {
        var newTensor = CreateTensor(tensor.Shape) as GpuTensor;
        ExecuteKernel1D(_cloneKernel, tensor.Length, (GpuTensor)tensor, newTensor);
        return newTensor;
    }

    public void Transpose(IMathTensor input, IMathTensor result)
    {
        int rows = input.Shape[0];
        int cols = input.Shape[1];
        ExecuteKernel2D(_transposeKernel, rows, cols, (GpuTensor)input, (GpuTensor)result, rows, cols);
    }

    public void Subtract(IMathTensor a, IMathTensor b, IMathTensor result)
        => ExecuteKernel1D(_subtractKernel, a.Length, (GpuTensor)a, (GpuTensor)b, (GpuTensor)result);

    public void SigmoidDerivative(IMathTensor output, IMathTensor result)
        => ExecuteKernel1D(_sigmoidDerivativeKernel, output.Length, (GpuTensor)output, (GpuTensor)result);

    public void TanhDerivative(IMathTensor output, IMathTensor result)
        => ExecuteKernel1D(_tanhDerivativeKernel, output.Length, (GpuTensor)output, (GpuTensor)result);

    public void MatrixMultiplyTransposeA(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        var tensorA = (GpuTensor)a;
        var tensorB = (GpuTensor)b;
        int M = tensorA.Shape[1];
        int K = tensorA.Shape[0];
        int P = tensorB.Shape[1];
        ExecuteKernel2D(_matrixMultiplyTransposeAKernel, M, P, tensorA, tensorB, (GpuTensor)result, M, K, P);
    }

    public void MatrixMultiplyTransposeB(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        var tensorA = (GpuTensor)a;
        var tensorB = (GpuTensor)b;
        int M = tensorA.Shape[0];
        int K = tensorA.Shape[1];
        int P = tensorB.Shape[0];
        ExecuteKernel2D(_matrixMultiplyTransposeBKernel, M, P, tensorA, tensorB, (GpuTensor)result, M, K, P);
    }

    public void AddScaled(IMathTensor target, IMathTensor source, double scalar)
        => ExecuteKernel1D(_addScaledKernel, target.Length, (GpuTensor)target, (GpuTensor)source, (float)scalar);

    public void SubtractScaled(IMathTensor target, IMathTensor source, double scalar)
        => ExecuteKernel1D(_subtractScaledKernel, target.Length, (GpuTensor)target, (GpuTensor)source, (float)scalar);

    public void Slice(IMathTensor source, int rowIndex, IMathTensor destination)
    {
        var featureSize = (int)destination.Length;
        var offset = rowIndex * featureSize;
        ExecuteKernel1D(_sliceKernel, featureSize, (GpuTensor)source, (GpuTensor)destination, offset, featureSize);
    }

    public void Set(IMathTensor destination, int rowIndex, IMathTensor source)
    {
        var featureSize = (int)source.Length;
        var offset = rowIndex * featureSize;
        ExecuteKernel1D(_setKernel, featureSize, (GpuTensor)destination, (GpuTensor)source, offset, featureSize);
    }

    public void Clip(IMathTensor tensor, double minValue, double maxValue)
        => ExecuteKernel1D(_clipKernel, tensor.Length, (GpuTensor)tensor, (float)minValue, (float)maxValue);

    public void Scale(IMathTensor tensor, double scalar)
        => ExecuteKernel1D(_scaleKernel, tensor.Length, (GpuTensor)tensor, (float)scalar);

    public void Softmax(IMathTensor input, IMathTensor result)
    {
        int rows = input.Shape[0];
        int cols = input.Shape[1];
        ExecuteKernel1D(_softmaxKernel, rows, (GpuTensor)input, (GpuTensor)result, cols);
    }

    public void Lookup(IMathTensor embeddingMatrix, int index, IMathTensor result)
    {
        var embeddingSize = embeddingMatrix.Shape[1];
        ExecuteKernel1D(_lookupKernel, embeddingSize, (GpuTensor)embeddingMatrix, (GpuTensor)result, index,
            embeddingSize);
    }

    public void AccumulateGradient(IMathTensor embeddingGradients, IMathTensor gradient, int index)
    {
        var embeddingSize = embeddingGradients.Shape[1];
        ExecuteKernel1D(_accumulateGradientKernel, embeddingSize, (GpuTensor)embeddingGradients, (GpuTensor)gradient,
            index, embeddingSize);
    }

    public IMathTensor CreateOneHotTensor(int[] indices, int totalClasses)
    {
        int sequenceLength = indices.Length;
        var resultTensor = CreateTensor(new[] { sequenceLength, totalClasses }) as GpuTensor;

        ErrorCode error;
        var indicesBuffer = CreateBuffer(_context, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
            (IntPtr)(indices.Length * sizeof(int)), indices, out error);
        CheckError(error);

        // Usando o helper ExecuteKernel1D para encapsular a lógica de execução
        ExecuteKernel1D(_oneHotEncodeKernel, sequenceLength, resultTensor, indicesBuffer, totalClasses);

        ReleaseMemObject(indicesBuffer);
        return resultTensor;
    }

    public void AdamUpdate(IMathTensor parameters, IMathTensor gradients, IMathTensor m, IMathTensor v,
        double lr, double beta1, double beta2, double epsilon, int t)
    {
        ExecuteKernel1D(_adamUpdateKernel, parameters.Length,
            (GpuTensor)parameters, (GpuTensor)gradients, (GpuTensor)m, (GpuTensor)v,
            (float)lr, (float)beta1, (float)beta2, (float)epsilon, t);
    }

    // ====================================================================
    // MÉTODOS PRIVADOS AUXILIARES PARA EXECUÇÃO DE KERNELS
    // ====================================================================

    private void ExecuteKernel1D(Kernel kernel, long globalSize, params object[] args)
    {
        SetKernelArgs(kernel, args);
        ErrorCode error = EnqueueNDRangeKernel(_commandQueue, kernel, 1, null, new[] { (IntPtr)globalSize }, null, 0,
            null, out Event kernelEvent);
        CheckError(error, $"Failed to enqueue 1D kernel '{GetKernelInfo(kernel, KernelInfo.FunctionName, out _)}'.");
        ReleaseEvent(kernelEvent);
    }

    private void ExecuteKernel2D(Kernel kernel, long globalSizeX, long globalSizeY, params object[] args)
    {
        SetKernelArgs(kernel, args);
        ErrorCode error = EnqueueNDRangeKernel(_commandQueue, kernel, 2, null,
            new[] { (IntPtr)globalSizeX, (IntPtr)globalSizeY }, null, 0, null, out Event kernelEvent);
        CheckError(error, $"Failed to enqueue 2D kernel '{GetKernelInfo(kernel, KernelInfo.FunctionName, out _)}'.");
        ReleaseEvent(kernelEvent);
    }

    private void SetKernelArgs(Kernel kernel, params object[] args)
    {
        ErrorCode error;
        for (int i = 0; i < args.Length; i++)
        {
            var arg = args[i];
            error = arg switch
            {
                GpuTensor tensor => SetKernelArg(kernel, (uint)i, tensor.Buffer),
                Mem memBuffer => SetKernelArg(kernel, (uint)i, memBuffer),
                int intVal => SetKernelArg(kernel, (uint)i, (uint)intVal),
                uint uintVal => SetKernelArg(kernel, (uint)i, uintVal),
                float floatVal => SetKernelArg(kernel, (uint)i, floatVal),
                _ => throw new ArgumentException(
                    $"Unsupported kernel argument type: {arg?.GetType().Name ?? "null"} at index {i}.")
            };
            CheckError(error,
                $"Failed to set kernel argument at index {i} for kernel '{GetKernelInfo(kernel, KernelInfo.FunctionName, out _)}'.");
        }
    }

    // ====================================================================
    // MÉTODO DISPOSE E HELPERS
    // ====================================================================

    public void Dispose()
    {
        if (_disposed) return;

        // Garante que todos os comandos terminaram antes de liberar os recursos
        Synchronize();

        ReleaseKernel(_matrixMultiplyKernel);
        ReleaseKernel(_vectorMatrixMultiplyKernel);
        ReleaseKernel(_addKernel);
        ReleaseKernel(_addBroadcastKernel);
        ReleaseKernel(_multiplyKernel);
        ReleaseKernel(_sigmoidKernel);
        ReleaseKernel(_tanhKernel);
        ReleaseKernel(_cloneKernel);
        ReleaseKernel(_transposeKernel);
        ReleaseKernel(_subtractKernel);
        ReleaseKernel(_sigmoidDerivativeKernel);
        ReleaseKernel(_tanhDerivativeKernel);
        ReleaseKernel(_matrixMultiplyTransposeAKernel);
        ReleaseKernel(_matrixMultiplyTransposeBKernel);
        ReleaseKernel(_addScaledKernel);
        ReleaseKernel(_subtractScaledKernel);
        ReleaseKernel(_sliceKernel);
        ReleaseKernel(_setKernel);
        ReleaseKernel(_clipKernel);
        ReleaseKernel(_scaleKernel);
        ReleaseKernel(_softmaxKernel);
        ReleaseKernel(_lookupKernel);
        ReleaseKernel(_accumulateGradientKernel);
        ReleaseKernel(_adamUpdateKernel);
        ReleaseKernel(_oneHotEncodeKernel);
        ReleaseProgram(_program);
        ReleaseCommandQueue(_commandQueue);
        ReleaseContext(_context);
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~GpuMathEngine()
    {
        // O finalizador é um fallback caso Dispose() não seja chamado
        Dispose();
    }

    private void CheckError(ErrorCode error, string message = "Erro OpenCL não especificado.")
    {
        if (error != ErrorCode.Success)
        {
            throw new OpenClException($"{message} (Código do erro: {error})", error);
        }
    }
}