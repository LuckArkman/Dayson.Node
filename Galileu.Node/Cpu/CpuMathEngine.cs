using System;
using Galileu.Node.Core;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Cpu;

public class CpuMathEngine : IMathEngine
{
    public bool IsGpu => false;

    public IMathTensor CreateTensor(int[] shape)
    {
        int size = 1;
        foreach (int dim in shape) size *= dim;
        // ALTERADO: Cria CpuTensor com array de float.
        return new CpuTensor(new float[size], shape);
    }

    // ALTERADO: Aceita float[] e clona.
    public IMathTensor CreateTensor(float[] hostData, int[] shape)
    {
        return new CpuTensor((float[])hostData.Clone(), shape);
    }
    
    public IMathTensor CreateOneHotTensor(int[] indices, int totalClasses)
    {
        int sequenceLength = indices.Length;
        // ALTERADO: O array de dados é float.
        var flatData = new float[sequenceLength * totalClasses];
        for (int i = 0; i < sequenceLength; i++)
        {
            int targetIndex = indices[i];
            if (targetIndex >= 0 && targetIndex < totalClasses)
            {
                flatData[i * totalClasses + targetIndex] = 1.0f;
            }
        }
        return new CpuTensor(flatData, new[] { sequenceLength, totalClasses });
    }

    // ALTERADO: Parâmetros e cálculos internos usam float.
    public void AdamUpdate(IMathTensor parameters, IMathTensor gradients, IMathTensor m_tensor, IMathTensor v_tensor, 
        float lr, float beta1, float beta2, float epsilon, int t)
    {
        var p_data = ((CpuTensor)parameters).GetData();
        var g_data = ((CpuTensor)gradients).GetData();
        var m_data = ((CpuTensor)m_tensor).GetData();
        var v_data = ((CpuTensor)v_tensor).GetData();
    
        float beta1_pow_t = MathF.Pow(beta1, t);
        float beta2_pow_t = MathF.Pow(beta2, t);

        for (int i = 0; i < p_data.Length; i++)
        {
            m_data[i] = beta1 * m_data[i] + (1 - beta1) * g_data[i];
            v_data[i] = beta2 * v_data[i] + (1 - beta2) * (g_data[i] * g_data[i]);
        
            float m_hat = m_data[i] / (1 - beta1_pow_t);
            float v_hat = v_data[i] / (1 - beta2_pow_t);
        
            p_data[i] -= lr * m_hat / (MathF.Sqrt(v_hat) + epsilon);
        }
    }

    // ... (Todas as outras operações matemáticas foram atualizadas para float) ...

    public void Add(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        var A = ((CpuTensor)a).GetData(); var B = ((CpuTensor)b).GetData(); var C = ((CpuTensor)result).GetData();
        for (int i = 0; i < A.Length; i++) C[i] = A[i] + B[i];
    }
    public void AddBroadcast(IMathTensor matrix, IMathTensor vector, IMathTensor result)
    {
        var A = ((CpuTensor)matrix).GetData(); var B = ((CpuTensor)vector).GetData(); var C = ((CpuTensor)result).GetData();
        int M = matrix.Shape[0]; int N = matrix.Shape[1];
        for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) C[i * N + j] = A[i * N + j] + B[j];
    }
    public void MatrixMultiply(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        var A = ((CpuTensor)a).GetData(); var B = ((CpuTensor)b).GetData(); var C = ((CpuTensor)result).GetData();
        int M = a.Shape[0]; int K = a.Shape[1]; int N = b.Shape[1];
        Array.Clear(C, 0, C.Length);
        for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { float sum = 0; for (int k = 0; k < K; k++) sum += A[i * K + k] * B[k * N + j]; C[i * N + j] = sum; }
    }
    public void Multiply(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        var A = ((CpuTensor)a).GetData(); var B = ((CpuTensor)b).GetData(); var C = ((CpuTensor)result).GetData();
        for (int i = 0; i < A.Length; i++) C[i] = A[i] * B[i];
    }
    public void Sigmoid(IMathTensor input, IMathTensor result)
    {
        var I = ((CpuTensor)input).GetData(); var O = ((CpuTensor)result).GetData();
        for (int i = 0; i < I.Length; i++) O[i] = 1.0f / (1.0f + MathF.Exp(-I[i]));
    }
    public void Tanh(IMathTensor input, IMathTensor result)
    {
        var I = ((CpuTensor)input).GetData(); var O = ((CpuTensor)result).GetData();
        for (int i = 0; i < I.Length; i++) O[i] = MathF.Tanh(I[i]);
    }
    public IMathTensor Clone(IMathTensor tensor)
    {
        var cpuTensor = (CpuTensor)tensor;
        return new CpuTensor((float[])cpuTensor.GetData().Clone(), (int[])cpuTensor.Shape.Clone());
    }
    public void Transpose(IMathTensor input, IMathTensor result)
    {
        var I = ((CpuTensor)input).GetData(); var O = ((CpuTensor)result).GetData();
        int rows = input.Shape[0]; int cols = input.Shape[1];
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) O[j * rows + i] = I[i * cols + j];
    }
    public void Subtract(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        var A = ((CpuTensor)a).GetData(); var B = ((CpuTensor)b).GetData(); var C = ((CpuTensor)result).GetData();
        for (int i = 0; i < A.Length; i++) C[i] = A[i] - B[i];
    }
    public void SigmoidDerivative(IMathTensor output, IMathTensor result)
    {
        var O = ((CpuTensor)output).GetData(); var R = ((CpuTensor)result).GetData();
        for (int i = 0; i < O.Length; i++) R[i] = O[i] * (1.0f - O[i]);
    }
    public void TanhDerivative(IMathTensor output, IMathTensor result)
    {
        var O = ((CpuTensor)output).GetData(); var R = ((CpuTensor)result).GetData();
        for (int i = 0; i < O.Length; i++) R[i] = 1.0f - O[i] * O[i];
    }
    public void MatrixMultiplyTransposeA(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        var A = ((CpuTensor)a).GetData(); var B = ((CpuTensor)b).GetData(); var C = ((CpuTensor)result).GetData();
        int K = a.Shape[0]; int M = a.Shape[1]; int N = b.Shape[1];
        Array.Clear(C, 0, C.Length);
        for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { float sum = 0; for (int k = 0; k < K; k++) { sum += A[k * M + i] * B[k * N + j]; } C[i * N + j] = sum; }
    }
    public void MatrixMultiplyTransposeB(IMathTensor a, IMathTensor b, IMathTensor result)
    {
        var A = ((CpuTensor)a).GetData(); var B = ((CpuTensor)b).GetData(); var C = ((CpuTensor)result).GetData();
        int M = a.Shape[0]; int K = a.Shape[1]; int N = b.Shape[0];
        Array.Clear(C, 0, C.Length);
        for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { float sum = 0; for (int k = 0; k < K; k++) { sum += A[i * K + k] * B[j * K + k]; } C[i * N + j] = sum; }
    }
    public void AddScaled(IMathTensor target, IMathTensor source, float scalar)
    {
        var T = ((CpuTensor)target).GetData(); var S = ((CpuTensor)source).GetData();
        for (int i = 0; i < T.Length; i++) T[i] += S[i] * scalar;
    }
    public void SubtractScaled(IMathTensor target, IMathTensor source, float scalar)
    {
        var T = ((CpuTensor)target).GetData(); var S = ((CpuTensor)source).GetData();
        for (int i = 0; i < T.Length; i++) T[i] -= S[i] * scalar;
    }
    public void Slice(IMathTensor source, int rowIndex, IMathTensor destination)
    {
        var srcData = ((CpuTensor)source).GetData(); var destData = ((CpuTensor)destination).GetData();
        int featureSize = destination.Shape[1]; int offset = rowIndex * featureSize;
        Array.Copy(srcData, offset, destData, 0, featureSize);
    }
    public void Set(IMathTensor destination, int rowIndex, IMathTensor source)
    {
        var srcData = ((CpuTensor)source).GetData(); var destData = ((CpuTensor)destination).GetData();
        int featureSize = source.Shape[1]; int offset = rowIndex * featureSize;
        Array.Copy(srcData, 0, destData, offset, featureSize);
    }
    public void Clip(IMathTensor tensor, float minValue, float maxValue)
    {
        var data = ((CpuTensor)tensor).GetData();
        for (int i = 0; i < data.Length; i++) { if (data[i] < minValue) data[i] = minValue; else if (data[i] > maxValue) data[i] = maxValue; }
    }
    public void Scale(IMathTensor tensor, float scalar)
    {
        var data = ((CpuTensor)tensor).GetData();
        for (int i = 0; i < data.Length; i++) data[i] *= scalar;
    }
    public void Softmax(IMathTensor input, IMathTensor result)
    {
        var inputData = ((CpuTensor)input).GetData(); var outputData = ((CpuTensor)result).GetData();
        int rows = input.Shape[0]; int cols = input.Shape[1];
        for (int row = 0; row < rows; row++)
        {
            int offset = row * cols; float maxVal = inputData[offset];
            for (int i = 1; i < cols; i++) if (inputData[offset + i] > maxVal) maxVal = inputData[offset + i];
            float sumExp = 0;
            for (int i = 0; i < cols; i++) { outputData[offset + i] = MathF.Exp(inputData[offset + i] - maxVal); sumExp += outputData[offset + i]; }
            for (int i = 0; i < cols; i++) outputData[offset + i] /= sumExp;
        }
    }
    public void Lookup(IMathTensor embeddingMatrix, int index, IMathTensor result)
    {
        var matrixData = ((CpuTensor)embeddingMatrix).GetData(); var resultData = ((CpuTensor)result).GetData();
        int embeddingSize = embeddingMatrix.Shape[1]; int offset = index * embeddingSize;
        Array.Copy(matrixData, offset, resultData, 0, embeddingSize);
    }
    public void AccumulateGradient(IMathTensor embeddingGradients, IMathTensor gradient, int index)
    {
        var embeddingGradData = ((CpuTensor)embeddingGradients).GetData(); var gradData = ((CpuTensor)gradient).GetData();
        int embeddingSize = embeddingGradients.Shape[1]; int offset = index * embeddingSize;
        for (int i = 0; i < embeddingSize; i++) embeddingGradData[offset + i] += gradData[i];
    }
    public void Dispose() { }
}