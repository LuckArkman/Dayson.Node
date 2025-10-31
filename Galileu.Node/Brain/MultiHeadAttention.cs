using System;
using System.Collections.Generic;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Brain;

public class MultiHeadAttention : IDisposable
{
    public readonly IMathEngine _mathEngine;
    public readonly int _embeddingSize;
    public readonly int _numHeads;
    public readonly int _headSize;
    public bool _disposed = false;

    public IMathTensor Wq { get; set; }
    public IMathTensor Wk { get; set; }
    public IMathTensor Wv { get; set; }
    public IMathTensor Wo { get; set; }

    public MultiHeadAttention(int embeddingSize, int numHeads, IMathEngine mathEngine, Random rand)
    {
        if (embeddingSize % numHeads != 0)
        {
            throw new ArgumentException("O tamanho do embedding deve ser divisível pelo número de cabeças.");
        }

        _mathEngine = mathEngine;
        _embeddingSize = embeddingSize;
        _numHeads = numHeads;
        _headSize = embeddingSize / numHeads;

        Wq = InitializeTensor(embeddingSize, embeddingSize, rand);
        Wk = InitializeTensor(embeddingSize, embeddingSize, rand);
        Wv = InitializeTensor(embeddingSize, embeddingSize, rand);
        Wo = InitializeTensor(embeddingSize, embeddingSize, rand);
    }

    public (IMathTensor output, IMathTensor attentionWeights) Forward(IMathTensor input, TensorPool pool)
    {
        int sequenceLength = input.Shape[0];
        var q = pool.Rent(new[] { sequenceLength, _embeddingSize });
        var k = pool.Rent(new[] { sequenceLength, _embeddingSize });
        var v = pool.Rent(new[] { sequenceLength, _embeddingSize });
        _mathEngine.MatrixMultiply(input, Wq, q);
        _mathEngine.MatrixMultiply(input, Wk, k);
        _mathEngine.MatrixMultiply(input, Wv, v);

        var scores = pool.Rent(new[] { sequenceLength, sequenceLength });
        _mathEngine.MatrixMultiplyTransposeB(q, k, scores);
        _mathEngine.Scale(scores, 1.0f / MathF.Sqrt(_headSize));

        var attentionWeights = pool.Rent(new[] { sequenceLength, sequenceLength });
        _mathEngine.Softmax(scores, attentionWeights);

        var context = pool.Rent(new[] { sequenceLength, _embeddingSize });
        _mathEngine.MatrixMultiply(attentionWeights, v, context);

        var output = pool.Rent(new[] { sequenceLength, _embeddingSize });
        _mathEngine.MatrixMultiply(context, Wo, output);

        pool.Return(q);
        pool.Return(k);
        pool.Return(v);
        pool.Return(scores);
        pool.Return(context);

        var finalOutput = _mathEngine.Clone(output);
        var finalWeights = _mathEngine.Clone(attentionWeights);
        pool.Return(output);
        pool.Return(attentionWeights);

        return (finalOutput, finalWeights);
    }
    
    public IMathTensor Backward(IMathTensor dOutput, IMathTensor originalInput, IMathTensor attentionWeights, Dictionary<string, IMathTensor> grads, TensorPool pool)
    {
        var dContext = pool.Rent(dOutput.Shape);
        _mathEngine.MatrixMultiplyTransposeB(dOutput, Wo, dContext);
        
        var dWo = pool.Rent(Wo.Shape);
        var context = pool.Rent(dOutput.Shape);
        var v = pool.Rent(originalInput.Shape);
        _mathEngine.MatrixMultiply(attentionWeights, v, context);
        _mathEngine.MatrixMultiplyTransposeA(context, dOutput, dWo);
        _mathEngine.Add(grads["w_att_o"], dWo, grads["w_att_o"]);

        var dAttentionWeights = pool.Rent(attentionWeights.Shape);
        _mathEngine.MatrixMultiplyTransposeB(dContext, v, dAttentionWeights);

        var dScores = pool.Rent(attentionWeights.Shape);
        //_mathEngine.SoftmaxDerivative(attentionWeights, dAttentionWeights, dScores);

        _mathEngine.Scale(dScores, 1.0f / MathF.Sqrt(_headSize));
        
        var q = pool.Rent(originalInput.Shape);
        var k = pool.Rent(originalInput.Shape);
        _mathEngine.MatrixMultiply(originalInput, Wq, q);
        _mathEngine.MatrixMultiply(originalInput, Wk, k);
        
        var dQ = pool.Rent(q.Shape);
        var dK = pool.Rent(k.Shape);
        _mathEngine.MatrixMultiply(dScores, k, dQ);
        _mathEngine.MatrixMultiplyTransposeA(dScores, q, dK);

        var dV = pool.Rent(v.Shape);
        _mathEngine.MatrixMultiplyTransposeA(attentionWeights, dContext, dV);

        var dWq = pool.Rent(Wq.Shape);
        var dWk = pool.Rent(Wk.Shape);
        var dWv = pool.Rent(Wv.Shape);
        _mathEngine.MatrixMultiplyTransposeA(originalInput, dQ, dWq);
        _mathEngine.MatrixMultiplyTransposeA(originalInput, dK, dWk);
        _mathEngine.MatrixMultiplyTransposeA(originalInput, dV, dWv);
        _mathEngine.Add(grads["w_att_q"], dWq, grads["w_att_q"]);
        _mathEngine.Add(grads["w_att_k"], dWk, grads["w_att_k"]);
        _mathEngine.Add(grads["w_att_v"], dWv, grads["w_att_v"]);

        var dInput = pool.Rent(originalInput.Shape);
        var tempGrad = pool.Rent(originalInput.Shape);
        _mathEngine.MatrixMultiplyTransposeB(dQ, Wq, dInput);
        _mathEngine.MatrixMultiplyTransposeB(dK, Wk, tempGrad);
        _mathEngine.Add(dInput, tempGrad, dInput);
        _mathEngine.MatrixMultiplyTransposeB(dV, Wv, tempGrad);
        _mathEngine.Add(dInput, tempGrad, dInput);
        
        pool.Return(dContext); pool.Return(dWo); pool.Return(context); pool.Return(v);
        pool.Return(dAttentionWeights); pool.Return(dScores); pool.Return(q); pool.Return(k);
        pool.Return(dQ); pool.Return(dK); pool.Return(dV);
        pool.Return(dWq); pool.Return(dWk); pool.Return(dWv);
        pool.Return(tempGrad);

        var finalDInput = _mathEngine.Clone(dInput);
        pool.Return(dInput);
        return finalDInput;
    }

    private IMathTensor InitializeTensor(int rows, int cols, Random rand)
    {
        float[] data = new float[rows * cols];
        float limit = MathF.Sqrt(6.0f / (rows + cols));
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (rand.NextSingle() * 2 - 1) * limit;
        }
        return _mathEngine.CreateTensor(data, new[] { rows, cols });
    }

    public void Dispose()
    {
        if (_disposed) return;
        Wq?.Dispose();
        Wk?.Dispose();
        Wv?.Dispose();
        Wo?.Dispose();
        _disposed = true;
    }
}