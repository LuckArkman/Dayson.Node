namespace Galileu.Node.Interfaces;

public interface IMathEngine : IDisposable
{
    bool IsGpu { get; }

    IMathTensor CreateTensor(int[] shape);
    IMathTensor CreateTensor(double[] hostData, int[] shape);
    
    void MatrixMultiply(IMathTensor a, IMathTensor b, IMathTensor result);
    void Add(IMathTensor a, IMathTensor b, IMathTensor result);
    void AddBroadcast(IMathTensor matrix, IMathTensor vector, IMathTensor result);
    void Multiply(IMathTensor a, IMathTensor b, IMathTensor result);
    void Sigmoid(IMathTensor input, IMathTensor result);
    void Tanh(IMathTensor input, IMathTensor result);
    void Slice(IMathTensor source, int rowIndex, IMathTensor destination);
    void Set(IMathTensor destination, int rowIndex, IMathTensor source);
    void Clip(IMathTensor tensor, double minValue, double maxValue);
    
    void Scale(IMathTensor tensor, double scalar);
    
    void Softmax(IMathTensor input, IMathTensor result);
    IMathTensor Clone(IMathTensor tensor);
    void Transpose(IMathTensor input, IMathTensor result);
    void Subtract(IMathTensor a, IMathTensor b, IMathTensor result);
    void SigmoidDerivative(IMathTensor output, IMathTensor result);
    void TanhDerivative(IMathTensor output, IMathTensor result);
    void MatrixMultiplyTransposeA(IMathTensor a, IMathTensor b, IMathTensor result);
    void MatrixMultiplyTransposeB(IMathTensor a, IMathTensor b, IMathTensor result);
    void AddScaled(IMathTensor target, IMathTensor source, double scalar);
    void SubtractScaled(IMathTensor target, IMathTensor source, double scalar);
    void Lookup(IMathTensor embeddingMatrix, int index, IMathTensor result);
    void AccumulateGradient(IMathTensor embeddingGradients, IMathTensor gradient, int index);
    IMathTensor? CreateOneHotTensor(int[] targetIndices, int outputSize);
    void AdamUpdate(IMathTensor parameters, IMathTensor gradients, IMathTensor m, IMathTensor v, 
        double lr, double beta1, double beta2, double epsilon, int t);
}