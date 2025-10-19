using Galileu.Node.Core;

namespace Galileu.Node.Interfaces;

public interface IMathTensor : IDisposable
{
    int[] Shape { get; }
    long Length { get; }
    bool IsGpu { get; }
    Tensor ToCpuTensor();
    void UpdateFromCpu(double[] data);
    void WriteToStream(BinaryWriter writer);
}