using Galileu.Node.Interfaces;

namespace Galileu.Node.Core;

public class Tensor : IMathTensor
{
    private readonly float[] data;
    public readonly int[] shape;

    public Tensor(float[] data, int[] shape)
    {
        this.data = data ?? throw new ArgumentNullException(nameof(data));
        this.shape = shape ?? throw new ArgumentNullException(nameof(shape));

        int expectedSize = 1;
        foreach (int dim in shape)
        {
            if (dim <= 0)
            {
                throw new ArgumentException("As dimensões do shape devem ser positivas.");
            }
            expectedSize *= dim;
        }

        if (data.Length != expectedSize)
        {
            throw new ArgumentException(
                $"O tamanho dos dados ({data.Length}) não corresponde às dimensões do shape ({string.Join("x", shape)}), esperado {expectedSize}.");
        }
    }

    public float Infer(int[] indices)
    {
        if (indices == null || indices.Length != shape.Length)
        {
            throw new ArgumentException("Os índices fornecidos não correspondem às dimensões do tensor.");
        }

        for (int i = 0; i < indices.Length; i++)
        {
            if (indices[i] < 0 || indices[i] >= shape[i])
            {
                throw new ArgumentOutOfRangeException(nameof(indices),
                    $"Índice {indices[i]} fora dos limites para a dimensão {i} (0 a {shape[i] - 1}).");
            }
        }

        int flatIndex = 0;
        int stride = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            flatIndex += indices[i] * stride;
            stride *= shape[i];
        }

        return data[flatIndex];
    }

    public float[] GetData() => (float[])data.Clone(); // Return a copy to prevent external modification
    public int[] Shape => (int[])shape.Clone(); // Return a copy to prevent external modification
    public long Length { get; }

    Tensor IMathTensor.ToCpuTensor()
    {
        return this;
    }

    public void UpdateFromCpu(float[] data)
    {
        if (data == null) throw new ArgumentNullException(nameof(data));
        if (data.Length != this.data.Length) throw new ArgumentException("Tamanho dos dados incompatível.");
        Array.Copy(data, this.data, data.Length);
    }

    public void WriteToStream(BinaryWriter writer)
    {
        throw new NotImplementedException();
    }

    public bool IsGpu => false; // This is a CPU-based tensor implementation

    public void Dispose() { } // Placeholder for IDisposable (no unmanaged resources here)

    public int[] GetShape() => shape;
}