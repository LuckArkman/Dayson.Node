namespace Galileu.Node.Core;

public class Tensor
{
    private readonly double[] data;
    public readonly int[] shape;

    public Tensor(double[] data, int[] shape)
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

    public double Infer(int[] indices)
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

    public double[] GetData() => data;
    public int[] GetShape() => shape;

    public int GetTotalSize()
    {
        int size = 1;
        foreach (int dim in shape)
        {
            size *= dim;
        }

        return size;
    }
}