using Galileu.Node.Core;

namespace Galileu.Node.Brain;

public static class TensorExtensions
{
    public static TensorData ToTensorData(this Tensor tensor)
    {
        return new TensorData { data = tensor.GetData(), shape = tensor.GetShape() };
    }
}