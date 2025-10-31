using Galileu.Node.Core;

namespace Galileu.Node.Brain;

public class NeuralNetworkModelDataEmbeddingLSTM
{
    // Hiperparâmetros da arquitetura
    public int VocabSize { get; set; }
    public int EmbeddingSize { get; set; }
    public int HiddenSize { get; set; }
    public int OutputSize { get; set; }

    // Camadas de Pesos e Biases
    public TensorData? WeightsEmbedding { get; set; }
    public TensorData? WeightsInputForget { get; set; }
    public TensorData? WeightsHiddenForget { get; set; }
    public TensorData? WeightsInputInput { get; set; }
    public TensorData? WeightsHiddenInput { get; set; }
    public TensorData? WeightsInputCell { get; set; }
    public TensorData? WeightsHiddenCell { get; set; }
    public TensorData? WeightsInputOutput { get; set; }
    public TensorData? WeightsHiddenOutput { get; set; }
    public TensorData? BiasForget { get; set; }
    public TensorData? BiasInput { get; set; }
    public TensorData? BiasCell { get; set; }
    public TensorData? BiasOutput { get; set; }
    public TensorData? WeightsHiddenOutputFinal { get; set; }
    public TensorData? BiasOutputFinal { get; set; }
}