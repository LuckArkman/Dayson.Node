namespace Galileu.Node.Models;

public record Trainer(string datasetPath,int epochs, double learningRate, double validationSplit, int batchSize);