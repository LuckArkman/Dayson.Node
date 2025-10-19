using Galileu.Node.Interfaces;

namespace Galileu.Node.Models;

public record GenerativeModel(
    int inputSize,
    int hiddenSize,
    int outputSize,
    string datasetPath,
    string modelPath,
    ISearchService searchService,
    int contextWindowSize = 5);