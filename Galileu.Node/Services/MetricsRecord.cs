namespace Galileu.Node.Services;

public record MetricsRecord(
    int Epoch,
    int Batch,
    string InputText,
    string ReferenceText,
    string CandidateText,
    double BleuScore,
    double RougeLScore,
    double CosineSimilarity
);