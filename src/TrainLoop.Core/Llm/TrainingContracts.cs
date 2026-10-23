namespace TrainLoop.Core.Llm;

public sealed record TrainingJobRequest(
    string? DatasetPath = null,
    string? ModelId = null,
    string? OutputDir = null,
    string? Trainer = null,
    int MaxSteps = 20,
    double Epochs = 1,
    int BatchSize = 1,
    int GradientAccumulationSteps = 4,
    int MaxLength = 256,
    int LoraRank = 8,
    int LoraAlpha = 16,
    double LearningRate = 1e-4,
    bool DryRun = true);

public sealed record TrainingPlan(
    string Trainer,
    string Executable,
    IReadOnlyList<string> Arguments,
    IReadOnlyDictionary<string, string> Environment,
    string WorkingDirectory,
    string Summary);

public sealed record TrainingJobResult(
    TrainingPlan Plan,
    bool Started,
    int? ExitCode,
    string StandardOutput,
    string StandardError);

public interface ITrainingOrchestrator
{
    TrainingPlan Plan(TrainingJobRequest request);

    Task<TrainingJobResult> RunAsync(TrainingJobRequest request, CancellationToken cancellationToken = default);
}
