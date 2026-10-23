namespace TrainLoop.Core.Llm;

public sealed record QwenInferenceRequest(
    string Prompt,
    string? SystemPrompt = null,
    int MaxTokens = 128,
    float Temperature = 0.7f);

public sealed record QwenInferenceResult(
    string Text,
    string ModelPath,
    string Runtime,
    int MaxTokens,
    float Temperature);

public sealed record QwenRuntimeInfo(
    string Stack,
    string ModelId,
    string ModelPath,
    bool ModelPresent,
    int GpuLayerCount,
    int ContextSize,
    int Threads,
    string Notes);

public interface IQwenInferenceService
{
    QwenRuntimeInfo Describe();

    Task<QwenInferenceResult> GenerateAsync(QwenInferenceRequest request, CancellationToken cancellationToken = default);
}
