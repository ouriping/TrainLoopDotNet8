using TrainLoop.Core.Llm;

namespace TrainLoop.Api;

public static class LlmEndpoints
{
    public static IEndpointRouteBuilder MapLlmEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapGet("/api/llm/runtime", (IQwenInferenceService inference) => Results.Ok(inference.Describe()))
            .WithName("GetLlmRuntime");

        app.MapPost("/api/llm/infer", async (
            QwenInferenceRequest request,
            IQwenInferenceService inference,
            CancellationToken cancellationToken) =>
        {
            try
            {
                var result = await inference.GenerateAsync(request, cancellationToken);
                return Results.Ok(result);
            }
            catch (Exception ex) when (ex is ArgumentException or FileNotFoundException or InvalidOperationException)
            {
                return Results.BadRequest(new { error = ex.Message });
            }
        }).WithName("InferQwen");

        app.MapPost("/api/llm/train/plan", (TrainingJobRequest request, ITrainingOrchestrator orchestrator) =>
        {
            try
            {
                return Results.Ok(orchestrator.Plan(request));
            }
            catch (Exception ex) when (ex is ArgumentException or InvalidOperationException)
            {
                return Results.BadRequest(new { error = ex.Message });
            }
        }).WithName("PlanQwenTraining");

        app.MapPost("/api/llm/train", async (
            TrainingJobRequest request,
            ITrainingOrchestrator orchestrator,
            CancellationToken cancellationToken) =>
        {
            try
            {
                var result = await orchestrator.RunAsync(request, cancellationToken);
                if (result.Started && result.ExitCode is not 0)
                    return Results.Json(result, statusCode: StatusCodes.Status500InternalServerError);
                return Results.Ok(result);
            }
            catch (Exception ex) when (ex is ArgumentException or FileNotFoundException or InvalidOperationException)
            {
                return Results.BadRequest(new { error = ex.Message });
            }
        }).WithName("TrainQwen");

        return app;
    }
}
