using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TrainLoop.Core.Llm;

namespace TrainLoop.Llm;

public static class LlmServiceCollectionExtensions
{
    public static IServiceCollection AddTrainLoopLlm(this IServiceCollection services, IConfiguration configuration)
    {
        services.Configure<LlmOptions>(configuration.GetSection(LlmOptions.SectionName));
        services.AddSingleton<IQwenInferenceService>(sp => new LlamaSharpQwenInferenceService(
            sp.GetRequiredService<IOptions<LlmOptions>>(),
            sp.GetRequiredService<IHostEnvironment>().ContentRootPath,
            sp.GetRequiredService<ILogger<LlamaSharpQwenInferenceService>>()));
        services.AddSingleton<ITrainingOrchestrator>(sp => new ExternalProcessTrainingOrchestrator(
            sp.GetRequiredService<IOptions<LlmOptions>>(),
            sp.GetRequiredService<IHostEnvironment>().ContentRootPath,
            sp.GetRequiredService<ILogger<ExternalProcessTrainingOrchestrator>>()));
        return services;
    }
}
