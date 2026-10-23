using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;

namespace TrainLoop.Api.Tests;

public class LlmEndpointTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;

    public LlmEndpointTests(WebApplicationFactory<Program> factory)
    {
        _factory = factory;
    }

    [Fact]
    public async Task Runtime_ReportsCpuStack()
    {
        var client = _factory.CreateClient();
        var body = await client.GetStringAsync("/api/llm/runtime");

        Assert.Contains("LLamaSharp", body);
        Assert.Contains("\"gpuLayerCount\":0", body);
    }

    [Fact]
    public async Task Train_DryRun_ReturnsCpuPlan()
    {
        var client = _factory.CreateClient();
        var response = await client.PostAsJsonAsync("/api/llm/train", new
        {
            dryRun = true,
            maxSteps = 20,
            batchSize = 1
        });

        response.EnsureSuccessStatusCode();
        var body = await response.Content.ReadAsStringAsync();
        Assert.Contains("train_qwen_lora.py", body);
        Assert.Contains("CUDA_VISIBLE_DEVICES", body);
        Assert.Contains("\"started\":false", body);
    }
}
