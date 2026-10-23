using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using TrainLoop.Core.Llm;
using TrainLoop.Llm;

namespace TrainLoop.Llm.Tests;

public class CpuOrchestratorTests
{
    [Fact]
    public async Task DryRun_DoesNotStartMissingPython()
    {
        var orchestrator = new ExternalProcessTrainingOrchestrator(
            Options.Create(new LlmOptions
            {
                PythonExecutable = "this-python-does-not-exist",
                Trainer = "transformers-peft"
            }),
            RepoRoot(),
            NullLogger<ExternalProcessTrainingOrchestrator>.Instance);

        var result = await orchestrator.RunAsync(new TrainingJobRequest(DryRun: true, MaxSteps: 20, BatchSize: 1));

        Assert.False(result.Started);
        Assert.Null(result.ExitCode);
        Assert.Contains("train_qwen_lora.py", string.Join(" ", result.Plan.Arguments));
        Assert.Equal(string.Empty, result.Plan.Environment["CUDA_VISIBLE_DEVICES"]);
        Assert.Equal(0, result.Plan.Arguments.Count(argument => argument.Contains("cuda", StringComparison.OrdinalIgnoreCase)));
    }

    [Fact]
    public void Describe_ReportsCpuAndDoesNotRequireWeights()
    {
        var service = new LlamaSharpQwenInferenceService(
            Options.Create(new LlmOptions
            {
                GgufPath = "models/missing-qwen.gguf"
            }),
            RepoRoot(),
            NullLogger<LlamaSharpQwenInferenceService>.Instance);

        var info = service.Describe();

        Assert.Equal(0, info.GpuLayerCount);
        Assert.False(info.ModelPresent);
        Assert.Contains("LLamaSharp", info.Stack);
        Assert.Contains("CPU", info.Stack, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void TrainerScript_DryRun_PrintsCpu()
    {
        var script = Path.Combine(RepoRoot(), "tools", "cpu-trainer", "train_qwen_lora.py");
        var dataset = Path.Combine(RepoRoot(), "tools", "cpu-trainer", "sample", "corrections.jsonl");
        Assert.True(File.Exists(script));
        Assert.True(File.Exists(dataset));

        using var process = new System.Diagnostics.Process
        {
            StartInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = "python3",
                WorkingDirectory = RepoRoot(),
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false
            }
        };
        process.StartInfo.ArgumentList.Add(script);
        process.StartInfo.ArgumentList.Add("--dry-run");
        process.StartInfo.ArgumentList.Add("--dataset");
        process.StartInfo.ArgumentList.Add(dataset);
        process.StartInfo.ArgumentList.Add("--output");
        process.StartInfo.ArgumentList.Add(Path.Combine(Path.GetTempPath(), "trainloop-dry-run"));
        process.StartInfo.Environment["CUDA_VISIBLE_DEVICES"] = string.Empty;

        Assert.True(process.Start());
        var stdout = process.StandardOutput.ReadToEnd();
        var stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, stderr);
        Assert.Contains("\"device\": \"cpu\"", stdout);
        Assert.Contains("\"dry_run\": true", stdout);
    }

    private static string RepoRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "tools", "cpu-trainer", "train_qwen_lora.py")))
                return directory.FullName;
            directory = directory.Parent;
        }

        throw new InvalidOperationException("Repository root was not found from the test output directory.");
    }
}
