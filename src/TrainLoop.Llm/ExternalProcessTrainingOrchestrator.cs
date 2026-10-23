using System.ComponentModel;
using System.Diagnostics;
using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TrainLoop.Core.Llm;

namespace TrainLoop.Llm;

/// <summary>
/// Starts an external CPU trainer. The default command is the in-repo
/// transformers + peft script. LLaMA-Factory is optional. Neither path trains
/// inside the .NET process.
/// </summary>
public sealed class ExternalProcessTrainingOrchestrator : ITrainingOrchestrator
{
    private readonly LlmOptions _options;
    private readonly string _contentRoot;
    private readonly ILogger<ExternalProcessTrainingOrchestrator> _logger;

    public ExternalProcessTrainingOrchestrator(
        IOptions<LlmOptions> options,
        string contentRoot,
        ILogger<ExternalProcessTrainingOrchestrator> logger)
    {
        _options = options.Value;
        _contentRoot = contentRoot;
        _logger = logger;
    }

    public TrainingPlan Plan(TrainingJobRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        var input = CreateInput(request, out _);
        return CpuTrainingCommandBuilder.Build(input);
    }

    public async Task<TrainingJobResult> RunAsync(
        TrainingJobRequest request,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        var input = CreateInput(request, out var files);
        var plan = CpuTrainingCommandBuilder.Build(input);
        if (request.DryRun)
        {
            return new TrainingJobResult(plan, Started: false, ExitCode: null, StandardOutput: string.Empty, StandardError: string.Empty);
        }

        EnsureReady(files);
        Directory.CreateDirectory(input.OutputDir);

        _logger.LogInformation(
            "Starting CPU trainer {Trainer}. Executable={Executable} Output={Output}",
            plan.Trainer,
            plan.Executable,
            input.OutputDir);

        var startInfo = new ProcessStartInfo
        {
            FileName = plan.Executable,
            WorkingDirectory = plan.WorkingDirectory,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };
        foreach (var argument in plan.Arguments)
            startInfo.ArgumentList.Add(argument);
        foreach (var pair in plan.Environment)
            startInfo.Environment[pair.Key] = pair.Value;

        using var process = new Process { StartInfo = startInfo, EnableRaisingEvents = true };
        try
        {
            if (!process.Start())
                throw new InvalidOperationException($"Failed to start '{plan.Executable}'.");
        }
        catch (Exception ex) when (ex is Win32Exception or FileNotFoundException)
        {
            throw new InvalidOperationException(
                $"Could not start '{plan.Executable}'. Install the CPU trainer dependencies described in the README.",
                ex);
        }

        var stdoutTask = ReadTailAsync(process.StandardOutput, cancellationToken);
        var stderrTask = ReadTailAsync(process.StandardError, cancellationToken);
        try
        {
            await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            TryKill(process);
            throw;
        }

        var stdout = await stdoutTask.ConfigureAwait(false);
        var stderr = await stderrTask.ConfigureAwait(false);
        return new TrainingJobResult(plan, Started: true, process.ExitCode, stdout, stderr);
    }

    private TrainingCommandInput CreateInput(TrainingJobRequest request, out ResolvedFiles files)
    {
        var kind = string.IsNullOrWhiteSpace(request.Trainer)
            ? _options.GetTrainerKind()
            : new LlmOptions { Trainer = request.Trainer }.GetTrainerKind();

        var repoRoot = RepoFileLocator.FindRepoRoot(_contentRoot);
        var modelId = string.IsNullOrWhiteSpace(request.ModelId) ? _options.ModelId : request.ModelId.Trim();
        var output = string.IsNullOrWhiteSpace(request.OutputDir) ? _options.DefaultOutputDir : request.OutputDir.Trim();
        if (!Path.IsPathRooted(output))
            output = Path.GetFullPath(Path.Combine(repoRoot, output));

        string executable;
        string scriptOrConfig;
        string dataset;
        if (kind == LlmTrainerKind.LlamaFactory)
        {
            executable = _options.LlamaFactoryExecutable;
            scriptOrConfig = RepoFileLocator.ResolveFile(_options.LlamaFactoryConfigPath, _contentRoot);
            dataset = ResolveLlamaFactoryDatasetDir(request, repoRoot);
        }
        else
        {
            executable = _options.PythonExecutable;
            scriptOrConfig = RepoFileLocator.ResolveFile(_options.TrainerScriptPath, _contentRoot);
            var requestedDataset = string.IsNullOrWhiteSpace(request.DatasetPath)
                ? _options.DefaultDatasetPath
                : request.DatasetPath.Trim();
            dataset = RepoFileLocator.ResolveFile(requestedDataset, _contentRoot);
        }

        files = new ResolvedFiles(kind, scriptOrConfig, dataset);
        return new TrainingCommandInput(
            kind,
            executable,
            scriptOrConfig,
            modelId,
            dataset,
            output,
            repoRoot,
            request.MaxSteps,
            request.Epochs,
            request.BatchSize,
            request.GradientAccumulationSteps,
            request.MaxLength,
            request.LoraRank,
            request.LoraAlpha <= 0 ? request.LoraRank * 2 : request.LoraAlpha,
            request.LearningRate);
    }

    private string ResolveLlamaFactoryDatasetDir(TrainingJobRequest request, string repoRoot)
    {
        if (!string.IsNullOrWhiteSpace(request.DatasetPath))
        {
            var requested = request.DatasetPath.Trim();
            var asDirectory = RepoFileLocator.ResolveDirectory(requested, _contentRoot);
            if (File.Exists(Path.Combine(asDirectory, "dataset_info.json")))
                return asDirectory;

            if (Directory.Exists(asDirectory) || Path.IsPathRooted(requested))
            {
                throw new ArgumentException(
                    "LLaMA-Factory mode expects DatasetPath to be a directory containing dataset_info.json. " +
                    "For a messages JSONL file, set Llm:Trainer to transformers-peft.");
            }
        }

        var configured = RepoFileLocator.ResolveDirectory(_options.LlamaFactoryDatasetDir, _contentRoot);
        if (Directory.Exists(configured))
            return configured;

        return Path.GetFullPath(Path.Combine(repoRoot, _options.LlamaFactoryDatasetDir));
    }

    private static void EnsureReady(ResolvedFiles files)
    {
        if (!File.Exists(files.ScriptOrConfig))
        {
            throw new FileNotFoundException(
                $"Trainer file not found: {files.ScriptOrConfig}",
                files.ScriptOrConfig);
        }

        if (files.Kind == LlmTrainerKind.TransformersPeft && !File.Exists(files.Dataset))
        {
            throw new FileNotFoundException(
                $"Training JSONL not found: {files.Dataset}",
                files.Dataset);
        }

        if (files.Kind == LlmTrainerKind.LlamaFactory &&
            !File.Exists(Path.Combine(files.Dataset, "dataset_info.json")))
        {
            throw new FileNotFoundException(
                $"dataset_info.json not found under {files.Dataset}",
                Path.Combine(files.Dataset, "dataset_info.json"));
        }
    }

    private static async Task<string> ReadTailAsync(StreamReader reader, CancellationToken cancellationToken)
    {
        const int maxChars = 64 * 1024;
        var buffer = new char[4096];
        var builder = new StringBuilder();
        while (true)
        {
            var read = await reader.ReadAsync(buffer.AsMemory(), cancellationToken).ConfigureAwait(false);
            if (read == 0)
                break;
            builder.Append(buffer, 0, read);
            if (builder.Length > maxChars)
                builder.Remove(0, builder.Length - maxChars);
        }

        return builder.ToString();
    }

    private static void TryKill(Process process)
    {
        try
        {
            if (!process.HasExited)
                process.Kill(entireProcessTree: true);
        }
        catch (InvalidOperationException)
        {
            // Process already exited.
        }
    }

    private readonly record struct ResolvedFiles(LlmTrainerKind Kind, string ScriptOrConfig, string Dataset);
}
