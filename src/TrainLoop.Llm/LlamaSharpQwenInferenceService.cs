using LLama;
using LLama.Common;
using LLama.Sampling;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TrainLoop.Core.Llm;

namespace TrainLoop.Llm;

/// <summary>
/// In-process CPU inference for a Qwen2.5 GGUF via LLamaSharp / llama.cpp.
/// GpuLayerCount is always 0. The CUDA backend is not referenced.
/// </summary>
public sealed class LlamaSharpQwenInferenceService : IQwenInferenceService, IDisposable
{
    public const string RuntimeName = "LLamaSharp 0.19 / llama.cpp CPU (LLamaSharp.Backend.Cpu)";

    private readonly LlmOptions _options;
    private readonly string _contentRoot;
    private readonly ILogger<LlamaSharpQwenInferenceService> _logger;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private LLamaWeights? _weights;
    private StatelessExecutor? _executor;
    private string? _loadedPath;

    public LlamaSharpQwenInferenceService(
        IOptions<LlmOptions> options,
        string contentRoot,
        ILogger<LlamaSharpQwenInferenceService> logger)
    {
        _options = options.Value;
        _contentRoot = contentRoot;
        _logger = logger;
    }

    public QwenRuntimeInfo Describe()
    {
        var path = RepoFileLocator.ResolveFile(_options.GgufPath, _contentRoot);
        return new QwenRuntimeInfo(
            RuntimeName,
            _options.ModelId,
            path,
            File.Exists(path),
            GpuLayerCount: 0,
            ContextSize: ClampContext(_options.ContextSize),
            Threads: ResolveThreads(),
            Notes: "纯 CPU。GpuLayerCount 固定为 0，不加载 CUDA backend。此处只服务基础 GGUF，不会加载训练产出的 LoRA adapter。");
    }

    public async Task<QwenInferenceResult> GenerateAsync(
        QwenInferenceRequest request,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        var prompt = QwenChatPrompt.Format(request.SystemPrompt, request.Prompt);
        var maxTokens = request.MaxTokens <= 0 ? _options.DefaultMaxTokens : request.MaxTokens;
        var limit = _options.MaxTokensLimit <= 0 ? 512 : _options.MaxTokensLimit;
        if (maxTokens > limit)
            maxTokens = limit;
        var temperature = Math.Clamp(request.Temperature, 0f, 2f);

        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            await EnsureLoadedAsync(cancellationToken).ConfigureAwait(false);
            var inference = new InferenceParams
            {
                MaxTokens = maxTokens,
                AntiPrompts = new List<string> { QwenChatPrompt.EndToken, "<|endoftext|>" },
                SamplingPipeline = new DefaultSamplingPipeline
                {
                    Temperature = temperature
                }
            };

            _logger.LogInformation(
                "Qwen CPU inference starting. PromptChars={PromptChars} MaxTokens={MaxTokens} Temperature={Temperature}",
                request.Prompt.Length,
                maxTokens,
                temperature);

            var text = new System.Text.StringBuilder();
            await foreach (var chunk in _executor!.InferAsync(prompt, inference, cancellationToken).ConfigureAwait(false))
                text.Append(chunk);

            return new QwenInferenceResult(
                text.ToString().Trim(),
                _loadedPath ?? string.Empty,
                RuntimeName,
                maxTokens,
                temperature);
        }
        catch (Exception ex) when (ex is DllNotFoundException or TypeInitializationException)
        {
            throw new InvalidOperationException(
                "LLamaSharp CPU backend failed to load. Reference LLamaSharp.Backend.Cpu (not a CUDA package) and install libgomp1 on Linux.",
                ex);
        }
        finally
        {
            _gate.Release();
        }
    }

    private async Task EnsureLoadedAsync(CancellationToken cancellationToken)
    {
        var path = RepoFileLocator.ResolveFile(_options.GgufPath, _contentRoot);
        if (!File.Exists(path))
        {
            throw new FileNotFoundException(
                $"Qwen GGUF not found at '{path}'. Download Qwen/Qwen2.5-0.5B-Instruct-GGUF " +
                "(qwen2.5-0.5b-instruct-q4_k_m.gguf) with tools/download-qwen-gguf.sh and set Llm:GgufPath. " +
                "Do not commit the weights.",
                path);
        }

        if (_weights is not null && string.Equals(_loadedPath, path, StringComparison.Ordinal))
            return;

        _weights?.Dispose();
        _executor = null;
        _weights = null;

        var threads = ResolveThreads();
        var parameters = new ModelParams(path)
        {
            ContextSize = (uint)ClampContext(_options.ContextSize),
            GpuLayerCount = 0,
            Threads = threads,
            BatchThreads = threads,
            BatchSize = 128,
            UBatchSize = 128
        };

        _logger.LogInformation(
            "Loading Qwen GGUF on CPU. Path={Path} ContextSize={ContextSize} Threads={Threads} GpuLayerCount=0",
            path,
            parameters.ContextSize,
            threads);

        _weights = await LLamaWeights.LoadFromFileAsync(parameters, cancellationToken).ConfigureAwait(false);
        _executor = new StatelessExecutor(_weights, parameters);
        _loadedPath = path;
    }

    private int ResolveThreads()
    {
        if (_options.Threads > 0)
            return _options.Threads;
        return Math.Max(1, Environment.ProcessorCount);
    }

    private static int ClampContext(int contextSize)
    {
        if (contextSize <= 0)
            return 1024;
        return Math.Clamp(contextSize, 128, 4096);
    }

    public void Dispose()
    {
        _executor = null;
        _weights?.Dispose();
        _weights = null;
        _gate.Dispose();
    }
}
