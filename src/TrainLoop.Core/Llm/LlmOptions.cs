namespace TrainLoop.Core.Llm;

/// <summary>
/// CPU Qwen settings. Paths may be absolute or relative to the repository root
/// (the host also searches parent directories of its content root).
/// </summary>
public sealed class LlmOptions
{
    public const string SectionName = "Llm";

    public string ModelId { get; set; } = "Qwen/Qwen2.5-0.5B-Instruct";

    /// <summary>
    /// GGUF used by in-process CPU inference. Weights are downloaded, never committed.
    /// </summary>
    public string GgufPath { get; set; } = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    public int ContextSize { get; set; } = 1024;

    /// <summary>0 selects the machine's processor count.</summary>
    public int Threads { get; set; }

    public int DefaultMaxTokens { get; set; } = 128;

    public int MaxTokensLimit { get; set; } = 512;

    /// <summary>transformers-peft (default) or llamafactory.</summary>
    public string Trainer { get; set; } = "transformers-peft";

    public string PythonExecutable { get; set; } = "python3";

    public string TrainerScriptPath { get; set; } = "tools/cpu-trainer/train_qwen_lora.py";

    public string LlamaFactoryExecutable { get; set; } = "llamafactory-cli";

    public string LlamaFactoryConfigPath { get; set; } = "tools/cpu-trainer/llamafactory/qwen25_0.5b_cpu_lora.yaml";

    public string LlamaFactoryDatasetDir { get; set; } = "tools/cpu-trainer/llamafactory/data";

    public string DefaultDatasetPath { get; set; } = "tools/cpu-trainer/sample/corrections.jsonl";

    public string DefaultOutputDir { get; set; } = "saves/qwen2.5-0.5b-cpu-lora";

    public LlmTrainerKind GetTrainerKind()
    {
        var name = (Trainer ?? string.Empty).Trim().ToLowerInvariant().Replace("_", "-");
        return name switch
        {
            "transformers-peft" or "peft" or "transformers" => LlmTrainerKind.TransformersPeft,
            "llamafactory" or "llama-factory" => LlmTrainerKind.LlamaFactory,
            _ => throw new InvalidOperationException(
                $"Unknown Llm:Trainer '{Trainer}'. Use 'transformers-peft' or 'llamafactory'.")
        };
    }
}
