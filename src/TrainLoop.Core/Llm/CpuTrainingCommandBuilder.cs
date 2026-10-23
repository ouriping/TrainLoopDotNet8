using System.Globalization;

namespace TrainLoop.Core.Llm;

public sealed record TrainingCommandInput(
    LlmTrainerKind Trainer,
    string Executable,
    string ScriptOrConfigPath,
    string ModelId,
    string DatasetPath,
    string OutputDir,
    string WorkingDirectory,
    int MaxSteps,
    double Epochs,
    int BatchSize,
    int GradientAccumulationSteps,
    int MaxLength,
    int LoraRank,
    int LoraAlpha,
    double LearningRate);

public static class CpuTrainingCommandBuilder
{
    public static TrainingPlan Build(TrainingCommandInput input)
    {
        ArgumentNullException.ThrowIfNull(input);
        Validate(input);

        var environment = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["CUDA_VISIBLE_DEVICES"] = string.Empty,
            ["HIP_VISIBLE_DEVICES"] = string.Empty,
            ["TOKENIZERS_PARALLELISM"] = "false"
        };

        if (input.Trainer == LlmTrainerKind.LlamaFactory)
        {
            var arguments = new List<string>
            {
                "train",
                input.ScriptOrConfigPath,
                $"model_name_or_path={input.ModelId}",
                "dataset=trainloop_corrections",
                $"dataset_dir={input.DatasetPath}",
                $"output_dir={input.OutputDir}",
                $"per_device_train_batch_size={input.BatchSize.ToString(CultureInfo.InvariantCulture)}",
                $"gradient_accumulation_steps={input.GradientAccumulationSteps.ToString(CultureInfo.InvariantCulture)}",
                $"cutoff_len={input.MaxLength.ToString(CultureInfo.InvariantCulture)}",
                $"learning_rate={input.LearningRate.ToString(CultureInfo.InvariantCulture)}",
                $"lora_rank={input.LoraRank.ToString(CultureInfo.InvariantCulture)}",
                $"lora_alpha={input.LoraAlpha.ToString(CultureInfo.InvariantCulture)}",
                "bf16=false",
                "fp16=false",
                "flash_attn=disabled",
                "optim=adamw_torch",
                "preprocessing_num_workers=1",
                "dataloader_num_workers=0",
                "report_to=none"
            };

            if (input.MaxSteps > 0)
                arguments.Add($"max_steps={input.MaxSteps.ToString(CultureInfo.InvariantCulture)}");
            else
                arguments.Add($"num_train_epochs={input.Epochs.ToString(CultureInfo.InvariantCulture)}");

            return new TrainingPlan(
                "llamafactory",
                input.Executable,
                arguments,
                environment,
                input.WorkingDirectory,
                "LLaMA-Factory LoRA SFT，进程外执行。CUDA_VISIBLE_DEVICES 被清空，bf16/fp16 强制关闭，不使用 NVIDIA GPU。");
        }

        var scriptArguments = new List<string>
        {
            input.ScriptOrConfigPath,
            "--model", input.ModelId,
            "--dataset", input.DatasetPath,
            "--output", input.OutputDir,
            "--batch-size", input.BatchSize.ToString(CultureInfo.InvariantCulture),
            "--grad-accum", input.GradientAccumulationSteps.ToString(CultureInfo.InvariantCulture),
            "--max-length", input.MaxLength.ToString(CultureInfo.InvariantCulture),
            "--lora-rank", input.LoraRank.ToString(CultureInfo.InvariantCulture),
            "--lora-alpha", input.LoraAlpha.ToString(CultureInfo.InvariantCulture),
            "--learning-rate", input.LearningRate.ToString(CultureInfo.InvariantCulture),
            "--epochs", input.Epochs.ToString(CultureInfo.InvariantCulture),
            "--max-steps", input.MaxSteps.ToString(CultureInfo.InvariantCulture)
        };

        return new TrainingPlan(
            "transformers-peft",
            input.Executable,
            scriptArguments,
            environment,
            input.WorkingDirectory,
            "transformers + peft LoRA，由 C# 启动外部 Python，脚本内强制 device=cpu。这不是进程内 C# 训练。");
    }

    private static void Validate(TrainingCommandInput input)
    {
        RequireSafePath(input.Executable, nameof(input.Executable));
        RequireSafePath(input.ScriptOrConfigPath, nameof(input.ScriptOrConfigPath));
        RequireSafePath(input.ModelId, nameof(input.ModelId));
        RequireSafePath(input.DatasetPath, nameof(input.DatasetPath));
        RequireSafePath(input.OutputDir, nameof(input.OutputDir));
        if (string.IsNullOrWhiteSpace(input.WorkingDirectory))
            throw new ArgumentException("Working directory is required.", nameof(input.WorkingDirectory));

        if (input.BatchSize is < 1 or > 4)
            throw new ArgumentOutOfRangeException(nameof(input.BatchSize), input.BatchSize, "CPU batch size must be 1 to 4.");
        if (input.GradientAccumulationSteps is < 1 or > 64)
            throw new ArgumentOutOfRangeException(nameof(input.GradientAccumulationSteps), "Gradient accumulation must be 1 to 64.");
        if (input.MaxLength is < 32 or > 2048)
            throw new ArgumentOutOfRangeException(nameof(input.MaxLength), "CPU max length must be 32 to 2048.");
        if (input.LoraRank is < 1 or > 64)
            throw new ArgumentOutOfRangeException(nameof(input.LoraRank), "LoRA rank must be 1 to 64.");
        if (input.LoraAlpha is < 1 or > 256)
            throw new ArgumentOutOfRangeException(nameof(input.LoraAlpha), "LoRA alpha must be 1 to 256.");
        if (input.MaxSteps < 0)
            throw new ArgumentOutOfRangeException(nameof(input.MaxSteps), "Max steps cannot be negative. Use 0 to train by epochs.");
        if (input.Epochs <= 0)
            throw new ArgumentOutOfRangeException(nameof(input.Epochs), "Epochs must be positive.");
        if (input.LearningRate <= 0 || input.LearningRate > 1)
            throw new ArgumentOutOfRangeException(nameof(input.LearningRate), "Learning rate must be in (0, 1].");
    }

    private static void RequireSafePath(string value, string name)
    {
        if (string.IsNullOrWhiteSpace(value))
            throw new ArgumentException($"{name} is required.", name);
        if (value.Contains('\0'))
            throw new ArgumentException($"{name} contains a NUL character.", name);
        if (value.TrimStart().StartsWith('-'))
            throw new ArgumentException($"{name} must not look like a command flag.", name);
    }
}
