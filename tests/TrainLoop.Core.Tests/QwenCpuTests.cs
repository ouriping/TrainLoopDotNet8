using TrainLoop.Core.Llm;

namespace TrainLoop.Core.Tests;

public class QwenCpuTests
{
    [Fact]
    public void QwenChatPrompt_UsesInstructMarkers()
    {
        var prompt = QwenChatPrompt.Format("系统", "你好");

        Assert.Contains("<|im_start|>system\n系统<|im_end|>", prompt);
        Assert.Contains("<|im_start|>user\n你好<|im_end|>", prompt);
        Assert.EndsWith("<|im_start|>assistant\n", prompt);
    }

    [Fact]
    public void QwenChatPrompt_RejectsBlankUserText()
    {
        Assert.Throws<ArgumentException>(() => QwenChatPrompt.Format(null, "  "));
    }

    [Fact]
    public void SftJsonlWriter_KeepsChineseText()
    {
        var jsonl = SftJsonlWriter.Write(
        [
            SftJsonlWriter.FromAnnotation("音质很好", "positive", "明确表扬")
        ]);

        Assert.Contains("\"role\":\"user\"", jsonl);
        Assert.Contains("音质很好", jsonl);
        Assert.Contains("标签：positive", jsonl);
        Assert.Contains("理由：明确表扬", jsonl);
        Assert.DoesNotContain("\\u", jsonl);
    }

    [Fact]
    public void CommandBuilder_PeftPlan_ForcesCpuEnvironment()
    {
        var plan = CpuTrainingCommandBuilder.Build(Sample(LlmTrainerKind.TransformersPeft));

        Assert.Equal("transformers-peft", plan.Trainer);
        Assert.Equal("python3", plan.Executable);
        Assert.Equal(string.Empty, plan.Environment["CUDA_VISIBLE_DEVICES"]);
        Assert.Equal(string.Empty, plan.Environment["HIP_VISIBLE_DEVICES"]);
        Assert.Contains("--dataset", plan.Arguments);
        Assert.Contains("--max-steps", plan.Arguments);
        Assert.Contains("1", plan.Arguments);
        Assert.DoesNotContain(plan.Arguments, argument => argument.Contains("cuda", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void CommandBuilder_LlamaFactoryPlan_DisablesHalfPrecision()
    {
        var plan = CpuTrainingCommandBuilder.Build(Sample(LlmTrainerKind.LlamaFactory) with
        {
            Executable = "llamafactory-cli",
            ScriptOrConfigPath = "/work/qwen.yaml"
        });

        Assert.Equal("llamafactory", plan.Trainer);
        Assert.Equal("llamafactory-cli", plan.Executable);
        Assert.Contains("bf16=false", plan.Arguments);
        Assert.Contains("fp16=false", plan.Arguments);
        Assert.Contains("flash_attn=disabled", plan.Arguments);
        Assert.Contains("max_steps=20", plan.Arguments);
    }

    [Fact]
    public void CommandBuilder_RejectsOversizedCpuBatch()
    {
        var input = Sample(LlmTrainerKind.TransformersPeft) with { BatchSize = 32 };
        Assert.Throws<ArgumentOutOfRangeException>(() => CpuTrainingCommandBuilder.Build(input));
    }

    [Fact]
    public void CommandBuilder_RejectsFlagShapedDatasetPath()
    {
        var input = Sample(LlmTrainerKind.TransformersPeft) with { DatasetPath = "--evil" };
        Assert.Throws<ArgumentException>(() => CpuTrainingCommandBuilder.Build(input));
    }

    [Fact]
    public void RepoFileLocator_FindsFileInParentDirectory()
    {
        var root = Directory.CreateTempSubdirectory("trainloop-locator");
        try
        {
            var nested = Directory.CreateDirectory(Path.Combine(root.FullName, "src", "TrainLoop.Api"));
            var tools = Directory.CreateDirectory(Path.Combine(root.FullName, "tools"));
            var script = Path.Combine(tools.FullName, "train.py");
            File.WriteAllText(script, "print(1)\n");

            var resolved = RepoFileLocator.ResolveFile("tools/train.py", nested.FullName);

            Assert.Equal(Path.GetFullPath(script), resolved);
        }
        finally
        {
            root.Delete(recursive: true);
        }
    }

    private static TrainingCommandInput Sample(LlmTrainerKind trainer) => new(
        trainer,
        "python3",
        "/work/train_qwen_lora.py",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "/work/corrections.jsonl",
        "/work/saves/adapter",
        "/work",
        MaxSteps: 20,
        Epochs: 1,
        BatchSize: 1,
        GradientAccumulationSteps: 4,
        MaxLength: 256,
        LoraRank: 8,
        LoraAlpha: 16,
        LearningRate: 1e-4);
}
