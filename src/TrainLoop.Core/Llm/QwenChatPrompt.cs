namespace TrainLoop.Core.Llm;

/// <summary>
/// Qwen2.5 Instruct chat markup used by llama.cpp when the GGUF template is not applied.
/// </summary>
public static class QwenChatPrompt
{
    public const string DefaultSystemPrompt = "You are a helpful assistant.";
    public const string EndToken = "<|im_end|>";

    public static string Format(string? systemPrompt, string userPrompt)
    {
        if (string.IsNullOrWhiteSpace(userPrompt))
            throw new ArgumentException("Prompt is required.", nameof(userPrompt));

        var system = string.IsNullOrWhiteSpace(systemPrompt)
            ? DefaultSystemPrompt
            : systemPrompt.Trim();

        return
            $"<|im_start|>system\n{system}{EndToken}\n" +
            $"<|im_start|>user\n{userPrompt.Trim()}{EndToken}\n" +
            "<|im_start|>assistant\n";
    }
}
