using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;

namespace TrainLoop.Core.Llm;

public sealed record SftMessage(string Role, string Content);

public sealed record SftConversation(IReadOnlyList<SftMessage> Messages);

/// <summary>
/// Turns human annotations into the messages JSONL consumed by the CPU LoRA trainer.
/// </summary>
public static class SftJsonlWriter
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
    };

    public static SftConversation FromAnnotation(string content, string label, string rationale)
    {
        if (string.IsNullOrWhiteSpace(content))
            throw new ArgumentException("Content is required.", nameof(content));
        if (string.IsNullOrWhiteSpace(label))
            throw new ArgumentException("Label is required.", nameof(label));
        if (string.IsNullOrWhiteSpace(rationale))
            throw new ArgumentException("Rationale is required.", nameof(rationale));

        return new SftConversation(
        [
            new SftMessage("system", "你是标注助手。根据文本给出标签，并用一句话说明理由。"),
            new SftMessage("user", content.Trim()),
            new SftMessage("assistant", $"标签：{label.Trim()}\n理由：{rationale.Trim()}")
        ]);
    }

    public static string Write(IEnumerable<SftConversation> conversations)
    {
        var builder = new StringBuilder();
        foreach (var conversation in conversations)
        {
            if (conversation.Messages.Count == 0)
                continue;

            var payload = new
            {
                messages = conversation.Messages.Select(message => new
                {
                    role = message.Role,
                    content = message.Content
                })
            };
            builder.AppendLine(JsonSerializer.Serialize(payload, JsonOptions));
        }

        return builder.ToString();
    }
}
