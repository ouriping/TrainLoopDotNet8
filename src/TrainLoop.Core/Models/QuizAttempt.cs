namespace TrainLoop.Core.Models;

public sealed class QuizAttempt
{
    public Guid Id { get; init; } = Guid.NewGuid();
    public Guid ReviewerId { get; init; }
    public Guid QuizItemId { get; init; }
    public required string Answer { get; set; }
    public bool IsCorrect { get; set; }
    public DateTimeOffset AnsweredAt { get; init; } = DateTimeOffset.UtcNow;
}
