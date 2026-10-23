namespace TrainLoop.Core.Entities;

public sealed class QuizItem
{
    public Guid Id { get; init; } = Guid.NewGuid();
    public Guid DataItemId { get; init; }
    public required string KnownLabel { get; set; }
    public string? Difficulty { get; set; } // easy, medium, hard
}
