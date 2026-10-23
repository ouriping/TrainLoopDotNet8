namespace TrainLoop.Core.Entities;

public sealed class Reviewer
{
    public Guid Id { get; init; } = Guid.NewGuid();
    public required string Name { get; set; }
    public double QualityScore { get; set; } = 1.0;
    public int TotalAnnotations { get; set; }
    public int QuizScore { get; set; }
    public DateTimeOffset CreatedAt { get; init; } = DateTimeOffset.UtcNow;
}
