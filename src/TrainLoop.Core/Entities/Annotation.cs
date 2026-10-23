namespace TrainLoop.Core.Entities;

public sealed class Annotation
{
    public Guid Id { get; init; } = Guid.NewGuid();
    public Guid DataItemId { get; init; }
    public Guid ReviewerId { get; init; }
    public required string Label { get; set; }
    public required string Rationale { get; set; }  // mandatory written reasoning
    public double Confidence { get; set; }
    public DateTimeOffset CreatedAt { get; init; } = DateTimeOffset.UtcNow;
    public TimeSpan TimeToLabel { get; set; }
}
