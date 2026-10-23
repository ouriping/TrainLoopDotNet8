namespace TrainLoop.Core.Entities;

public sealed class DataItem
{
    public Guid Id { get; init; } = Guid.NewGuid();
    public Guid DatasetId { get; init; }
    public required string Content { get; set; }
    public string? ContentType { get; set; }
    public ICollection<Annotation> Annotations { get; init; } = [];
}
