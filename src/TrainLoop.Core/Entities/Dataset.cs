namespace TrainLoop.Core.Entities;

public sealed class Dataset
{
    public Guid Id { get; init; } = Guid.NewGuid();
    public required string Name { get; set; }
    public string? Description { get; set; }
    public DateTimeOffset CreatedAt { get; init; } = DateTimeOffset.UtcNow;
    public ICollection<DataItem> Items { get; init; } = [];
}
