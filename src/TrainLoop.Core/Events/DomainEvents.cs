namespace TrainLoop.Core.Events;

public abstract record DomainEvent(DateTimeOffset Timestamp);

public sealed record AnnotationSubmitted(
    Guid ItemId,
    Guid ReviewerId,
    string Label,
    string Rationale,
    DateTimeOffset Timestamp) : DomainEvent(Timestamp);

public sealed record QuizAnswered(
    Guid ReviewerId,
    Guid QuizItemId,
    string Answer,
    bool Correct,
    DateTimeOffset Timestamp) : DomainEvent(Timestamp);

public sealed record ConsensusResolved(
    Guid ItemId,
    string FinalLabel,
    string Strategy,
    DateTimeOffset Timestamp) : DomainEvent(Timestamp);

public sealed record ReviewerCalibrated(
    Guid ReviewerId,
    double NewScore,
    double OldScore,
    DateTimeOffset Timestamp) : DomainEvent(Timestamp);
