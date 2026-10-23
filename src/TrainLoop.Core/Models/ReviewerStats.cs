namespace TrainLoop.Core.Models;

public sealed record ReviewerStats(
    Guid ReviewerId,
    string Name,
    double QualityScore,
    int TotalAnnotations,
    int QuizAttempts,
    int QuizCorrect,
    double QuizAccuracy,
    double InterAnnotatorAgreement);
