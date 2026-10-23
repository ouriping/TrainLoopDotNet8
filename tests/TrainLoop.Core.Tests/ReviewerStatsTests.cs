using TrainLoop.Core.Models;

namespace TrainLoop.Core.Tests;

public class ReviewerStatsTests
{
    [Fact]
    public void ReviewerStats_ComputesQuizAccuracy_Correctly()
    {
        var stats = new ReviewerStats(
            ReviewerId: Guid.NewGuid(),
            Name: "Alice",
            QualityScore: 0.85,
            TotalAnnotations: 100,
            QuizAttempts: 10,
            QuizCorrect: 8,
            QuizAccuracy: 0.8,
            InterAnnotatorAgreement: 0.9);

        Assert.Equal(0.8, stats.QuizAccuracy);
        Assert.Equal(10, stats.QuizAttempts);
        Assert.Equal(8, stats.QuizCorrect);
    }

    [Fact]
    public void ReviewerStats_WithZeroAttempts_HasZeroAccuracy()
    {
        var stats = new ReviewerStats(
            ReviewerId: Guid.NewGuid(),
            Name: "Bob",
            QualityScore: 1.0,
            TotalAnnotations: 0,
            QuizAttempts: 0,
            QuizCorrect: 0,
            QuizAccuracy: 0.0,
            InterAnnotatorAgreement: 0.0);

        Assert.Equal(0.0, stats.QuizAccuracy);
        Assert.Equal(0, stats.QuizAttempts);
    }

    [Fact]
    public void ReviewerStats_IsValueEqual_WhenSameData()
    {
        var id = Guid.NewGuid();
        var a = new ReviewerStats(id, "Carol", 0.75, 50, 5, 4, 0.8, 0.85);
        var b = new ReviewerStats(id, "Carol", 0.75, 50, 5, 4, 0.8, 0.85);

        Assert.Equal(a, b);
    }

    [Fact]
    public void ReviewerStats_IsNotEqual_WhenDifferentReviewerId()
    {
        var a = new ReviewerStats(Guid.NewGuid(), "Dave", 1.0, 0, 0, 0, 0.0, 0.0);
        var b = new ReviewerStats(Guid.NewGuid(), "Dave", 1.0, 0, 0, 0, 0.0, 0.0);

        Assert.NotEqual(a, b);
    }

    [Fact]
    public void QuizAttempt_DefaultsToNewGuid_AndUtcNow()
    {
        var attempt = new QuizAttempt
        {
            ReviewerId = Guid.NewGuid(),
            QuizItemId = Guid.NewGuid(),
            Answer = "positive"
        };

        Assert.NotEqual(Guid.Empty, attempt.Id);
        Assert.Equal("positive", attempt.Answer);
        Assert.False(attempt.IsCorrect); // default
        Assert.True(attempt.AnsweredAt <= DateTimeOffset.UtcNow);
    }

    [Fact]
    public void QuizAttempt_IsCorrect_CanBeSet()
    {
        var attempt = new QuizAttempt
        {
            ReviewerId = Guid.NewGuid(),
            QuizItemId = Guid.NewGuid(),
            Answer = "negative",
            IsCorrect = true
        };

        Assert.True(attempt.IsCorrect);
    }
}
