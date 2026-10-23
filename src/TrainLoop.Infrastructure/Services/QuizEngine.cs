using Microsoft.EntityFrameworkCore;
using TrainLoop.Core.Entities;
using TrainLoop.Core.Events;
using TrainLoop.Core.Models;
using TrainLoop.Core.Services;
using TrainLoop.Infrastructure.Data;

namespace TrainLoop.Infrastructure.Services;

public sealed class QuizEngine : IQuizEngine
{
    private readonly TrainLoopDbContext _db;

    public QuizEngine(TrainLoopDbContext db)
    {
        _db = db;
    }

    public async Task<QuizItem?> GetNextQuizItemAsync(Guid reviewerId, CancellationToken ct = default)
    {
        // Get IDs of quiz items this reviewer has already attempted
        var attemptedIds = await _db.QuizAttempts
            .Where(qa => qa.ReviewerId == reviewerId)
            .Select(qa => qa.QuizItemId)
            .ToListAsync(ct);

        // Return a random unattempted quiz item
        var candidates = await _db.QuizItems
            .Where(qi => !attemptedIds.Contains(qi.Id))
            .ToListAsync(ct);

        if (candidates.Count == 0)
            return null;

        var index = Random.Shared.Next(candidates.Count);
        return candidates[index];
    }

    public async Task<bool> SubmitAnswerAsync(Guid reviewerId, Guid quizItemId, string answer, CancellationToken ct = default)
    {
        var quizItem = await _db.QuizItems.FindAsync([quizItemId], ct);
        if (quizItem is null)
            return false;

        var reviewer = await _db.Reviewers.FindAsync([reviewerId], ct);
        if (reviewer is null)
            return false;

        var isCorrect = string.Equals(quizItem.KnownLabel, answer, StringComparison.OrdinalIgnoreCase);

        var attempt = new QuizAttempt
        {
            ReviewerId = reviewerId,
            QuizItemId = quizItemId,
            Answer = answer,
            IsCorrect = isCorrect
        };
        _db.QuizAttempts.Add(attempt);

        // Update reviewer quality score based on running quiz accuracy
        var allAttempts = await _db.QuizAttempts
            .Where(qa => qa.ReviewerId == reviewerId)
            .ToListAsync(ct);

        var totalCorrect = allAttempts.Count(a => a.IsCorrect) + (isCorrect ? 1 : 0);
        var totalAttempts = allAttempts.Count + 1;
        var quizAccuracy = (double)totalCorrect / totalAttempts;

        var oldScore = reviewer.QualityScore;
        // Blend existing quality score with quiz accuracy (50/50 weight)
        reviewer.QualityScore = Math.Round((oldScore + quizAccuracy) / 2.0, 4);

        await _db.SaveChangesAsync(ct);

        // Emit domain events (fire-and-forget; no event bus wired yet — log to console)
        var now = DateTimeOffset.UtcNow;
        _ = new QuizAnswered(reviewerId, quizItemId, answer, isCorrect, now);
        _ = new ReviewerCalibrated(reviewerId, reviewer.QualityScore, oldScore, now);

        return isCorrect;
    }

    public async Task<ReviewerStats> GetReviewerStatsAsync(Guid reviewerId, CancellationToken ct = default)
    {
        var reviewer = await _db.Reviewers.FindAsync([reviewerId], ct)
            ?? throw new InvalidOperationException($"Reviewer {reviewerId} not found.");

        var attempts = await _db.QuizAttempts
            .Where(qa => qa.ReviewerId == reviewerId)
            .ToListAsync(ct);

        var quizAttempts = attempts.Count;
        var quizCorrect = attempts.Count(a => a.IsCorrect);
        var quizAccuracy = quizAttempts == 0 ? 0.0 : (double)quizCorrect / quizAttempts;

        // Inter-annotator agreement: fraction of reviewer's annotations whose label
        // matches the majority label among all annotations on the same data item.
        var reviewerAnnotations = await _db.Annotations
            .Where(a => a.ReviewerId == reviewerId)
            .ToListAsync(ct);

        double interAnnotatorAgreement = 0.0;
        if (reviewerAnnotations.Count > 0)
        {
            var dataItemIds = reviewerAnnotations.Select(a => a.DataItemId).Distinct().ToList();

            var allAnnotationsForItems = await _db.Annotations
                .Where(a => dataItemIds.Contains(a.DataItemId))
                .ToListAsync(ct);

            var majorityLabels = allAnnotationsForItems
                .GroupBy(a => a.DataItemId)
                .ToDictionary(
                    g => g.Key,
                    g => g.GroupBy(a => a.Label)
                           .OrderByDescending(lg => lg.Count())
                           .First().Key);

            var agreementCount = reviewerAnnotations
                .Count(a => majorityLabels.TryGetValue(a.DataItemId, out var majority)
                            && string.Equals(a.Label, majority, StringComparison.OrdinalIgnoreCase));

            interAnnotatorAgreement = Math.Round((double)agreementCount / reviewerAnnotations.Count, 4);
        }

        return new ReviewerStats(
            ReviewerId: reviewer.Id,
            Name: reviewer.Name,
            QualityScore: reviewer.QualityScore,
            TotalAnnotations: reviewer.TotalAnnotations,
            QuizAttempts: quizAttempts,
            QuizCorrect: quizCorrect,
            QuizAccuracy: Math.Round(quizAccuracy, 4),
            InterAnnotatorAgreement: interAnnotatorAgreement);
    }
}
