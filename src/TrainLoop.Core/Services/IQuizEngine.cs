using TrainLoop.Core.Entities;
using TrainLoop.Core.Models;

namespace TrainLoop.Core.Services;

public interface IQuizEngine
{
    Task<QuizItem?> GetNextQuizItemAsync(Guid reviewerId, CancellationToken ct = default);
    Task<bool> SubmitAnswerAsync(Guid reviewerId, Guid quizItemId, string answer, CancellationToken ct = default);
    Task<ReviewerStats> GetReviewerStatsAsync(Guid reviewerId, CancellationToken ct = default);
}
