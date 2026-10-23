using System.Text;
using Microsoft.EntityFrameworkCore;
using TrainLoop.Api;
using TrainLoop.Core.Entities;
using TrainLoop.Core.Llm;
using TrainLoop.Core.Models;
using TrainLoop.Core.Services;
using TrainLoop.Infrastructure.Data;
using TrainLoop.Infrastructure.Services;
using TrainLoop.Llm;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

builder.Services.AddDbContext<TrainLoopDbContext>(options =>
{
    var connectionString = builder.Configuration.GetConnectionString("DefaultConnection")
        ?? throw new InvalidOperationException("Connection string 'DefaultConnection' is not configured.");
    options.UseNpgsql(connectionString);
});

builder.Services.AddScoped<IQuizEngine, QuizEngine>();
builder.Services.AddTrainLoopLlm(builder.Configuration);

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

// --- Datasets ---

app.MapGet("/api/datasets", async (TrainLoopDbContext db) =>
    await db.Datasets.ToListAsync())
    .WithName("GetDatasets");

app.MapPost("/api/datasets", async (CreateDatasetRequest request, TrainLoopDbContext db) =>
{
    var dataset = new Dataset { Name = request.Name, Description = request.Description };
    db.Datasets.Add(dataset);
    await db.SaveChangesAsync();
    return Results.Created($"/api/datasets/{dataset.Id}", dataset);
})
.WithName("CreateDataset");

// --- Data Items ---

app.MapGet("/api/datasets/{id:guid}/items", async (Guid id, TrainLoopDbContext db) =>
{
    var exists = await db.Datasets.AnyAsync(d => d.Id == id);
    if (!exists) return Results.NotFound();

    var items = await db.DataItems.Where(i => i.DatasetId == id).ToListAsync();
    return Results.Ok(items);
})
.WithName("GetDatasetItems");

app.MapPost("/api/datasets/{id:guid}/items", async (Guid id, CreateDataItemRequest request, TrainLoopDbContext db) =>
{
    var exists = await db.Datasets.AnyAsync(d => d.Id == id);
    if (!exists) return Results.NotFound();

    var item = new DataItem
    {
        DatasetId = id,
        Content = request.Content,
        ContentType = request.ContentType
    };
    db.DataItems.Add(item);
    await db.SaveChangesAsync();
    return Results.Created($"/api/datasets/{id}/items/{item.Id}", item);
})
.WithName("CreateDatasetItem");

// --- Annotations ---

app.MapPost("/api/items/{id:guid}/annotations", async (Guid id, CreateAnnotationRequest request, TrainLoopDbContext db) =>
{
    if (string.IsNullOrWhiteSpace(request.Rationale))
        return Results.BadRequest("Rationale is required.");

    var itemExists = await db.DataItems.AnyAsync(i => i.Id == id);
    if (!itemExists) return Results.NotFound();

    var reviewerExists = await db.Reviewers.AnyAsync(r => r.Id == request.ReviewerId);
    if (!reviewerExists) return Results.NotFound();

    var annotation = new Annotation
    {
        DataItemId = id,
        ReviewerId = request.ReviewerId,
        Label = request.Label,
        Rationale = request.Rationale,
        Confidence = request.Confidence,
        TimeToLabel = request.TimeToLabel
    };
    db.Annotations.Add(annotation);
    await db.SaveChangesAsync();
    return Results.Created($"/api/items/{id}/annotations/{annotation.Id}", annotation);
})
.WithName("CreateAnnotation");

// --- Reviewers ---

app.MapGet("/api/reviewers", async (TrainLoopDbContext db) =>
    await db.Reviewers.ToListAsync())
    .WithName("GetReviewers");

// --- Quiz ---

app.MapGet("/api/reviewers/{id:guid}/quiz", async (Guid id, IQuizEngine quizEngine) =>
{
    var item = await quizEngine.GetNextQuizItemAsync(id);
    return item is null ? Results.NoContent() : Results.Ok(item);
})
.WithName("GetNextQuizItem");

app.MapPost("/api/reviewers/{id:guid}/quiz/{quizItemId:guid}", async (
    Guid id,
    Guid quizItemId,
    SubmitAnswerRequest request,
    IQuizEngine quizEngine) =>
{
    var isCorrect = await quizEngine.SubmitAnswerAsync(id, quizItemId, request.Answer);
    return Results.Ok(new { IsCorrect = isCorrect });
})
.WithName("SubmitQuizAnswer");

app.MapGet("/api/reviewers/{id:guid}/stats", async (Guid id, IQuizEngine quizEngine) =>
{
    try
    {
        var stats = await quizEngine.GetReviewerStatsAsync(id);
        return Results.Ok(stats);
    }
    catch (InvalidOperationException)
    {
        return Results.NotFound();
    }
})
.WithName("GetReviewerStats");

// --- Quiz Items (admin) ---

app.MapGet("/api/datasets/{id:guid}/corrections.jsonl", async (Guid id, TrainLoopDbContext db) =>
{
    var exists = await db.Datasets.AnyAsync(d => d.Id == id);
    if (!exists) return Results.NotFound();

    var rows = await (
        from item in db.DataItems
        join annotation in db.Annotations on item.Id equals annotation.DataItemId
        where item.DatasetId == id
        orderby annotation.CreatedAt
        select new { item.Content, annotation.Label, annotation.Rationale })
        .ToListAsync();

    var jsonl = SftJsonlWriter.Write(rows.Select(row =>
        SftJsonlWriter.FromAnnotation(row.Content, row.Label, row.Rationale)));
    return Results.Text(jsonl, "text/plain", Encoding.UTF8);
})
.WithName("ExportCorrectionJsonl");

app.MapLlmEndpoints();

app.MapPost("/api/quiz-items", async (CreateQuizItemRequest request, TrainLoopDbContext db) =>
{
    var item = new QuizItem
    {
        DataItemId = request.DataItemId,
        KnownLabel = request.KnownLabel,
        Difficulty = request.Difficulty
    };
    db.QuizItems.Add(item);
    await db.SaveChangesAsync();
    return Results.Created($"/api/quiz-items/{item.Id}", item);
})
.WithName("CreateQuizItem");

app.Run();

// --- Request records ---

record CreateDatasetRequest(string Name, string? Description);
record CreateDataItemRequest(string Content, string? ContentType);
record CreateAnnotationRequest(Guid ReviewerId, string Label, string Rationale, double Confidence, TimeSpan TimeToLabel);
record SubmitAnswerRequest(string Answer);
record CreateQuizItemRequest(Guid DataItemId, string KnownLabel, string? Difficulty);

public partial class Program
{
}
