using Microsoft.EntityFrameworkCore;
using TrainLoop.Core.Entities;
using TrainLoop.Core.Models;

namespace TrainLoop.Infrastructure.Data;

public sealed class TrainLoopDbContext : DbContext
{
    public TrainLoopDbContext(DbContextOptions<TrainLoopDbContext> options) : base(options) { }

    public DbSet<Dataset> Datasets => Set<Dataset>();
    public DbSet<DataItem> DataItems => Set<DataItem>();
    public DbSet<Annotation> Annotations => Set<Annotation>();
    public DbSet<Reviewer> Reviewers => Set<Reviewer>();
    public DbSet<QuizItem> QuizItems => Set<QuizItem>();
    public DbSet<QuizAttempt> QuizAttempts => Set<QuizAttempt>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        base.OnModelCreating(modelBuilder);

        modelBuilder.Entity<Dataset>(entity =>
        {
            entity.HasKey(d => d.Id);
            entity.Property(d => d.Name).IsRequired();
            entity.HasMany(d => d.Items)
                  .WithOne()
                  .HasForeignKey(i => i.DatasetId)
                  .OnDelete(DeleteBehavior.Cascade);
        });

        modelBuilder.Entity<DataItem>(entity =>
        {
            entity.HasKey(i => i.Id);
            entity.Property(i => i.Content).IsRequired();
            entity.HasIndex(i => i.DatasetId);
            entity.HasMany(i => i.Annotations)
                  .WithOne()
                  .HasForeignKey(a => a.DataItemId)
                  .OnDelete(DeleteBehavior.Cascade);
        });

        modelBuilder.Entity<Annotation>(entity =>
        {
            entity.HasKey(a => a.Id);
            entity.Property(a => a.Label).IsRequired();
            entity.Property(a => a.Rationale).IsRequired();
            entity.HasIndex(a => a.DataItemId);
            entity.HasIndex(a => a.ReviewerId);
        });

        modelBuilder.Entity<Reviewer>(entity =>
        {
            entity.HasKey(r => r.Id);
            entity.Property(r => r.Name).IsRequired();
        });

        modelBuilder.Entity<QuizItem>(entity =>
        {
            entity.HasKey(q => q.Id);
            entity.Property(q => q.KnownLabel).IsRequired();
        });

        modelBuilder.Entity<QuizAttempt>(entity =>
        {
            entity.HasKey(qa => qa.Id);
            entity.Property(qa => qa.Answer).IsRequired();
            entity.HasIndex(qa => qa.ReviewerId);
            entity.HasIndex(qa => qa.QuizItemId);
            entity.HasIndex(qa => new { qa.ReviewerId, qa.QuizItemId });
        });
    }
}
