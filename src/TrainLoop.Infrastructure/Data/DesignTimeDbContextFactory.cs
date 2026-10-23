using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Design;

namespace TrainLoop.Infrastructure.Data;

public sealed class DesignTimeDbContextFactory : IDesignTimeDbContextFactory<TrainLoopDbContext>
{
    public TrainLoopDbContext CreateDbContext(string[] args)
    {
        var optionsBuilder = new DbContextOptionsBuilder<TrainLoopDbContext>();

        // Use the connection string from environment variable or fall back to a default for migrations
        var connectionString = Environment.GetEnvironmentVariable("TRAINLOOP_CONNECTION_STRING")
            ?? "Host=localhost;Database=trainloop;Username=postgres;Password=postgres";

        optionsBuilder.UseNpgsql(connectionString);

        return new TrainLoopDbContext(optionsBuilder.Options);
    }
}
