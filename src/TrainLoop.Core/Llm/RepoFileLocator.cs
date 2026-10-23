namespace TrainLoop.Core.Llm;

public static class RepoFileLocator
{
    public static string ResolveFile(string configuredPath, string startDirectory)
    {
        if (string.IsNullOrWhiteSpace(configuredPath))
            throw new ArgumentException("Path is required.", nameof(configuredPath));
        if (string.IsNullOrWhiteSpace(startDirectory))
            throw new ArgumentException("Start directory is required.", nameof(startDirectory));

        if (Path.IsPathRooted(configuredPath))
            return Path.GetFullPath(configuredPath);

        var relative = configuredPath.Replace('/', Path.DirectorySeparatorChar);
        foreach (var directory in Ancestors(startDirectory))
        {
            var candidate = Path.GetFullPath(Path.Combine(directory, relative));
            if (File.Exists(candidate))
                return candidate;
        }

        return Path.GetFullPath(Path.Combine(startDirectory, relative));
    }

    public static string ResolveDirectory(string configuredPath, string startDirectory)
    {
        if (string.IsNullOrWhiteSpace(configuredPath))
            throw new ArgumentException("Path is required.", nameof(configuredPath));

        if (Path.IsPathRooted(configuredPath))
            return Path.GetFullPath(configuredPath);

        var relative = configuredPath.Replace('/', Path.DirectorySeparatorChar);
        foreach (var directory in Ancestors(startDirectory))
        {
            var candidate = Path.GetFullPath(Path.Combine(directory, relative));
            if (Directory.Exists(candidate))
                return candidate;
        }

        return Path.GetFullPath(Path.Combine(startDirectory, relative));
    }

    public static string FindRepoRoot(string startDirectory)
    {
        foreach (var directory in Ancestors(startDirectory))
        {
            if (File.Exists(Path.Combine(directory, "TrainLoop.sln")) ||
                File.Exists(Path.Combine(directory, "tools", "cpu-trainer", "train_qwen_lora.py")))
            {
                return directory;
            }
        }

        return Path.GetFullPath(startDirectory);
    }

    private static IEnumerable<string> Ancestors(string startDirectory)
    {
        var directory = new DirectoryInfo(Path.GetFullPath(startDirectory));
        while (directory is not null)
        {
            yield return directory.FullName;
            directory = directory.Parent;
        }
    }
}
