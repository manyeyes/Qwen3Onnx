using Microsoft.Extensions.Logging;

namespace Qwen3.Desktop.Services;

public sealed class FileLoggerProvider : ILoggerProvider
{
    private readonly object _sync = new();
    private readonly StreamWriter _writer;

    public FileLoggerProvider(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)
            ?? throw new ArgumentException("Log file path must include a directory.", nameof(path)));
        _writer = new StreamWriter(
            new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.ReadWrite | FileShare.Delete))
        {
            AutoFlush = true
        };
    }

    public ILogger CreateLogger(string categoryName) => new FileLogger(categoryName, _writer, _sync);

    public void Dispose()
    {
        lock (_sync)
        {
            _writer.Dispose();
        }
    }

    private sealed class FileLogger(string categoryName, StreamWriter writer, object sync) : ILogger
    {
        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => logLevel >= LogLevel.Information;

        public void Log<TState>(
            LogLevel logLevel,
            EventId eventId,
            TState state,
            Exception? exception,
            Func<TState, Exception?, string> formatter)
        {
            if (!IsEnabled(logLevel))
            {
                return;
            }

            var message = $"{DateTimeOffset.Now:O} [{logLevel}] {categoryName} " +
                          $"({eventId.Id}): {formatter(state, exception)}";
            if (exception is not null)
            {
                message += Environment.NewLine + exception;
            }

            lock (sync)
            {
                writer.WriteLine(message);
            }
        }
    }
}
