using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace Qwen3.Cli;

internal sealed class JsonFileLoggerProvider : ILoggerProvider
{
    private readonly object _syncRoot = new();
    private readonly StreamWriter _writer;

    public JsonFileLoggerProvider(string path)
    {
        var fullPath = Path.GetFullPath(path);
        var directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        _writer = new StreamWriter(fullPath, append: true, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false))
        {
            AutoFlush = true
        };
    }

    public ILogger CreateLogger(string categoryName) => new JsonFileLogger(this, categoryName);

    public void Dispose()
    {
        lock (_syncRoot)
        {
            _writer.Dispose();
        }
    }

    private void Write<TState>(
        string category,
        LogLevel logLevel,
        EventId eventId,
        TState state,
        Exception? exception,
        Func<TState, Exception?, string> formatter)
    {
        var entry = new
        {
            TimestampUtc = DateTimeOffset.UtcNow,
            Level = logLevel.ToString(),
            Category = category,
            EventId = eventId.Id,
            Message = formatter(state, exception),
            Exception = exception?.ToString()
        };

        lock (_syncRoot)
        {
            _writer.WriteLine(JsonSerializer.Serialize(entry));
        }
    }

    private sealed class JsonFileLogger(JsonFileLoggerProvider provider, string category) : ILogger
    {
        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
        public bool IsEnabled(LogLevel logLevel) => logLevel != LogLevel.None;

        public void Log<TState>(
            LogLevel logLevel,
            EventId eventId,
            TState state,
            Exception? exception,
            Func<TState, Exception?, string> formatter)
        {
            if (IsEnabled(logLevel))
            {
                provider.Write(category, logLevel, eventId, state, exception, formatter);
            }
        }
    }
}
