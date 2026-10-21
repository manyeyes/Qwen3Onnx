using Microsoft.Extensions.Logging;

namespace Qwen3.Desktop.Services;

public sealed class DesktopServices : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly ILoggerFactory _loggerFactory;

    public SqliteDatabase Database { get; }
    public AuthenticationService Authentication { get; }
    public ModelDownloadService ModelDownloader { get; }
    public DesktopChatService Chat { get; }
    public ILoggerFactory LoggerFactory => _loggerFactory;

    public DesktopServices(string? applicationDataDirectory = null)
    {
        var localData = applicationDataDirectory ??
                        Path.Combine(
                            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                            "Qwen3OnnxDesktop");
        var appDataPath = Path.GetFullPath(localData);
        Database = new SqliteDatabase(appDataPath);
        Authentication = new AuthenticationService(Database);
        _httpClient = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        ModelDownloader = new ModelDownloadService(_httpClient);
        _loggerFactory = Microsoft.Extensions.Logging.LoggerFactory.Create(builder =>
        {
            builder.AddConsole();
            builder.AddProvider(new FileLoggerProvider(
                Path.Combine(appDataPath, "logs", "desktop.log")));
            builder.SetMinimumLevel(LogLevel.Information);
        });
        Chat = new DesktopChatService(Database, _loggerFactory);
    }

    public void Dispose()
    {
        Chat.Dispose();
        _httpClient.Dispose();
        _loggerFactory.Dispose();
    }
}
