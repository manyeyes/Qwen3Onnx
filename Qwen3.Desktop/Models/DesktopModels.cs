namespace Qwen3.Desktop.Models;

public sealed record UserAccount(
    string Id,
    string Username,
    string Role,
    bool IsLocked,
    DateTimeOffset CreatedAt,
    long TokensUsed)
{
    public bool CanBeLocked => Role != "admin";
}

public sealed record UserProfile(
    string UserId,
    string Username,
    string Nickname,
    string AvatarUrl,
    string Birthday,
    string BlogUrl,
    string Bio,
    string Role,
    DateTimeOffset CreatedAt,
    long TokensUsed);

public sealed record UserDomainInsight(string Name, int Count, double Percentage);

public sealed record UserKeywordInsight(string Term, int Count);

public sealed record UserInsights(
    int ConversationCount,
    int MessageCount,
    long TokensUsed,
    IReadOnlyList<UserKeywordInsight> Keywords,
    IReadOnlyList<UserDomainInsight> Domains);

public sealed record ModelDefinition(
    string Id,
    string Title,
    string EnglishName,
    string Description,
    string RepositoryUrl,
    string LocalPath,
    bool IsInstalled,
    string EmbeddingFileName,
    string DecoderFileName,
    IReadOnlyList<string> RequiredFiles,
    IReadOnlyList<ModelFileLink> HttpFileLinks,
    string EngineId = "qwen3-onnx")
{
    public string InstallationStatus => IsInstalled ? "已安装" : "未安装";
    private static readonly HashSet<string> GenaiEngineIds =
    new(StringComparer.OrdinalIgnoreCase)
    {
        "qwen3-onnx-genai",
        "llama-onnx-genai",
        "phi-onnx-genai",
        "deepseek-onnx-genai",
    };
    public bool RequiresGenAi => GenaiEngineIds.Contains(EngineId);
    public bool IsLlamaGenAi =>
        string.Equals(EngineId, "llama-onnx-genai", StringComparison.OrdinalIgnoreCase);
    public bool IsPhiGenAi =>
        string.Equals(EngineId, "phi-onnx-genai", StringComparison.OrdinalIgnoreCase);
    public bool IsDeepSeekGenAi =>
        string.Equals(EngineId, "deepseek-onnx-genai", StringComparison.OrdinalIgnoreCase);
    public string EngineDisplayName => RequiresGenAi ? "ONNX Runtime GenAI" : "Qwen3 ONNX";

    /// <summary>HTTP 文件数量；为 null 时返回 0，供 XAML 绑定使用。</summary>
    public int HttpFileCount => HttpFileLinks?.Count ?? 0;
}

public sealed record ModelFileLink(string Path, string Url, long Size = 0, string? Sha256 = null)
{
    public string DisplayText => $"{Path} | {Url}";
}

public sealed record ModelDownloadResult(ModelDefinition Model, string? CleanupWarning);

public sealed record ConversationSummary(
    string Id,
    string Title,
    string? OwnerUserId,
    string ModelId,
    DateTimeOffset CreatedAt,
    DateTimeOffset UpdatedAt,
    bool IsPinned = false);

public sealed record StoredMessage(
    string Id,
    string ConversationId,
    string Role,
    string Content,
    IReadOnlyList<string> Attachments,
    int TokenCount,
    DateTimeOffset CreatedAt,
    bool WasInterrupted = false,
    double GenerationTimeSeconds = 0,
    string StopReason = "");

public sealed record ModelDownloadProgress(
    string Message,
    long BytesReceived,
    long TotalBytes,
    double? Percentage);

public sealed record BootstrapResult(string? GeneratedAdminPassword);
