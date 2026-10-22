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

public sealed record UserInsights(
    int ConversationCount,
    int MessageCount,
    long TokensUsed,
    IReadOnlyList<string> Keywords,
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
    IReadOnlyList<ModelFileLink> HttpFileLinks);

public sealed record ModelFileLink(string Path, string Url, long Size = 0, string? Sha256 = null);

public sealed record ModelDownloadResult(ModelDefinition Model, string? CleanupWarning);

public sealed record ConversationSummary(
    string Id,
    string Title,
    string? OwnerUserId,
    string ModelId,
    DateTimeOffset CreatedAt,
    DateTimeOffset UpdatedAt);

public sealed record StoredMessage(
    string Id,
    string ConversationId,
    string Role,
    string Content,
    IReadOnlyList<string> Attachments,
    int TokenCount,
    DateTimeOffset CreatedAt,
    bool WasInterrupted = false);

public sealed record ModelDownloadProgress(
    string Message,
    long BytesReceived,
    long TotalBytes,
    double? Percentage);

public sealed record BootstrapResult(string? GeneratedAdminPassword);
