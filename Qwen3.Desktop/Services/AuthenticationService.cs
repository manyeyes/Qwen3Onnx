using Qwen3.Desktop.Models;

namespace Qwen3.Desktop.Services;

public sealed class AuthenticationService(SqliteDatabase database)
{
    private static readonly TimeSpan LoginSessionDuration = TimeSpan.FromDays(30);

    public Task<UserAccount?> SignInAsync(
        string username, string password, CancellationToken cancellationToken = default) =>
        database.AuthenticateAsync(username, password, cancellationToken);

    public Task<UserAccount?> RestoreSessionAsync(CancellationToken cancellationToken = default) =>
        database.RestoreLoginSessionAsync(cancellationToken);

    public Task SaveSessionAsync(string userId, CancellationToken cancellationToken = default) =>
        database.SaveLoginSessionAsync(
            userId, DateTimeOffset.UtcNow.Add(LoginSessionDuration), cancellationToken);

    public Task ClearSessionAsync(CancellationToken cancellationToken = default) =>
        database.ClearLoginSessionAsync(cancellationToken);

    public Task<IReadOnlyList<UserAccount>> GetUsersAsync(CancellationToken cancellationToken = default) =>
        database.GetUsersAsync(cancellationToken);

    public Task<string> CreateUserAsync(
        string username, string password, CancellationToken cancellationToken = default) =>
        database.CreateUserAsync(username, password, cancellationToken);

    public Task SetLockedAsync(string userId, bool locked, CancellationToken cancellationToken = default) =>
        database.SetUserLockedAsync(userId, locked, cancellationToken);

    public Task<UserProfile> GetProfileAsync(string userId, CancellationToken cancellationToken = default) =>
        database.GetUserProfileAsync(userId, cancellationToken);

    public Task UpdateProfileAsync(
        string userId,
        string nickname,
        string avatarUrl,
        string birthday,
        string blogUrl,
        string bio,
        CancellationToken cancellationToken = default) =>
        database.UpdateUserProfileAsync(
            userId, nickname, avatarUrl, birthday, blogUrl, bio, cancellationToken);

    public Task<UserInsights> GetInsightsAsync(string userId, CancellationToken cancellationToken = default) =>
        database.GetOwnUserInsightsAsync(userId, cancellationToken);

    public Task ChangePasswordAsync(
        string userId, string currentPassword, string newPassword,
        CancellationToken cancellationToken = default) =>
        database.ChangeOwnPasswordAsync(userId, currentPassword, newPassword, cancellationToken);

    public Task DeleteOwnAccountAsync(
        string userId, string password, CancellationToken cancellationToken = default) =>
        database.DeleteOwnAccountAsync(userId, password, cancellationToken);
}
