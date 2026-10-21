using Qwen3.Desktop.Models;

namespace Qwen3.Desktop.Services;

public sealed class AuthenticationService(SqliteDatabase database)
{
    public Task<UserAccount?> SignInAsync(
        string username, string password, CancellationToken cancellationToken = default) =>
        database.AuthenticateAsync(username, password, cancellationToken);

    public Task<IReadOnlyList<UserAccount>> GetUsersAsync(CancellationToken cancellationToken = default) =>
        database.GetUsersAsync(cancellationToken);

    public Task<string> CreateUserAsync(
        string username, string password, CancellationToken cancellationToken = default) =>
        database.CreateUserAsync(username, password, cancellationToken);

    public Task SetLockedAsync(string userId, bool locked, CancellationToken cancellationToken = default) =>
        database.SetUserLockedAsync(userId, locked, cancellationToken);
}
