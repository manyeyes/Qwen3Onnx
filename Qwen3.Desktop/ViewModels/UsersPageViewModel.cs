using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Extensions.Logging;
using Qwen3.Desktop.Models;
using Qwen3.Desktop.Services;

namespace Qwen3.Desktop.ViewModels;

public partial class UsersPageViewModel : ObservableObject
{
    private readonly DesktopServices _services;
    private readonly MainWindowViewModel _shell;
    private readonly ILogger<UsersPageViewModel> _logger;

    [ObservableProperty]
    private ObservableCollection<UserAccount> users = [];

    [ObservableProperty]
    private string newUsername = string.Empty;

    [ObservableProperty]
    private string newPassword = string.Empty;

    [ObservableProperty]
    private string statusText = string.Empty;

    [ObservableProperty]
    private bool isBusy;

    public bool HasNoUsers => Users.Count == 0;

    public UsersPageViewModel(DesktopServices services, MainWindowViewModel shell)
    {
        _services = services;
        _shell = shell;
        _logger = services.LoggerFactory.CreateLogger<UsersPageViewModel>();
    }

    public async Task RefreshAsync(CancellationToken cancellationToken = default)
    {
        Users = new ObservableCollection<UserAccount>(
            await _services.Authentication.GetUsersAsync(cancellationToken));
        OnPropertyChanged(nameof(HasNoUsers));
    }

    [RelayCommand]
    private async Task CreateUserAsync(CancellationToken cancellationToken)
    {
        if (!_shell.IsAdmin)
        {
            StatusText = "只有管理员可以创建本地账户。";
            return;
        }
        try
        {
            IsBusy = true;
            await _services.Authentication.CreateUserAsync(NewUsername, NewPassword, cancellationToken);
            NewUsername = string.Empty;
            NewPassword = string.Empty;
            StatusText = "账户已创建。";
            await RefreshAsync(cancellationToken);
        }
        catch (Exception exception)
        {
            _logger.LogError(exception, "Local user creation failed");
            StatusText = $"创建账户失败：{exception.Message}";
        }
        finally
        {
            IsBusy = false;
        }
    }

    [RelayCommand]
    private async Task ToggleLockAsync(UserAccount? user, CancellationToken cancellationToken)
    {
        if (!_shell.IsAdmin || user is null || user.Role == "admin")
        {
            return;
        }
        try
        {
            await _services.Authentication.SetLockedAsync(user.Id, !user.IsLocked, cancellationToken);
            StatusText = user.IsLocked ? $"已解锁 {user.Username}" : $"已锁定 {user.Username}";
            await RefreshAsync(cancellationToken);
        }
        catch (Exception exception)
        {
            _logger.LogError(exception, "Changing lock state failed for user {UserId}", user.Id);
            StatusText = $"更新账户状态失败：{exception.Message}";
        }
    }
}
