using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Extensions.Logging;
using Qwen3.Desktop.Models;
using Qwen3.Desktop.Services;

namespace Qwen3.Desktop.ViewModels;

public partial class ProfilePageViewModel : ObservableObject
{
    private readonly DesktopServices _services;
    private readonly MainWindowViewModel _shell;
    private readonly ILogger<ProfilePageViewModel> _logger;

    [ObservableProperty]
    private UserProfile? profile;

    [ObservableProperty]
    private UserInsights? insights;

    [ObservableProperty]
    private ObservableCollection<UserDomainInsight> domains = [];

    [ObservableProperty]
    private string nickname = string.Empty;

    [ObservableProperty]
    private string avatarUrl = string.Empty;

    [ObservableProperty]
    private string birthday = string.Empty;

    [ObservableProperty]
    private string blogUrl = string.Empty;

    [ObservableProperty]
    private string bio = string.Empty;

    [ObservableProperty]
    private string currentPassword = string.Empty;

    [ObservableProperty]
    private string newPassword = string.Empty;

    [ObservableProperty]
    private string confirmNewPassword = string.Empty;

    [ObservableProperty]
    private string deletePassword = string.Empty;

    [ObservableProperty]
    private string statusText = string.Empty;

    [ObservableProperty]
    private bool isBusy;

    [ObservableProperty]
    private bool isDeleteConfirmationOpen;

    public string Initials => string.IsNullOrWhiteSpace(Profile?.Nickname)
        ? Profile?.Username[..1].ToUpperInvariant() ?? "?"
        : Profile.Nickname[..1].ToUpperInvariant();

    public string KeywordSummary => Insights is null || Insights.Keywords.Count == 0
        ? "对话积累后，这里会显示你常关注的关键词。"
        : string.Join("  ·  ", Insights.Keywords);

    public bool CanDeleteAccount => Profile?.Role != "admin";
    public bool HasNoDomainInsights => Domains.Count == 0;
    public bool IsNotBusy => !IsBusy;
    public string AccountType => Profile?.Role == "admin" ? "管理员" : "普通用户";

    public ProfilePageViewModel(DesktopServices services, MainWindowViewModel shell)
    {
        _services = services;
        _shell = shell;
        _logger = services.LoggerFactory.CreateLogger<ProfilePageViewModel>();
    }

    partial void OnIsBusyChanged(bool value) => OnPropertyChanged(nameof(IsNotBusy));

    public async Task RefreshAsync(CancellationToken cancellationToken = default)
    {
        var userId = _shell.CurrentUser?.Id;
        if (userId is null)
        {
            Profile = null;
            Insights = null;
            Domains.Clear();
            return;
        }

        var loadedProfile = await _services.Authentication.GetProfileAsync(userId, cancellationToken);
        var loadedInsights = await _services.Authentication.GetInsightsAsync(userId, cancellationToken);
        if (!string.Equals(userId, _shell.CurrentUser?.Id, StringComparison.Ordinal))
        {
            return;
        }

        Profile = loadedProfile;
        Insights = loadedInsights;
        Nickname = loadedProfile.Nickname.Length == 0 ? loadedProfile.Username : loadedProfile.Nickname;
        AvatarUrl = loadedProfile.AvatarUrl;
        Birthday = loadedProfile.Birthday;
        BlogUrl = loadedProfile.BlogUrl;
        Bio = loadedProfile.Bio;
        Domains = new ObservableCollection<UserDomainInsight>(loadedInsights.Domains);
        OnPropertyChanged(nameof(Initials));
        OnPropertyChanged(nameof(KeywordSummary));
        OnPropertyChanged(nameof(CanDeleteAccount));
        OnPropertyChanged(nameof(HasNoDomainInsights));
        OnPropertyChanged(nameof(AccountType));
    }

    [RelayCommand]
    private async Task SaveProfileAsync(CancellationToken cancellationToken)
    {
        if (!TryGetCurrentUserId(out var userId) || IsBusy)
        {
            StatusText = "请登录后修改个人资料。";
            return;
        }
        IsBusy = true;
        try
        {
            await _services.Authentication.UpdateProfileAsync(
                userId, Nickname, AvatarUrl, Birthday, BlogUrl, Bio, cancellationToken);
            StatusText = "个人资料已保存。";
            await RefreshAsync(cancellationToken);
        }
        catch (Exception exception)
        {
            _logger.LogError(exception, "Saving profile failed for {UserId}", userId);
            StatusText = $"保存失败：{exception.Message}";
        }
        finally
        {
            IsBusy = false;
        }
    }

    [RelayCommand]
    private async Task ChangePasswordAsync(CancellationToken cancellationToken)
    {
        if (!TryGetCurrentUserId(out var userId) || IsBusy)
        {
            StatusText = "请登录后修改密码。";
            return;
        }
        if (!string.Equals(NewPassword, ConfirmNewPassword, StringComparison.Ordinal))
        {
            StatusText = "两次输入的新密码不一致。";
            return;
        }
        IsBusy = true;
        try
        {
            await _services.Authentication.ChangePasswordAsync(
                userId, CurrentPassword, NewPassword, cancellationToken);
            CurrentPassword = string.Empty;
            NewPassword = string.Empty;
            ConfirmNewPassword = string.Empty;
            StatusText = "密码已更新。";
        }
        catch (Exception exception)
        {
            _logger.LogWarning(exception, "Changing password failed for {UserId}", userId);
            StatusText = $"修改密码失败：{exception.Message}";
        }
        finally
        {
            IsBusy = false;
        }
    }

    [RelayCommand]
    private void OpenDeleteConfirmation()
    {
        if (CanDeleteAccount && !IsBusy)
        {
            DeletePassword = string.Empty;
            IsDeleteConfirmationOpen = true;
        }
    }

    [RelayCommand]
    private void CancelDelete() => IsDeleteConfirmationOpen = false;

    [RelayCommand]
    private async Task ConfirmDeleteAccountAsync(CancellationToken cancellationToken)
    {
        if (!TryGetCurrentUserId(out var userId) || !CanDeleteAccount || IsBusy)
        {
            StatusText = "当前账户不能自主注销。";
            IsDeleteConfirmationOpen = false;
            return;
        }
        IsBusy = true;
        try
        {
            await _services.Authentication.DeleteOwnAccountAsync(userId, DeletePassword, cancellationToken);
            IsDeleteConfirmationOpen = false;
            DeletePassword = string.Empty;
            await _shell.CompleteSelfServiceAccountDeletionAsync(cancellationToken);
        }
        catch (Exception exception)
        {
            _logger.LogWarning(exception, "Self-service account deletion failed for {UserId}", userId);
            StatusText = $"注销失败：{exception.Message}";
        }
        finally
        {
            IsBusy = false;
        }
    }

    private bool TryGetCurrentUserId(out string userId)
    {
        userId = _shell.CurrentUser?.Id ?? string.Empty;
        return userId.Length > 0;
    }
}
