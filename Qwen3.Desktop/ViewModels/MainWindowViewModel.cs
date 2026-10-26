using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Extensions.Logging;
using System.Diagnostics;
using Qwen3.Desktop.Models;
using Qwen3.Desktop.Services;

namespace Qwen3.Desktop.ViewModels;

public partial class MainWindowViewModel : ObservableObject
{
    private readonly DesktopServices _services;
    private readonly ILogger<MainWindowViewModel> _logger;
    private double _expandedSidebarWidth = 250;

    [ObservableProperty]
    private object currentViewModel = null!;

    [ObservableProperty]
    private UserAccount? currentUser;

    [ObservableProperty]
    private bool isSidebarExpanded = true;

    [ObservableProperty]
    private double sidebarWidth = 250;

    [ObservableProperty]
    private bool isAuthenticationOpen;

    [ObservableProperty]
    private bool isBottomMenuOpen;

    [ObservableProperty]
    private bool isBootstrapNoticeVisible;

    [ObservableProperty]
    private string loginUsername = string.Empty;

    [ObservableProperty]
    private string loginPassword = string.Empty;

    [ObservableProperty]
    private string registerPassword = string.Empty;

    [ObservableProperty]
    private string registerConfirmPassword = string.Empty;

    [ObservableProperty]
    private bool isRegisterMode;

    [ObservableProperty]
    private string statusMessage = string.Empty;

    [ObservableProperty]
    private string? generatedAdminPassword;

    public ChatPageViewModel Chat { get; }
    public HistoryPageViewModel History { get; }
    public ModelsPageViewModel Models { get; }
    public UsersPageViewModel Users { get; }
    public ProfilePageViewModel Profile { get; }
    public AboutPageViewModel About { get; }
    public Task Initialization { get; }
    public bool IsAdmin => CurrentUser?.Role == "admin";
    public bool IsGuest => CurrentUser is null;
    public bool IsLoggedIn => CurrentUser is not null;
    public bool IsLoginMode => !IsRegisterMode;
    public bool CanChangeIdentity => !Chat.IsBusy;
    public string IdentityLabel => CurrentUser is null ? "访客模式" : $"账户：{CurrentUser.Username}";
    public string IdentityInitial => string.IsNullOrEmpty(CurrentUser?.Username)
        ? "访"
        : CurrentUser.Username[..1].ToUpperInvariant();
    public string BottomMenuGlyph => IsBottomMenuOpen ? "⌄" : "⌃";
    public string AuthenticationTitle => IsRegisterMode ? "创建本地账户" : "本地账户登录";
    public string AuthenticationSubmitText => IsRegisterMode ? "注册并登录" : "登录";
    public string AuthenticationToggleText => IsRegisterMode ? "已有账户？返回登录" : "没有账户？立即注册";
    public MainWindowViewModel(DesktopServices services, BootstrapResult bootstrap)
    {
        _services = services;
        _logger = services.LoggerFactory.CreateLogger<MainWindowViewModel>();
        Chat = new ChatPageViewModel(services, this);
        History = new HistoryPageViewModel(services, this);
        Models = new ModelsPageViewModel(services, this);
        Users = new UsersPageViewModel(services, this);
        Profile = new ProfilePageViewModel(services, this);
        About = new AboutPageViewModel();
        CurrentViewModel = Chat;
        GeneratedAdminPassword = bootstrap.GeneratedAdminPassword;
        IsBootstrapNoticeVisible = GeneratedAdminPassword is not null;
        Chat.ConversationsChanged = () => _ = RefreshHistoryAsync();
        Initialization = InitializeAsync();
    }

    public void Navigate(string page)
    {
        switch (page)
        {
            case "Chat":
                CurrentViewModel = Chat;
                _ = Chat.RefreshModelsAsync();
                break;
            case "Conversations" when IsLoggedIn:
                CurrentViewModel = History;
                _ = History.RefreshAsync();
                break;
            case "Models":
                CurrentViewModel = Models;
                _ = Models.RefreshAsync();
                break;
            case "Users" when IsAdmin:
                CurrentViewModel = Users;
                _ = Users.RefreshAsync();
                break;
            case "Profile" when IsLoggedIn:
                CurrentViewModel = Profile;
                _ = Profile.RefreshAsync();
                break;
            case "About":
                CurrentViewModel = About;
                break;
            default:
                StatusMessage = "当前账户无权打开此页面。";
                return;
        }

        IsBottomMenuOpen = false;
        StatusMessage = string.Empty;
    }

    partial void OnIsSidebarExpandedChanged(bool value) =>
        SidebarWidth = value ? _expandedSidebarWidth : 72;

    partial void OnSidebarWidthChanged(double value)
    {
        if (IsSidebarExpanded)
        {
            _expandedSidebarWidth = value;
        }
    }

    public void SetSidebarWidth(double width)
    {
        if (IsSidebarExpanded)
        {
            SidebarWidth = Math.Clamp(width, 200, 440);
        }
    }
    partial void OnIsBottomMenuOpenChanged(bool value) => OnPropertyChanged(nameof(BottomMenuGlyph));
    partial void OnIsRegisterModeChanged(bool value)
    {
        OnPropertyChanged(nameof(AuthenticationTitle));
        OnPropertyChanged(nameof(AuthenticationSubmitText));
        OnPropertyChanged(nameof(AuthenticationToggleText));
        OnPropertyChanged(nameof(IsLoginMode));
    }

    partial void OnCurrentUserChanged(UserAccount? value)
    {
        OnPropertyChanged(nameof(IsAdmin));
        OnPropertyChanged(nameof(IsGuest));
        OnPropertyChanged(nameof(IsLoggedIn));
        OnPropertyChanged(nameof(IdentityLabel));
        OnPropertyChanged(nameof(IdentityInitial));
        Models.NotifyAdminStatusChanged();
        History.NotifyIdentityChanged();
        if (!IsLoggedIn && ReferenceEquals(CurrentViewModel, History))
        {
            CurrentViewModel = Chat;
        }
        if (!IsLoggedIn && ReferenceEquals(CurrentViewModel, Profile))
        {
            CurrentViewModel = Chat;
        }
    }

    public void OnChatBusyChanged()
    {
        OnPropertyChanged(nameof(CanChangeIdentity));
        OpenAuthenticationCommand.NotifyCanExecuteChanged();
        SignOutCommand.NotifyCanExecuteChanged();
        NewConversationCommand.NotifyCanExecuteChanged();
    }

    private async Task InitializeAsync()
    {
        CurrentUser = await _services.Authentication.RestoreSessionAsync();
        await Chat.ChangeIdentityAsync();
        await History.RefreshAsync();
        await Models.RefreshAsync();
    }

    private async Task RefreshHistoryAsync()
    {
        try
        {
            await History.RefreshAsync();
        }
        catch (Exception exception)
        {
            _logger.LogError(exception, "Refreshing conversation history failed");
            StatusMessage = $"刷新历史失败：{exception.Message}";
        }
    }

    [RelayCommand]
    private void ToggleSidebar() => IsSidebarExpanded = !IsSidebarExpanded;

    [RelayCommand(CanExecute = nameof(CanChangeIdentity))]
    private void NewConversation()
    {
        Chat.StartNewConversation();
        Navigate("Chat");
    }

    [RelayCommand]
    private void NavigateTo(string page) => Navigate(page);

    [RelayCommand(CanExecute = nameof(CanChangeIdentity))]
    private void OpenAuthentication()
    {
        IsBottomMenuOpen = false;
        IsAuthenticationOpen = true;
        IsRegisterMode = false;
        StatusMessage = string.Empty;
    }

    [RelayCommand]
    private void CloseAuthentication() => IsAuthenticationOpen = false;

    [RelayCommand]
    private void ToggleAuthenticationMode()
    {
        IsRegisterMode = !IsRegisterMode;
        LoginPassword = string.Empty;
        RegisterPassword = string.Empty;
        RegisterConfirmPassword = string.Empty;
        StatusMessage = string.Empty;
    }

    [RelayCommand]
    private void ToggleBottomMenu() => IsBottomMenuOpen = !IsBottomMenuOpen;

    [RelayCommand]
    private void OpenAboutPage() => Navigate("About");

    [RelayCommand]
    private void OpenWebsite()
    {
        IsBottomMenuOpen = false;
        try
        {
            Process.Start(new ProcessStartInfo("https://manyeyes.github.io/manyspeech/")
            {
                UseShellExecute = true
            });
        }
        catch (Exception exception)
        {
            _logger.LogError(exception, "Opening the official website failed");
            StatusMessage = $"无法打开官网：{exception.Message}";
        }
    }

    [RelayCommand]
    private void DismissBootstrapNotice()
    {
        IsBootstrapNoticeVisible = false;
        GeneratedAdminPassword = null;
    }

    [RelayCommand]
    private async Task SignInAsync(CancellationToken cancellationToken)
    {
        try
        {
            var user = await _services.Authentication.SignInAsync(
                LoginUsername, LoginPassword, cancellationToken);
            if (user is null)
            {
                StatusMessage = "用户名或密码错误，或账户已锁定。";
                return;
            }
            await _services.Authentication.SaveSessionAsync(user.Id, cancellationToken);
            CurrentUser = user;
            _services.Chat.ClearConversationState();
            LoginPassword = string.Empty;
            IsAuthenticationOpen = false;
            await Chat.ChangeIdentityAsync(cancellationToken);
            await History.RefreshAsync(cancellationToken);
            Navigate("Chat");
        }
        catch (Exception exception)
        {
            _logger.LogWarning(exception, "Local sign-in failed");
            StatusMessage = $"登录失败：{exception.Message}";
        }
    }

    [RelayCommand]
    private async Task RegisterAsync(CancellationToken cancellationToken)
    {
        if (!string.Equals(RegisterPassword, RegisterConfirmPassword, StringComparison.Ordinal))
        {
            StatusMessage = "两次输入的密码不一致。";
            return;
        }

        try
        {
            var userId = await _services.Authentication.CreateUserAsync(
                LoginUsername, RegisterPassword, cancellationToken);
            var user = await _services.Authentication.SignInAsync(
                LoginUsername, RegisterPassword, cancellationToken);
            if (user is null || !string.Equals(user.Id, userId, StringComparison.Ordinal))
            {
                throw new InvalidOperationException("账户已创建，但无法自动登录，请使用新账户重新登录。");
            }

            await _services.Authentication.SaveSessionAsync(user.Id, cancellationToken);
            CurrentUser = user;
            _services.Chat.ClearConversationState();
            LoginPassword = string.Empty;
            RegisterPassword = string.Empty;
            RegisterConfirmPassword = string.Empty;
            IsAuthenticationOpen = false;
            await Chat.ChangeIdentityAsync(cancellationToken);
            await History.RefreshAsync(cancellationToken);
            Navigate("Chat");
            StatusMessage = "账户已创建并登录。可在用户中心补充个人资料。";
        }
        catch (Exception exception)
        {
            _logger.LogWarning(exception, "Local account registration failed");
            StatusMessage = $"注册失败：{exception.Message}";
        }
    }

    public async Task CompleteSelfServiceAccountDeletionAsync(
        CancellationToken cancellationToken = default)
    {
        CurrentUser = null;
        IsAuthenticationOpen = false;
        _services.Chat.ClearConversationState();
        await Chat.ChangeIdentityAsync(cancellationToken);
        await History.RefreshAsync(cancellationToken);
        await Profile.RefreshAsync(cancellationToken);
        Navigate("Chat");
        StatusMessage = "账户及其私有对话已注销。";
    }

    [RelayCommand(CanExecute = nameof(CanChangeIdentity))]
    private async Task SignOutAsync(CancellationToken cancellationToken)
    {
        IsBottomMenuOpen = false;
        try
        {
            await _services.Authentication.ClearSessionAsync(cancellationToken);
        }
        catch (Exception exception)
        {
            _logger.LogError(exception, "Clearing the persistent login session failed");
            StatusMessage = $"退出登录失败：{exception.Message}";
            return;
        }
        CurrentUser = null;
        _services.Chat.ClearConversationState();
        await Chat.ChangeIdentityAsync(cancellationToken);
        await History.RefreshAsync(cancellationToken);
        Navigate("Chat");
    }
}
