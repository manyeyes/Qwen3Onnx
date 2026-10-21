using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Extensions.Logging;
using Qwen3.Desktop.Models;
using Qwen3.Desktop.Services;

namespace Qwen3.Desktop.ViewModels;

public partial class MainWindowViewModel : ObservableObject
{
    private readonly DesktopServices _services;
    private readonly ILogger<MainWindowViewModel> _logger;

    [ObservableProperty]
    private object currentViewModel = null!;

    [ObservableProperty]
    private UserAccount? currentUser;

    [ObservableProperty]
    private bool isSidebarExpanded = true;

    [ObservableProperty]
    private bool isAuthenticationOpen;

    [ObservableProperty]
    private bool isBootstrapNoticeVisible;

    [ObservableProperty]
    private string loginUsername = string.Empty;

    [ObservableProperty]
    private string loginPassword = string.Empty;

    [ObservableProperty]
    private string statusMessage = string.Empty;

    [ObservableProperty]
    private string? generatedAdminPassword;

    public ChatPageViewModel Chat { get; }
    public HistoryPageViewModel History { get; }
    public ModelsPageViewModel Models { get; }
    public UsersPageViewModel Users { get; }
    public Task Initialization { get; }
    public bool IsAdmin => CurrentUser?.Role == "admin";
    public bool IsGuest => CurrentUser is null;
    public bool IsLoggedIn => CurrentUser is not null;
    public bool CanChangeIdentity => !Chat.IsBusy;
    public string IdentityLabel => CurrentUser is null ? "访客模式" : $"账户：{CurrentUser.Username}";
    public double SidebarWidth => IsSidebarExpanded ? 250 : 72;

    public MainWindowViewModel(DesktopServices services, BootstrapResult bootstrap)
    {
        _services = services;
        _logger = services.LoggerFactory.CreateLogger<MainWindowViewModel>();
        Chat = new ChatPageViewModel(services, this);
        History = new HistoryPageViewModel(services, this);
        Models = new ModelsPageViewModel(services, this);
        Users = new UsersPageViewModel(services, this);
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
            default:
                StatusMessage = "当前账户无权打开此页面。";
                return;
        }

        StatusMessage = string.Empty;
    }

    partial void OnIsSidebarExpandedChanged(bool value) => OnPropertyChanged(nameof(SidebarWidth));

    partial void OnCurrentUserChanged(UserAccount? value)
    {
        OnPropertyChanged(nameof(IsAdmin));
        OnPropertyChanged(nameof(IsGuest));
        OnPropertyChanged(nameof(IsLoggedIn));
        OnPropertyChanged(nameof(IdentityLabel));
        Models.NotifyAdminStatusChanged();
        History.NotifyIdentityChanged();
        if (!IsLoggedIn && ReferenceEquals(CurrentViewModel, History))
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
        IsAuthenticationOpen = true;
        StatusMessage = string.Empty;
    }

    [RelayCommand]
    private void CloseAuthentication() => IsAuthenticationOpen = false;

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
            CurrentUser = user;
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

    [RelayCommand(CanExecute = nameof(CanChangeIdentity))]
    private async Task SignOutAsync(CancellationToken cancellationToken)
    {
        CurrentUser = null;
        await Chat.ChangeIdentityAsync(cancellationToken);
        await History.RefreshAsync(cancellationToken);
        Navigate("Chat");
    }
}
