using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Qwen3.Desktop.Models;
using Qwen3.Desktop.Services;

namespace Qwen3.Desktop.ViewModels;

public partial class HistoryPageViewModel : ObservableObject
{
    private readonly DesktopServices _services;
    private readonly MainWindowViewModel _shell;
    private int _identityVersion;

    [ObservableProperty]
    private ObservableCollection<ConversationSummary> recentConversations = [];

    [ObservableProperty]
    private ObservableCollection<HistoryGroupViewModel> groups = [];

    [ObservableProperty]
    private ObservableCollection<ConversationSummary> filteredConversations = [];

    [ObservableProperty]
    private ConversationSummary? selectedConversation;

    [ObservableProperty]
    private string searchText = string.Empty;

    [ObservableProperty]
    private string titleDraft = string.Empty;

    [ObservableProperty]
    private string statusText = string.Empty;

    [ObservableProperty]
    private bool isRenameDialogOpen;

    [ObservableProperty]
    private bool isDeleteConfirmationOpen;

    public bool CanManage => _shell.IsLoggedIn;
    public bool CanEditSelected => CanManage && SelectedConversation is not null;
    public bool HasNoFilteredConversations => FilteredConversations.Count == 0;

    public HistoryPageViewModel(DesktopServices services, MainWindowViewModel shell)
    {
        _services = services;
        _shell = shell;
    }

    public async Task RefreshAsync(CancellationToken cancellationToken = default)
    {
        var ownerId = _shell.CurrentUser?.Id;
        var identityVersion = Volatile.Read(ref _identityVersion);
        var items = await _services.Database.GetConversationsAsync(ownerId, cancellationToken);
        if (identityVersion != Volatile.Read(ref _identityVersion) ||
            !string.Equals(ownerId, _shell.CurrentUser?.Id, StringComparison.Ordinal))
        {
            return;
        }
        var ordered = items.OrderByDescending(item => item.IsPinned)
            .ThenByDescending(item => item.UpdatedAt).ToArray();
        var recent = ordered.Take(100).ToArray();
        RecentConversations = new ObservableCollection<ConversationSummary>(recent);
        Groups = BuildRecentGroups(recent.Select(item => new SidebarConversationViewModel(item, this)).ToArray());
        ApplySearch(ordered);
    }

    partial void OnSearchTextChanged(string value) => _ = RefreshAsync();

    partial void OnFilteredConversationsChanged(ObservableCollection<ConversationSummary> value) =>
        OnPropertyChanged(nameof(HasNoFilteredConversations));

    partial void OnSelectedConversationChanged(ConversationSummary? value)
    {
        TitleDraft = value?.Title ?? string.Empty;
        OnPropertyChanged(nameof(CanEditSelected));
    }

    [RelayCommand]
    private async Task OpenConversationAsync(ConversationSummary? conversation)
    {
        if (conversation is null)
        {
            return;
        }
        var ownerId = _shell.CurrentUser?.Id;
        var identityVersion = Volatile.Read(ref _identityVersion);
        try
        {
            await _shell.Chat.LoadConversationAsync(conversation, ownerId);
            if (identityVersion != Volatile.Read(ref _identityVersion) ||
                !string.Equals(ownerId, _shell.CurrentUser?.Id, StringComparison.Ordinal))
            {
                return;
            }
            _shell.Navigate("Chat");
        }
        catch (Exception exception)
        {
            _shell.StatusMessage = exception.Message;
        }
    }

    [RelayCommand]
    private async Task RenameSelectedAsync(CancellationToken cancellationToken)
    {
        if (SelectedConversation is null)
        {
            StatusText = "请选择需要重命名的对话。";
            return;
        }
        try
        {
            await _services.Database.RenameConversationAsync(
                SelectedConversation.Id, _shell.CurrentUser?.Id, TitleDraft, cancellationToken);
            StatusText = "对话标题已更新。";
            IsRenameDialogOpen = false;
            var conversationId = SelectedConversation.Id;
            await RefreshAsync(cancellationToken);
            SelectedConversation = RecentConversations.FirstOrDefault(item =>
                item.Id == conversationId);
            if (_shell.Chat.ConversationId == conversationId)
            {
                _shell.Chat.ConversationTitle = SelectedConversation?.Title ?? TitleDraft.Trim();
            }
        }
        catch (Exception exception)
        {
            StatusText = $"修改标题失败：{exception.Message}";
        }
    }

    [RelayCommand]
    private async Task DeleteConversationAsync(
        ConversationSummary? conversation, CancellationToken cancellationToken)
    {
        if (conversation is null)
        {
            return;
        }
        try
        {
            await _services.Database.DeleteConversationAsync(
                conversation.Id, _shell.CurrentUser?.Id, cancellationToken);
            if (_shell.Chat.ConversationId == conversation.Id)
            {
                _shell.Chat.StartNewConversation();
            }
            StatusText = "对话已删除。";
            IsDeleteConfirmationOpen = false;
            await RefreshAsync(cancellationToken);
        }
        catch (Exception exception)
        {
            StatusText = $"删除对话失败：{exception.Message}";
        }
    }

    internal void RequestRename(ConversationSummary conversation)
    {
        SelectedConversation = conversation;
        TitleDraft = conversation.Title;
        IsRenameDialogOpen = true;
    }

    internal void RequestDelete(ConversationSummary conversation)
    {
        SelectedConversation = conversation;
        IsDeleteConfirmationOpen = true;
    }

    [RelayCommand]
    private Task ConfirmDeleteRequestedAsync(CancellationToken cancellationToken) =>
        DeleteConversationAsync(SelectedConversation, cancellationToken);

    internal async Task TogglePinnedAsync(
        ConversationSummary conversation, CancellationToken cancellationToken = default)
    {
        try
        {
            await _services.Database.SetConversationPinnedAsync(
                conversation.Id, _shell.CurrentUser?.Id, !conversation.IsPinned, cancellationToken);
            await RefreshAsync(cancellationToken);
        }
        catch (Exception exception)
        {
            StatusText = $"置顶状态更新失败：{exception.Message}";
        }
    }

    internal async Task OpenSidebarConversationAsync(ConversationSummary conversation)
    {
        await OpenConversationAsync(conversation);
    }

    [RelayCommand]
    private void CancelRename() => IsRenameDialogOpen = false;

    [RelayCommand]
    private void CancelDelete() => IsDeleteConfirmationOpen = false;

    public void NotifyIdentityChanged()
    {
        Interlocked.Increment(ref _identityVersion);
        OnPropertyChanged(nameof(CanManage));
        OnPropertyChanged(nameof(CanEditSelected));
        RecentConversations.Clear();
        Groups.Clear();
        FilteredConversations.Clear();
        SelectedConversation = null;
        IsRenameDialogOpen = false;
        IsDeleteConfirmationOpen = false;
        StatusText = string.Empty;
    }

    private void ApplySearch(IEnumerable<ConversationSummary> conversations)
    {
        var term = SearchText.Trim();
        var filtered = string.IsNullOrEmpty(term)
            ? conversations
            : conversations.Where(item => item.Title.Contains(term, StringComparison.OrdinalIgnoreCase));
        FilteredConversations = new ObservableCollection<ConversationSummary>(filtered);
    }

    private static ObservableCollection<HistoryGroupViewModel> BuildRecentGroups(
        IReadOnlyList<SidebarConversationViewModel> conversations)
    {
        var now = DateTimeOffset.Now;
        var pinned = conversations.Where(item => item.IsPinned).ToArray();
        var unpinned = conversations.Where(item => !item.IsPinned).ToArray();
        var recent = unpinned.Where(item => item.UpdatedAt.ToLocalTime() >= now.AddDays(-7));
        var lastMonth = unpinned.Where(item =>
            item.UpdatedAt.ToLocalTime() < now.AddDays(-7) &&
            item.UpdatedAt.ToLocalTime() >= now.AddDays(-30));
        var older = unpinned.Where(item => item.UpdatedAt.ToLocalTime() < now.AddDays(-30))
            .GroupBy(item => item.UpdatedAt.ToLocalTime().ToString("yyyy年MM月"))
            .OrderByDescending(group => group.Key, StringComparer.Ordinal)
            .Select(group => new HistoryGroupViewModel(group.Key, group));

        var groups = new List<HistoryGroupViewModel>();
        if (pinned.Length > 0)
        {
            groups.Add(new HistoryGroupViewModel("置顶", pinned));
        }
        if (recent.Any())
        {
            groups.Add(new HistoryGroupViewModel("7 天内", recent));
        }
        if (lastMonth.Any())
        {
            groups.Add(new HistoryGroupViewModel("30 天内", lastMonth));
        }
        groups.AddRange(older);
        return new ObservableCollection<HistoryGroupViewModel>(groups);
    }
}
