using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Qwen3.Desktop.Models;

namespace Qwen3.Desktop.ViewModels;

public partial class SidebarConversationViewModel : ObservableObject
{
    private readonly HistoryPageViewModel _history;
    private ConversationSummary _conversation;

    public string Id => _conversation.Id;
    public string Title => _conversation.Title;
    public DateTimeOffset UpdatedAt => _conversation.UpdatedAt;
    public bool IsPinned => _conversation.IsPinned;
    public string PinMenuText => IsPinned ? "取消置顶" : "置顶";

    public SidebarConversationViewModel(
        ConversationSummary conversation, HistoryPageViewModel history)
    {
        _conversation = conversation;
        _history = history;
    }

    [RelayCommand]
    private Task OpenAsync() => _history.OpenSidebarConversationAsync(_conversation);

    [RelayCommand]
    private void Rename()
    {
        _history.RequestRename(_conversation);
    }

    [RelayCommand]
    private void Delete()
    {
        _history.RequestDelete(_conversation);
    }

    [RelayCommand]
    private async Task TogglePinAsync()
    {
        await _history.TogglePinnedAsync(_conversation);
    }
}
