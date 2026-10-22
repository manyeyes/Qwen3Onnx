using System.Collections.ObjectModel;

namespace Qwen3.Desktop.ViewModels;

public sealed class HistoryGroupViewModel(
    string title, IEnumerable<SidebarConversationViewModel> conversations)
{
    public string Title { get; } = title;
    public ObservableCollection<SidebarConversationViewModel> Conversations { get; } =
        new(conversations);
}
