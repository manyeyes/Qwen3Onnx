using System.Collections.ObjectModel;
using Qwen3.Desktop.Models;

namespace Qwen3.Desktop.ViewModels;

public sealed class HistoryGroupViewModel(string title, IEnumerable<ConversationSummary> conversations)
{
    public string Title { get; } = title;
    public ObservableCollection<ConversationSummary> Conversations { get; } = new(conversations);
}
