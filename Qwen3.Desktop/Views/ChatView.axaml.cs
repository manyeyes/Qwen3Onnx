using Avalonia.Controls;
using Qwen3.Desktop.ViewModels;

namespace Qwen3.Desktop.Views;

public sealed partial class ChatView : UserControl
{
    public ChatView() => Avalonia.Markup.Xaml.AvaloniaXamlLoader.Load(this);

    private void ThinkingScrollViewer_OnScrollChanged(object? sender, ScrollChangedEventArgs eventArgs)
    {
        if (sender is ScrollViewer { DataContext: MessageViewModel { IsPending: true } } scrollViewer)
        {
            scrollViewer.ScrollToEnd();
        }
    }
}
