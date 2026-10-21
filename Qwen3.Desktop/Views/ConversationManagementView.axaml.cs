using Avalonia.Controls;

namespace Qwen3.Desktop.Views;

public sealed partial class ConversationManagementView : UserControl
{
    public ConversationManagementView() => Avalonia.Markup.Xaml.AvaloniaXamlLoader.Load(this);
}
