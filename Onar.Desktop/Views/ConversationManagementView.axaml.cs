using Avalonia.Controls;

namespace Onar.Desktop.Views;

public sealed partial class ConversationManagementView : UserControl
{
    public ConversationManagementView() => Avalonia.Markup.Xaml.AvaloniaXamlLoader.Load(this);
}
