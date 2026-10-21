using Avalonia.Controls;

namespace Qwen3.Desktop.Views;

public sealed partial class UsersView : UserControl
{
    public UsersView() => Avalonia.Markup.Xaml.AvaloniaXamlLoader.Load(this);
}
